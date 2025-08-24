#!/usr/bin/env python3
"""
Real-time Traffic Dashboard

FastAPI-based web dashboard for visualizing real-time traffic classification
with WebSocket updates and interactive monitoring capabilities.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import psutil

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from ..utils.logging import setup_logger


class TrafficDashboard:
    """
    Real-time web dashboard for traffic classification monitoring.
    
    Provides live visualization of traffic patterns, model predictions,
    system performance, and network statistics.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        """
        Initialize traffic dashboard.
        
        Args:
            host: Host address to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.logger = setup_logger("traffic_dashboard")
        
        # FastAPI app
        self.app = FastAPI(title="Real-time Traffic Detection Dashboard")
        
        # WebSocket connections
        self.connections: List[WebSocket] = []
        
        # Data storage for dashboard
        self.metrics_history = []
        self.prediction_history = []
        self.system_stats = {}
        
        # Configuration
        self.max_history_size = 1000
        self.update_interval = 1.0  # seconds
        
        # Setup routes and WebSocket handlers
        self._setup_routes()
        self._setup_websocket()
        
        # Background task for system monitoring
        self.monitoring_task = None
    
    def _setup_routes(self):
        """Setup HTTP routes for the dashboard."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Main dashboard page."""
            return self._get_dashboard_html()
        
        @self.app.get("/api/status")
        async def get_status():
            """Get current system status."""
            return {
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "connections": len(self.connections),
                "metrics_count": len(self.metrics_history)
            }
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get latest metrics data."""
            return {
                "metrics": self.metrics_history[-100:],  # Last 100 points
                "predictions": self.prediction_history[-100:],
                "system_stats": self.system_stats
            }
    
    def _setup_websocket(self):
        """Setup WebSocket endpoints for real-time updates."""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self._handle_websocket_connection(websocket)
    
    async def _handle_websocket_connection(self, websocket: WebSocket):
        """Handle WebSocket connection lifecycle."""
        await websocket.accept()
        self.connections.append(websocket)
        self.logger.info(f"WebSocket connection established. Total connections: {len(self.connections)}")
        
        try:
            # Send initial data
            await self._send_initial_data(websocket)
            
            # Keep connection alive and handle messages
            while True:
                try:
                    # Wait for client message or timeout
                    message = await asyncio.wait_for(
                        websocket.receive_text(), 
                        timeout=30.0
                    )
                    
                    # Handle client messages (ping, commands, etc.)
                    await self._handle_client_message(websocket, message)
                    
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await websocket.send_text(json.dumps({
                        "type": "ping",
                        "timestamp": datetime.now().isoformat()
                    }))
                
        except WebSocketDisconnect:
            self.logger.info("WebSocket connection closed by client")
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
        finally:
            if websocket in self.connections:
                self.connections.remove(websocket)
            self.logger.info(f"WebSocket connection removed. Total connections: {len(self.connections)}")
    
    async def _send_initial_data(self, websocket: WebSocket):
        """Send initial dashboard data to new connections."""
        initial_data = {
            "type": "initial_data",
            "metrics": self.metrics_history[-50:],  # Last 50 points
            "predictions": self.prediction_history[-50:],
            "system_stats": self.system_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        await websocket.send_text(json.dumps(initial_data))
    
    async def _handle_client_message(self, websocket: WebSocket, message: str):
        """Handle messages from WebSocket clients."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
            elif message_type == "request_update":
                await self._send_latest_update(websocket)
            
        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON message: {message}")
    
    async def _send_latest_update(self, websocket: WebSocket):
        """Send latest data update to a specific connection."""
        update_data = {
            "type": "update",
            "latest_metrics": self.metrics_history[-1:],
            "latest_predictions": self.prediction_history[-1:],
            "system_stats": self.system_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        await websocket.send_text(json.dumps(update_data))
    
    async def broadcast_update(self, data: Dict[str, Any]):
        """
        Broadcast update to all connected WebSocket clients.
        
        Args:
            data: Data to broadcast
        """
        if not self.connections:
            return
        
        message = json.dumps({
            "type": "broadcast",
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send to all connections (remove failed ones)
        failed_connections = []
        for connection in self.connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                self.logger.warning(f"Failed to send to WebSocket connection: {e}")
                failed_connections.append(connection)
        
        # Remove failed connections
        for failed_conn in failed_connections:
            if failed_conn in self.connections:
                self.connections.remove(failed_conn)
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """
        Update dashboard metrics.
        
        Args:
            metrics: Metrics dictionary to add
        """
        # Add timestamp
        metrics["timestamp"] = datetime.now().isoformat()
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Limit history size
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)
        
        # Schedule broadcast (non-blocking)
        asyncio.create_task(self.broadcast_update({
            "type": "metrics_update",
            "metrics": metrics
        }))
    
    def update_prediction(self, prediction: Dict[str, Any]):
        """
        Update dashboard with new prediction.
        
        Args:
            prediction: Prediction result dictionary
        """
        # Add timestamp
        prediction["timestamp"] = datetime.now().isoformat()
        
        # Add to history
        self.prediction_history.append(prediction)
        
        # Limit history size
        if len(self.prediction_history) > self.max_history_size:
            self.prediction_history.pop(0)
        
        # Schedule broadcast (non-blocking)
        asyncio.create_task(self.broadcast_update({
            "type": "prediction_update",
            "prediction": prediction
        }))
    
    def _update_system_stats(self):
        """Update system performance statistics."""
        try:
            self.system_stats = {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_percent": psutil.virtual_memory().percent,
                "network_io": psutil.net_io_counters()._asdict(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.warning(f"Failed to update system stats: {e}")
    
    async def _system_monitoring_loop(self):
        """Background loop for system monitoring."""
        while True:
            try:
                self._update_system_stats()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    def start_monitoring(self):
        """Start background system monitoring."""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._system_monitoring_loop())
            self.logger.info("Started system monitoring")
    
    def stop_monitoring(self):
        """Stop background system monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
            self.logger.info("Stopped system monitoring")
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML page."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-time Traffic Detection Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5; 
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
        }
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
            color: #333; 
        }
        .stats-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px; 
        }
        .stat-card { 
            background: white; 
            padding: 20px; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        }
        .stat-value { 
            font-size: 2em; 
            font-weight: bold; 
            color: #007bff; 
        }
        .stat-label { 
            color: #666; 
            margin-top: 5px; 
        }
        .charts-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
            gap: 20px; 
        }
        .chart-container { 
            background: white; 
            padding: 20px; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        }
        .status-indicator { 
            width: 12px; 
            height: 12px; 
            border-radius: 50%; 
            display: inline-block; 
            margin-right: 8px; 
        }
        .status-online { background-color: #28a745; }
        .status-offline { background-color: #dc3545; }
        .prediction-badge { 
            padding: 4px 8px; 
            border-radius: 4px; 
            color: white; 
            font-weight: bold; 
        }
        .reel-traffic { background-color: #ff6b6b; }
        .non-reel-traffic { background-color: #4ecdc4; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Real-time Traffic Detection Dashboard</h1>
            <p>
                <span class="status-indicator" id="connection-status"></span>
                <span id="connection-text">Connecting...</span>
            </p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="total-packets">0</div>
                <div class="stat-label">Total Packets</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="reel-percentage">0%</div>
                <div class="stat-label">Reel Traffic</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="confidence-score">0%</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="throughput">0 MB/s</div>
                <div class="stat-label">Throughput</div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <h3>Traffic Classification</h3>
                <canvas id="classificationChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Confidence Scores</h3>
                <canvas id="confidenceChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>System Performance</h3>
                <canvas id="systemChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Recent Predictions</h3>
                <div id="predictions-list"></div>
            </div>
        </div>
    </div>

    <script>
        // WebSocket connection
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        // Chart configurations
        const chartConfig = {
            responsive: true,
            scales: {
                x: { display: true },
                y: { display: true }
            },
            animation: { duration: 0 }
        };
        
        // Initialize charts
        const classificationChart = new Chart(
            document.getElementById('classificationChart'),
            {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Reel Traffic %',
                        data: [],
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        tension: 0.4
                    }]
                },
                options: chartConfig
            }
        );
        
        const confidenceChart = new Chart(
            document.getElementById('confidenceChart'),
            {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Confidence Score',
                        data: [],
                        borderColor: '#4ecdc4',
                        backgroundColor: 'rgba(78, 205, 196, 0.1)',
                        tension: 0.4
                    }]
                },
                options: chartConfig
            }
        );
        
        const systemChart = new Chart(
            document.getElementById('systemChart'),
            {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'CPU %',
                            data: [],
                            borderColor: '#ff9f43',
                            backgroundColor: 'rgba(255, 159, 67, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'Memory %',
                            data: [],
                            borderColor: '#6c5ce7',
                            backgroundColor: 'rgba(108, 92, 231, 0.1)',
                            tension: 0.4
                        }
                    ]
                },
                options: chartConfig
            }
        );
        
        // WebSocket event handlers
        ws.onopen = function(event) {
            updateConnectionStatus(true);
        };
        
        ws.onclose = function(event) {
            updateConnectionStatus(false);
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
            updateConnectionStatus(false);
        };
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        };
        
        function updateConnectionStatus(connected) {
            const indicator = document.getElementById('connection-status');
            const text = document.getElementById('connection-text');
            
            if (connected) {
                indicator.className = 'status-indicator status-online';
                text.textContent = 'Connected';
            } else {
                indicator.className = 'status-indicator status-offline';
                text.textContent = 'Disconnected';
            }
        }
        
        function handleWebSocketMessage(data) {
            switch(data.type) {
                case 'initial_data':
                    updateDashboard(data);
                    break;
                case 'broadcast':
                    handleBroadcastUpdate(data.data);
                    break;
                case 'update':
                    updateDashboard(data);
                    break;
            }
        }
        
        function handleBroadcastUpdate(data) {
            if (data.type === 'metrics_update') {
                updateMetrics(data.metrics);
            } else if (data.type === 'prediction_update') {
                updatePrediction(data.prediction);
            }
        }
        
        function updateDashboard(data) {
            if (data.metrics) {
                data.metrics.forEach(metric => updateMetrics(metric));
            }
            if (data.predictions) {
                data.predictions.forEach(pred => updatePrediction(pred));
            }
            if (data.system_stats) {
                updateSystemStats(data.system_stats);
            }
        }
        
        function updateMetrics(metrics) {
            // Update stat cards
            document.getElementById('total-packets').textContent = 
                metrics.total_packets || 0;
            document.getElementById('throughput').textContent = 
                `${(metrics.throughput || 0).toFixed(2)} MB/s`;
        }
        
        function updatePrediction(prediction) {
            // Update prediction stats
            const reelProb = prediction.reel_probability || 0;
            const confidence = prediction.confidence || 0;
            
            document.getElementById('reel-percentage').textContent = 
                `${(reelProb * 100).toFixed(1)}%`;
            document.getElementById('confidence-score').textContent = 
                `${(confidence * 100).toFixed(1)}%`;
            
            // Add to charts
            addToChart(classificationChart, reelProb * 100);
            addToChart(confidenceChart, confidence * 100);
            
            // Update predictions list
            updatePredictionsList(prediction);
        }
        
        function updateSystemStats(stats) {
            addToChart(systemChart, stats.cpu_percent, 0);
            addToChart(systemChart, stats.memory_percent, 1);
        }
        
        function addToChart(chart, value, datasetIndex = 0) {
            const now = new Date().toLocaleTimeString();
            
            if (chart.data.labels.length > 50) {
                chart.data.labels.shift();
                chart.data.datasets[datasetIndex].data.shift();
            }
            
            chart.data.labels.push(now);
            chart.data.datasets[datasetIndex].data.push(value);
            chart.update('none');
        }
        
        function updatePredictionsList(prediction) {
            const list = document.getElementById('predictions-list');
            const isReel = prediction.reel_probability > 0.5;
            
            const item = document.createElement('div');
            item.style.padding = '8px';
            item.style.marginBottom = '4px';
            item.style.borderRadius = '4px';
            item.style.backgroundColor = '#f8f9fa';
            
            item.innerHTML = `
                <span class="prediction-badge ${isReel ? 'reel-traffic' : 'non-reel-traffic'}">
                    ${isReel ? 'REEL' : 'NON-REEL'}
                </span>
                Confidence: ${(prediction.confidence * 100).toFixed(1)}%
                <small style="float: right;">${new Date().toLocaleTimeString()}</small>
            `;
            
            list.insertBefore(item, list.firstChild);
            
            // Keep only last 10 predictions
            while (list.children.length > 10) {
                list.removeChild(list.lastChild);
            }
        }
        
        // Keep connection alive
        setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'ping'}));
            }
        }, 30000);
    </script>
</body>
</html>
        """
    
    def run(self):
        """Run the dashboard server."""
        self.logger.info(f"Starting dashboard server on {self.host}:{self.port}")
        
        # Start system monitoring
        self.start_monitoring()
        
        try:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
        except KeyboardInterrupt:
            self.logger.info("Dashboard server stopped by user")
        finally:
            self.stop_monitoring()


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Traffic Detection Dashboard")
    parser.add_argument('--host', default='localhost', help="Host to bind to")
    parser.add_argument('--port', type=int, default=8080, help="Port to listen on")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create and run dashboard
    dashboard = TrafficDashboard(host=args.host, port=args.port)
    dashboard.run()


if __name__ == "__main__":
    main()
