#!/usr/bin/env python3
"""
Samsung Real-Time Traffic Detection Dashboard
Integrated system with live data updates
"""

import asyncio
import json
import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Any

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
    from fastapi.responses import HTMLResponse
    import uvicorn
    import psutil
    REQUIRED_MODULES = True
except ImportError as e:
    print(f"Missing required modules: {e}")
    print("Install with: pip install fastapi uvicorn psutil")
    REQUIRED_MODULES = False
    exit(1)

# Configure logging
logging.basicConfig(level=logging.WARNING)

# Global shared state
shared_state = {
    'total_packets': 0,
    'reel_video': 0,
    'social_media': 0,
    'regular_traffic': 0,
    'current_classification': 'REGULAR_TRAFFIC',
    'current_confidence': 0,
    'current_bandwidth': 0.0,
    'current_packets': 0,
    'last_update': time.time(),
    'analyzer_running': False
}

class RealTimeAnalyzer:
    def __init__(self):
        self.previous_net = psutil.net_io_counters()
        self.running = False
        
    def get_network_stats(self):
        """Get real network statistics"""
        current_net = psutil.net_io_counters()
        
        bytes_sent_rate = current_net.bytes_sent - self.previous_net.bytes_sent
        bytes_recv_rate = current_net.bytes_recv - self.previous_net.bytes_recv
        packets_sent_rate = current_net.packets_sent - self.previous_net.packets_sent
        packets_recv_rate = current_net.packets_recv - self.previous_net.packets_recv
        
        self.previous_net = current_net
        
        total_bandwidth = bytes_sent_rate + bytes_recv_rate
        total_packets = packets_sent_rate + packets_recv_rate
        
        return total_bandwidth, total_packets
    
    def classify_traffic(self, bandwidth, packets):
        """Classify traffic based on patterns"""
        if bandwidth > 500_000:  # > 500 KB/s
            if packets > 800:
                return "REEL_VIDEO", 95
            else:
                return "SOCIAL_MEDIA", 78
        elif bandwidth > 50_000:  # > 50 KB/s
            return "SOCIAL_MEDIA", 85
        else:
            return "REGULAR_TRAFFIC", 70
    
    def run_continuous_analysis(self):
        """Main analysis loop"""
        global shared_state
        
        print("Samsung Real-Time Traffic Analyzer Started")
        print("Analyzing network traffic every 3 seconds...")
        print("-" * 50)
        
        self.running = True
        shared_state['analyzer_running'] = True
        
        while self.running:
            try:
                # Get network data
                bandwidth, packets = self.get_network_stats()
                
                # Classify traffic
                traffic_type, confidence = self.classify_traffic(bandwidth, packets)
                
                # Update shared state
                shared_state['total_packets'] += packets
                shared_state['current_classification'] = traffic_type
                shared_state['current_confidence'] = confidence
                shared_state['current_bandwidth'] = bandwidth / 1024  # KB/s
                shared_state['current_packets'] = packets
                shared_state['last_update'] = time.time()
                
                # Update counters
                if traffic_type == "REEL_VIDEO":
                    shared_state['reel_video'] += 1
                    indicator = "REEL_VIDEO"
                elif traffic_type == "SOCIAL_MEDIA":
                    shared_state['social_media'] += 1
                    indicator = "SOCIAL_MEDIA"
                else:
                    shared_state['regular_traffic'] += 1
                    indicator = "REGULAR_TRAFFIC"
                
                # Console output
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] {indicator:15} | "
                      f"Confidence: {confidence:2}% | "
                      f"Bandwidth: {bandwidth/1024:7.1f} KB/s | "
                      f"Packets/s: {packets:4}")
                
                time.sleep(3)  # 3-second intervals
                
            except Exception as e:
                print(f"Analysis error: {e}")
                time.sleep(1)
        
        shared_state['analyzer_running'] = False
        print("Traffic analyzer stopped")

# Create analyzer instance
analyzer = RealTimeAnalyzer()

# FastAPI app
app = FastAPI(title="Samsung Traffic Dashboard")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, data: dict):
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(data))
            except Exception:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Samsung Traffic Detection Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .subtitle { font-size: 1.2em; opacity: 0.9; margin-bottom: 20px; }
        .status {
            display: inline-block;
            padding: 8px 16px;
            background: #4CAF50;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 0;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .stat-card h3 { margin: 0 0 10px 0; font-size: 1.1em; opacity: 0.8; }
        .stat-number { font-size: 2.5em; font-weight: bold; margin: 10px 0; }
        .stat-percentage { font-size: 1.2em; opacity: 0.9; }
        .current-classification {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }
        .classification-type { font-size: 1.8em; font-weight: bold; margin-bottom: 10px; }
        .confidence-bar {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            height: 20px;
            margin: 10px 0;
            overflow: hidden;
        }
        .confidence-fill {
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            height: 100%;
            transition: width 0.3s ease;
            border-radius: 10px;
        }
        .live-data {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        .data-row {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 5px;
        }
        .blinking { animation: blink 1s linear infinite; }
        @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0.3; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Samsung EnnovateX 2025</h1>
            <div class="subtitle">Real-time Traffic Classification Dashboard</div>
            <div class="subtitle">Detecting Reel/Video vs Regular Traffic in Social Media Applications</div>
            <div class="status" id="connection-status">Connecting...</div>
            <div class="status" id="analyzer-status">Starting analyzer...</div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Packets</h3>
                <div class="stat-number" id="total-packets">0</div>
            </div>
            <div class="stat-card">
                <h3>Reel/Video Traffic</h3>
                <div class="stat-number" id="reel-count">0</div>
                <div class="stat-percentage" id="reel-percentage">0%</div>
            </div>
            <div class="stat-card">
                <h3>Social Media</h3>
                <div class="stat-number" id="social-count">0</div>
                <div class="stat-percentage" id="social-percentage">0%</div>
            </div>
            <div class="stat-card">
                <h3>Regular Traffic</h3>
                <div class="stat-number" id="regular-count">0</div>
                <div class="stat-percentage" id="regular-percentage">0%</div>
            </div>
        </div>

        <div class="current-classification">
            <div class="classification-type" id="current-type">REGULAR_TRAFFIC</div>
            <div>Confidence: <span id="confidence-value">0</span>%</div>
            <div class="confidence-bar">
                <div class="confidence-fill" id="confidence-bar" style="width: 0%"></div>
            </div>
        </div>

        <div class="live-data">
            <h3>Real-time Network Data</h3>
            <div class="data-row">
                <span>Bandwidth:</span>
                <span id="bandwidth-value">0.0 KB/s</span>
            </div>
            <div class="data-row">
                <span>Packets/sec:</span>
                <span id="packets-value">0</span>
            </div>
            <div class="data-row">
                <span>Last Update:</span>
                <span id="last-update">Never</span>
            </div>
        </div>
    </div>

    <script>
        const ws = new WebSocket('ws://localhost:8003/ws');
        
        ws.onopen = function(event) {
            document.getElementById('connection-status').innerHTML = 'Connected to capture system';
            console.log('WebSocket connected');
        };

        ws.onclose = function(event) {
            document.getElementById('connection-status').innerHTML = 'Disconnected';
            console.log('WebSocket disconnected');
        };

        ws.onerror = function(error) {
            console.log('WebSocket error:', error);
        };

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            console.log('Received data:', data);
            
            // Update connection status
            if (data.analyzer_running) {
                document.getElementById('analyzer-status').innerHTML = 'Analyzer running';
            } else {
                document.getElementById('analyzer-status').innerHTML = 'Analyzer stopped';
            }
            
            // Update statistics
            document.getElementById('total-packets').textContent = data.total_packets.toLocaleString();
            document.getElementById('reel-count').textContent = data.reel_video;
            document.getElementById('social-count').textContent = data.social_media;
            document.getElementById('regular-count').textContent = data.regular_traffic;
            
            // Calculate percentages
            const total = data.reel_video + data.social_media + data.regular_traffic;
            if (total > 0) {
                document.getElementById('reel-percentage').textContent = 
                    Math.round((data.reel_video / total) * 100) + '%';
                document.getElementById('social-percentage').textContent = 
                    Math.round((data.social_media / total) * 100) + '%';
                document.getElementById('regular-percentage').textContent = 
                    Math.round((data.regular_traffic / total) * 100) + '%';
            }
            
            // Update current classification
            const classificationNames = {
                'REEL_VIDEO': 'REEL VIDEO',
                'SOCIAL_MEDIA': 'SOCIAL MEDIA',
                'REGULAR_TRAFFIC': 'REGULAR TRAFFIC'
            };
            
            const displayName = classificationNames[data.current_classification] || 'REGULAR TRAFFIC';
            const typeElement = document.getElementById('current-type');
            typeElement.textContent = displayName;
            typeElement.className = 'classification-type blinking';
            
            document.getElementById('confidence-value').textContent = data.current_confidence;
            document.getElementById('confidence-bar').style.width = data.current_confidence + '%';
            
            // Update live data
            document.getElementById('bandwidth-value').textContent = 
                data.current_bandwidth.toFixed(1) + ' KB/s';
            document.getElementById('packets-value').textContent = data.current_packets;
            document.getElementById('last-update').textContent = 
                new Date().toLocaleTimeString();
            
            // Remove blinking after 2 seconds
            setTimeout(() => {
                typeElement.className = 'classification-type';
            }, 2000);
        };
    </script>
</body>
</html>
    """)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Send current state to client
            await manager.broadcast(shared_state.copy())
            await asyncio.sleep(1)  # Send updates every second
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def start_analyzer():
    """Start the traffic analyzer in a separate thread"""
    analyzer.run_continuous_analysis()

@app.on_event("startup")
async def startup_event():
    """Start the analyzer when the app starts"""
    print("Starting traffic analyzer...")
    analyzer_thread = threading.Thread(target=start_analyzer, daemon=True)
    analyzer_thread.start()
    print("Traffic analyzer started in background")

if __name__ == "__main__":
    print("Samsung EnnovateX 2025 - Integrated Traffic Detection System")
    print("Dashboard: http://localhost:8003")
    print("Real-time analysis starting...")
    print("-" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="warning")
