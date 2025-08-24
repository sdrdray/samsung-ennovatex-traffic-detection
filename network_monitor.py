#!/usr/bin/env python3
"""
Samsung EnnovateX 2025 - Real Network Traffic Monitor
Alternative approach using system network statistics (no admin required)
"""

import sys
import os
import time
import json
import threading
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from queue import Queue, Empty
import subprocess

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("ERROR: psutil required. Install with: pip install psutil")
    PSUTIL_AVAILABLE = False

try:
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    print("ERROR: numpy required. Install with: pip install numpy")
    ML_AVAILABLE = False


class NetworkMonitor:
    """Monitor network activity without requiring admin privileges."""
    
    def __init__(self):
        self.baseline_stats = {}
        self.previous_stats = {}
        self.initialize_baseline()
        
    def initialize_baseline(self):
        """Initialize baseline network statistics."""
        if PSUTIL_AVAILABLE:
            self.baseline_stats = psutil.net_io_counters()
            self.previous_stats = self.baseline_stats
            
    def get_network_activity(self) -> Dict[str, Any]:
        """Get current network activity metrics."""
        if not PSUTIL_AVAILABLE:
            return {}
            
        current_stats = psutil.net_io_counters()
        
        # Calculate deltas
        bytes_sent_delta = current_stats.bytes_sent - self.previous_stats.bytes_sent
        bytes_recv_delta = current_stats.bytes_recv - self.previous_stats.bytes_recv
        packets_sent_delta = current_stats.packets_sent - self.previous_stats.packets_sent
        packets_recv_delta = current_stats.packets_recv - self.previous_stats.packets_recv
        
        self.previous_stats = current_stats
        
        activity = {
            'timestamp': time.time(),
            'bytes_sent_rate': max(0, bytes_sent_delta),
            'bytes_recv_rate': max(0, bytes_recv_delta),
            'packets_sent_rate': max(0, packets_sent_delta),
            'packets_recv_rate': max(0, packets_recv_delta),
            'total_bytes_rate': max(0, bytes_sent_delta + bytes_recv_delta),
            'total_packets_rate': max(0, packets_sent_delta + packets_recv_delta)
        }
        
        return activity
    
    def get_active_connections(self) -> List[Dict]:
        """Get active network connections."""
        if not PSUTIL_AVAILABLE:
            return []
            
        connections = []
        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.status == 'ESTABLISHED':
                    connections.append({
                        'local_port': conn.laddr.port if conn.laddr else 0,
                        'remote_port': conn.raddr.port if conn.raddr else 0,
                        'remote_ip': conn.raddr.ip if conn.raddr else 'unknown',
                        'pid': conn.pid,
                        'family': conn.family.name if hasattr(conn.family, 'name') else str(conn.family),
                        'type': conn.type.name if hasattr(conn.type, 'name') else str(conn.type)
                    })
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
            
        return connections


class ProcessMonitor:
    """Monitor running processes for social media applications."""
    
    def __init__(self):
        # Known social media and video apps
        self.social_apps = {
            'chrome.exe': ['instagram', 'tiktok', 'youtube', 'facebook', 'twitter'],
            'firefox.exe': ['instagram', 'tiktok', 'youtube', 'facebook', 'twitter'],
            'msedge.exe': ['instagram', 'tiktok', 'youtube', 'facebook', 'twitter'],
            'WhatsApp.exe': ['whatsapp'],
            'Telegram.exe': ['telegram'],
            'Discord.exe': ['discord'],
            'Spotify.exe': ['spotify'],
            'vlc.exe': ['video'],
            'zoom.exe': ['video_call'],
            'Teams.exe': ['video_call'],
            'Instagram.exe': ['instagram'],
            'TikTok.exe': ['tiktok']
        }
        
    def get_running_social_apps(self) -> List[Dict]:
        """Get currently running social media applications."""
        if not PSUTIL_AVAILABLE:
            return []
            
        running_apps = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                proc_name = proc.info['name'].lower()
                
                for app_pattern, categories in self.social_apps.items():
                    if app_pattern.lower() in proc_name:
                        running_apps.append({
                            'name': proc.info['name'],
                            'pid': proc.info['pid'],
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_mb': proc.info['memory_info'].rss / 1024 / 1024 if proc.info['memory_info'] else 0,
                            'categories': categories
                        })
                        break
                        
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
            
        return running_apps


class TrafficClassifier:
    """Classify network traffic based on system metrics."""
    
    def __init__(self):
        self.history = []
        self.window_size = 30  # 30 second window
        
    def analyze_traffic(self, network_activity: Dict, active_connections: List, running_apps: List) -> Dict:
        """Analyze and classify current traffic."""
        
        # Store in history
        analysis_point = {
            'timestamp': time.time(),
            'network': network_activity,
            'connections': active_connections,
            'apps': running_apps
        }
        self.history.append(analysis_point)
        
        # Keep only recent history
        current_time = time.time()
        self.history = [h for h in self.history if current_time - h['timestamp'] <= self.window_size]
        
        # Analyze patterns
        return self._classify_current_activity(analysis_point)
    
    def _classify_current_activity(self, analysis: Dict) -> Dict:
        """Classify the current network activity."""
        network = analysis['network']
        connections = analysis['connections']
        apps = analysis['apps']
        
        # Initialize scores
        video_score = 0.0
        social_score = 0.0
        
        # Network activity analysis
        total_bytes = network.get('total_bytes_rate', 0)
        total_packets = network.get('total_packets_rate', 0)
        
        # High bandwidth usage suggests video
        if total_bytes > 1000000:  # > 1MB/s
            video_score += 0.4
        elif total_bytes > 500000:  # > 500KB/s
            video_score += 0.2
            
        # High packet rate
        if total_packets > 100:
            video_score += 0.2
        elif total_packets > 50:
            social_score += 0.1
            
        # Connection analysis
        video_ports = [80, 443, 1935, 8080, 8443]  # Common video streaming ports
        social_ports = [80, 443, 3000, 8000]  # Common social media ports
        
        video_connections = sum(1 for conn in connections if conn['remote_port'] in video_ports)
        if video_connections > 5:
            video_score += 0.2
        elif video_connections > 2:
            social_score += 0.1
            
        # Running applications analysis
        video_apps = ['instagram', 'tiktok', 'youtube', 'video', 'spotify']
        social_apps = ['facebook', 'twitter', 'whatsapp', 'telegram', 'discord']
        
        for app in apps:
            for category in app['categories']:
                if category in video_apps:
                    # High CPU/memory usage suggests active video
                    if app['cpu_percent'] > 5 or app['memory_mb'] > 200:
                        video_score += 0.3
                    else:
                        social_score += 0.1
                elif category in social_apps:
                    social_score += 0.1
                    
        # Determine classification
        if video_score > 0.6:
            prediction = "reel_video"
            confidence = min(0.95, 0.7 + video_score * 0.25)
        elif social_score > 0.3 or video_score > 0.3:
            prediction = "social_media"
            confidence = min(0.85, 0.6 + max(social_score, video_score) * 0.25)
        else:
            prediction = "regular_traffic"
            confidence = min(0.80, 0.5 + (1 - max(video_score, social_score)) * 0.3)
            
        return {
            'prediction': prediction,
            'confidence': confidence,
            'video_score': video_score,
            'social_score': social_score,
            'total_bytes_rate': total_bytes,
            'total_packets_rate': total_packets,
            'active_connections': len(connections),
            'social_apps_running': len(apps)
        }


class SystemTrafficMonitor:
    """Main system for monitoring network traffic without admin privileges."""
    
    def __init__(self):
        self.network_monitor = NetworkMonitor()
        self.process_monitor = ProcessMonitor()
        self.classifier = TrafficClassifier()
        self.running = False
        self.stats = {
            'total_classifications': 0,
            'reel_video': 0,
            'social_media': 0,
            'regular_traffic': 0
        }
        
    def send_to_dashboard(self, data: Dict):
        """Send data to dashboard if running."""
        try:
            # Try to send to dashboard
            requests.post('http://localhost:8000/classification', json=data, timeout=1)
        except:
            pass  # Dashboard not running or not reachable
            
    def update_dashboard_stats(self):
        """Update dashboard statistics."""
        try:
            requests.post('http://localhost:8000/update_stats', json=self.stats, timeout=1)
        except:
            pass
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if not PSUTIL_AVAILABLE:
            print("ERROR: psutil not available. Install with: pip install psutil")
            return False
            
        print("Samsung EnnovateX 2025 - Network Traffic Monitor")
        print("=" * 60)
        print("Method: System Network Statistics (No Admin Required)")
        print("Target: Reel/Video vs Regular Traffic Detection")
        print("Monitoring: Active apps and network usage patterns")
        print("-" * 60)
        
        self.running = True
        
        try:
            print("Starting network monitoring... (Press Ctrl+C to stop)")
            
            while self.running:
                # Collect data
                network_activity = self.network_monitor.get_network_activity()
                active_connections = self.network_monitor.get_active_connections()
                running_apps = self.process_monitor.get_running_social_apps()
                
                # Analyze and classify
                result = self.classifier.analyze_traffic(network_activity, active_connections, running_apps)
                
                # Update statistics
                self.stats['total_classifications'] += 1
                self.stats[result['prediction']] += 1
                
                # Display results
                self._display_result(result)
                
                # Send to dashboard
                self.send_to_dashboard(result)
                if self.stats['total_classifications'] % 5 == 0:
                    self.update_dashboard_stats()
                
                # Wait before next analysis
                time.sleep(3)  # Analyze every 3 seconds
                
        except KeyboardInterrupt:
            print(f"\n\nMonitoring stopped by user")
        except Exception as e:
            print(f"Monitoring error: {e}")
        finally:
            self.running = False
            self._show_final_stats()
    
    def _display_result(self, result: Dict):
        """Display analysis results."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"[{timestamp}] "
              f"Traffic: {result['prediction']:15s} | "
              f"Confidence: {result['confidence']:.1%} | "
              f"Bandwidth: {result['total_bytes_rate']/1024:.0f}KB/s | "
              f"Apps: {result['social_apps_running']} | "
              f"Connections: {result['active_connections']}")
        
        # Show statistics every 20 classifications
        if self.stats['total_classifications'] % 20 == 0:
            total = sum([self.stats['reel_video'], self.stats['social_media'], self.stats['regular_traffic']])
            if total > 0:
                print(f"\nTraffic Analysis Summary:")
                print(f"   Reel/Video Traffic:  {self.stats['reel_video']:3d} ({self.stats['reel_video']/total:.1%})")
                print(f"   Social Media:        {self.stats['social_media']:3d} ({self.stats['social_media']/total:.1%})")
                print(f"   Regular Traffic:     {self.stats['regular_traffic']:3d} ({self.stats['regular_traffic']/total:.1%})")
                print(f"   Total Classifications: {self.stats['total_classifications']}")
                print("-" * 60)
    
    def _show_final_stats(self):
        """Show final statistics."""
        total = sum([self.stats['reel_video'], self.stats['social_media'], self.stats['regular_traffic']])
        print(f"\nFinal Analysis Results:")
        print(f"   Total Classifications: {self.stats['total_classifications']}")
        if total > 0:
            print(f"   Reel/Video Traffic:   {self.stats['reel_video']} ({self.stats['reel_video']/total:.1%})")
            print(f"   Social Media:         {self.stats['social_media']} ({self.stats['social_media']/total:.1%})")
            print(f"   Regular Traffic:      {self.stats['regular_traffic']} ({self.stats['regular_traffic']/total:.1%})")
        print(f"\nSamsung EnnovateX 2025 Network Analysis Complete!")
        print(f"This system monitors REAL network activity without admin privileges.")


def main():
    """Main entry point."""
    # Check dependencies
    if not PSUTIL_AVAILABLE:
        print("Missing dependency: psutil")
        print("Install with: pip install psutil")
        return 1
    
    # Start monitoring
    monitor = SystemTrafficMonitor()
    monitor.start_monitoring()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
