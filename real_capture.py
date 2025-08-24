#!/usr/bin/env python3
"""
Real-time Traffic Classification System

Production system for real-time detection of video traffic
vs non-video traffic in social networking applications.

Uses real packet capture for accurate classification.
"""

import sys
import os
import time
import logging
import threading
import json
from queue import Queue, Empty
from datetime import datetime
from typing import Dict, List, Any, Optional
import subprocess

# Check for required packages
try:
    import scapy.all as scapy
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.http import HTTPRequest, HTTPResponse
    SCAPY_AVAILABLE = True
except ImportError:
    print("ERROR: Scapy not available. Install with: pip install scapy")
    SCAPY_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("ERROR: psutil not available. Install with: pip install psutil")
    PSUTIL_AVAILABLE = False

try:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    print("ERROR: ML packages not available. Install with: pip install numpy pandas scikit-learn")
    ML_AVAILABLE = False


class NetworkFeatureExtractor:
    """Real-time feature extraction from network packets."""
    
    def __init__(self):
        self.packet_buffer = []
        self.window_size = 60  # 60 seconds window
        self.features_list = []
        
    def extract_packet_features(self, packet) -> Dict[str, Any]:
        """Extract features from a single packet."""
        features = {
            'timestamp': time.time(),
            'size': len(packet),
            'protocol': 'Unknown',
            'src_port': 0,
            'dst_port': 0,
            'flags': 0,
            'is_tcp': False,
            'is_udp': False,
            'is_http': False,
            'payload_size': 0
        }
        
        if packet.haslayer(IP):
            features['ip_len'] = packet[IP].len
            features['ip_flags'] = packet[IP].flags
            
        if packet.haslayer(TCP):
            features['protocol'] = 'TCP'
            features['src_port'] = packet[TCP].sport
            features['dst_port'] = packet[TCP].dport
            features['flags'] = packet[TCP].flags
            features['is_tcp'] = True
            features['window_size'] = packet[TCP].window
            features['payload_size'] = len(packet[TCP].payload)
            
        elif packet.haslayer(UDP):
            features['protocol'] = 'UDP'
            features['src_port'] = packet[UDP].sport
            features['dst_port'] = packet[UDP].dport
            features['is_udp'] = True
            features['payload_size'] = len(packet[UDP].payload)
            
        if packet.haslayer(HTTPRequest) or packet.haslayer(HTTPResponse):
            features['is_http'] = True
            
        return features
    
    def extract_flow_features(self, packets: List[Dict]) -> List[float]:
        """Extract aggregated features from packet flow."""
        if not packets:
            return [0.0] * 20
        
        # Basic statistics
        sizes = [p['size'] for p in packets]
        timestamps = [p['timestamp'] for p in packets]
        
        if len(timestamps) > 1:
            duration = max(timestamps) - min(timestamps)
            packet_rate = len(packets) / max(duration, 0.001)
        else:
            duration = 0
            packet_rate = 0
        
        # Port analysis
        common_video_ports = [80, 443, 1935, 8080, 8443]  # HTTP, HTTPS, RTMP, alt-HTTP
        video_port_ratio = sum(1 for p in packets if p['dst_port'] in common_video_ports) / len(packets)
        
        # Protocol analysis
        tcp_ratio = sum(1 for p in packets if p['is_tcp']) / len(packets)
        udp_ratio = sum(1 for p in packets if p['is_udp']) / len(packets)
        http_ratio = sum(1 for p in packets if p['is_http']) / len(packets)
        
        # Size analysis
        large_packets = sum(1 for s in sizes if s > 1000)
        large_packet_ratio = large_packets / len(packets)
        
        # Burst detection
        payload_sizes = [p['payload_size'] for p in packets]
        avg_payload = np.mean(payload_sizes) if payload_sizes else 0
        
        features = [
            len(packets),               # Total packets
            np.mean(sizes),            # Average packet size
            np.std(sizes),             # Packet size variance
            duration,                  # Flow duration
            packet_rate,               # Packets per second
            video_port_ratio,          # Video port usage ratio
            tcp_ratio,                 # TCP ratio
            udp_ratio,                 # UDP ratio
            http_ratio,                # HTTP ratio
            large_packet_ratio,        # Large packet ratio
            avg_payload,               # Average payload size
            np.max(sizes) if sizes else 0,  # Max packet size
            np.min(sizes) if sizes else 0,  # Min packet size
            np.percentile(sizes, 75) if sizes else 0,  # 75th percentile
            np.percentile(sizes, 25) if sizes else 0,  # 25th percentile
            len(set(p['dst_port'] for p in packets)),  # Unique destination ports
            len(set(p['src_port'] for p in packets)),  # Unique source ports
            sum(1 for p in packets if p['size'] > 1400),  # MTU-sized packets
            sum(1 for p in packets if p['flags'] & 0x18),  # PSH+ACK flags (data)
            sum(1 for p in packets if p['payload_size'] > 500)  # Large payload packets
        ]
        
        return features


class SimpleTrafficClassifier:
    """Simple but effective traffic classifier for video/reel detection."""
    
    def __init__(self):
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        
    def classify(self, features: List[float]) -> Dict[str, Any]:
        """Classify traffic based on features."""
        if len(features) < 20:
            features.extend([0.0] * (20 - len(features)))
        
        # Video/Reel detection heuristics based on real traffic patterns
        packet_count = features[0]
        avg_size = features[1]
        packet_rate = features[4]
        video_port_ratio = features[5]
        large_packet_ratio = features[9]
        avg_payload = features[10]
        max_size = features[11]
        
        # Scoring system based on video traffic characteristics
        video_score = 0.0
        
        # High packet rate suggests streaming
        if packet_rate > 50:
            video_score += 0.3
        elif packet_rate > 20:
            video_score += 0.15
            
        # Large average packet size suggests video data
        if avg_size > 1200:
            video_score += 0.25
        elif avg_size > 800:
            video_score += 0.1
            
        # High proportion of large packets
        if large_packet_ratio > 0.7:
            video_score += 0.2
        elif large_packet_ratio > 0.4:
            video_score += 0.1
            
        # Video ports usage
        if video_port_ratio > 0.8:
            video_score += 0.15
            
        # Large payload suggests video content
        if avg_payload > 800:
            video_score += 0.1
            
        # Determine classification
        if video_score > 0.6:
            prediction = "reel_video"
            confidence = min(0.95, 0.7 + video_score * 0.3)
        elif video_score > 0.3:
            prediction = "social_media"
            confidence = min(0.85, 0.6 + video_score * 0.25)
        else:
            prediction = "regular_traffic"
            confidence = min(0.80, 0.5 + (1 - video_score) * 0.3)
            
        return {
            'prediction': prediction,
            'confidence': confidence,
            'video_score': video_score,
            'packet_count': int(packet_count),
            'avg_packet_size': avg_size,
            'packet_rate': packet_rate
        }


class RealTimeCapture:
    """Production-ready real-time packet capture and classification."""
    
    def __init__(self, interface=None):
        self.interface = interface or self._get_active_interface()
        self.feature_extractor = NetworkFeatureExtractor()
        self.classifier = SimpleTrafficClassifier()
        self.packet_queue = Queue()
        self.running = False
        self.stats = {
            'total_packets': 0,
            'reel_video': 0,
            'social_media': 0,
            'regular_traffic': 0
        }
        
    def _get_active_interface(self):
        """Get the active network interface."""
        if not PSUTIL_AVAILABLE:
            return None
            
        interfaces = psutil.net_if_addrs()
        for interface_name, addresses in interfaces.items():
            for addr in addresses:
                if addr.family == 2 and not addr.address.startswith('127.'):  # IPv4, not localhost
                    return interface_name
        return None
    
    def packet_handler(self, packet):
        """Handle captured packets."""
        try:
            features = self.feature_extractor.extract_packet_features(packet)
            self.packet_queue.put(features)
        except Exception as e:
            logger.error(f"Error processing packet: {e}")
    
    def classification_worker(self):
        """Worker thread for packet classification."""
        window_packets = []
        last_classification = time.time()
        
        while self.running:
            try:
                # Get packet from queue
                packet_features = self.packet_queue.get(timeout=1.0)
                window_packets.append(packet_features)
                
                # Clean old packets (keep 60 second window)
                current_time = time.time()
                window_packets = [p for p in window_packets 
                                if current_time - p['timestamp'] <= 60]
                
                # Classify every 5 seconds or when window is full
                if (current_time - last_classification >= 5.0 or 
                    len(window_packets) >= 100):
                    
                    if window_packets:
                        flow_features = self.feature_extractor.extract_flow_features(window_packets)
                        result = self.classifier.classify(flow_features)
                        
                        # Update statistics
                        self.stats['total_packets'] += result['packet_count']
                        self.stats[result['prediction']] += 1
                        
                        # Display results
                        self._display_result(result, current_time)
                        
                    last_classification = current_time
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Classification error: {e}")
    
    def _display_result(self, result: Dict, timestamp: float):
        """Display classification results."""
        time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
        
        print(f"[{time_str}] "
              f"Packets: {result['packet_count']:3d} | "
              f"Type: {result['prediction']:15s} | "
              f"Confidence: {result['confidence']:.1%} | "
              f"Avg Size: {result['avg_packet_size']:.0f}B | "
              f"Rate: {result['packet_rate']:.1f}/s")
        
        # Show statistics every 20 classifications
        if self.stats['total_packets'] > 0 and self.stats['total_packets'] % 200 == 0:
            total = sum([self.stats['reel_video'], self.stats['social_media'], self.stats['regular_traffic']])
            if total > 0:
                print(f"\nTraffic Analysis Summary:")
                print(f"   Reel/Video Traffic:  {self.stats['reel_video']:3d} ({self.stats['reel_video']/total:.1%})")
                print(f"   Social Media:        {self.stats['social_media']:3d} ({self.stats['social_media']/total:.1%})")
                print(f"   Regular Traffic:     {self.stats['regular_traffic']:3d} ({self.stats['regular_traffic']/total:.1%})")
                print(f"   Total Packets:       {self.stats['total_packets']:,}")
                print("-" * 70)
    
    def start_capture(self):
        """Start real-time packet capture."""
        if not SCAPY_AVAILABLE:
            print("ERROR: Scapy not available. Real packet capture requires Scapy.")
            print("Install with: pip install scapy")
            return False
            
        print("Real-Time Traffic Detection System")
        print("=" * 70)
        print(f"Interface: {self.interface}")
        print("Target: Video vs Regular Traffic Detection")
        # This script captures and analyzes live network packets
        print("-" * 70)
        
        self.running = True
        
        # Start classification worker thread
        classifier_thread = threading.Thread(target=self.classification_worker)
        classifier_thread.daemon = True
        classifier_thread.start()
        
        try:
            # Start packet capture
            print("Starting packet capture... (Press Ctrl+C to stop)")
            scapy.sniff(
                iface=self.interface,
                prn=self.packet_handler,
                store=False,
                stop_filter=lambda x: not self.running
            )
        except KeyboardInterrupt:
            print(f"\n\nCapture stopped by user")
        except Exception as e:
            print(f"Capture error: {e}")
        finally:
            self.running = False
            self._show_final_stats()
    
    def _show_final_stats(self):
        """Show final statistics."""
        total = sum([self.stats['reel_video'], self.stats['social_media'], self.stats['regular_traffic']])
        print(f"\nFinal Analysis Results:")
        print(f"   Total Packets Analyzed: {self.stats['total_packets']:,}")
        if total > 0:
            print(f"   Reel/Video Traffic:   {self.stats['reel_video']} ({self.stats['reel_video']/total:.1%})")
            print(f"   Social Media:         {self.stats['social_media']} ({self.stats['social_media']/total:.1%})")
            print(f"   Regular Traffic:      {self.stats['regular_traffic']} ({self.stats['regular_traffic']/total:.1%})")
        print(f"\nAnalysis complete.")


def check_permissions():
    """Check if running with appropriate permissions for packet capture."""
    import os
    if os.name == 'nt':  # Windows
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin()
    else:  # Linux/macOS
        return os.geteuid() == 0


def main():
    """Main entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check dependencies
    missing_deps = []
    if not SCAPY_AVAILABLE:
        missing_deps.append("scapy")
    if not ML_AVAILABLE:
        missing_deps.append("numpy pandas scikit-learn")
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print(f"Install with: pip install {' '.join(missing_deps)}")
        return 1
    
    # Check permissions
    if not check_permissions():
        print("ERROR: Packet capture requires administrator/root privileges")
        print("Windows: Run as Administrator")
        print("Linux/macOS: Run with sudo")
        return 1
    
    # Start capture
    capture = RealTimeCapture()
    capture.start_capture()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
