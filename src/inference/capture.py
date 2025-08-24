#!/usr/bin/env python3
"""
Real-time packet capture for traffic classification.

This module provides real-time packet capture functionality that feeds
into the traffic classification pipeline. It captures packets and
forwards them for feature extraction and classification.
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from typing import Dict, List, Optional, Callable
import threading
from queue import Queue, Empty
import json
import os

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Network capture imports
try:
    import scapy.all as scapy
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.http import HTTPRequest, HTTPResponse
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# Local imports
from src.features.extractor import FeatureExtractor
from src.inference.infer import InferenceEngine
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = logging.getLogger(__name__)


class RealTimeCapture:
    """
    Real-time packet capture and classification system.
    
    This class manages the complete pipeline from packet capture
    to traffic classification and result broadcasting.
    """
    
    def __init__(
        self,
        interface: str = "auto",
        window_size: float = 3.0,
        max_packets_per_window: int = 1000,
        output_queue: Optional[Queue] = None
    ):
        """
        Initialize real-time capture system.
        
        Args:
            interface: Network interface to capture from ("auto" for automatic)
            window_size: Time window for feature aggregation (seconds)
            max_packets_per_window: Maximum packets to process per window
            output_queue: Queue for sending classification results
        """
        self.interface = interface
        self.window_size = window_size
        self.max_packets_per_window = max_packets_per_window
        self.output_queue = output_queue or Queue()
        
        # Components
        self.feature_extractor = FeatureExtractor(window_size=window_size)
        self.inference_engine = InferenceEngine()
        
        # State management
        self.is_running = False
        self.packet_buffer = []
        self.last_classification_time = time.time()
        self.packet_count = 0
        self.classification_count = 0
        
        # Threading
        self.capture_thread = None
        self.processing_thread = None
        self.packet_queue = Queue(maxsize=10000)
        
        # Statistics
        self.stats = {
            'total_packets': 0,
            'processed_packets': 0,
            'classifications': 0,
            'errors': 0,
            'start_time': None,
            'last_classification': None
        }
        
        logger.info(f"Initialized real-time capture for interface: {interface}")
    
    def start_capture(self, duration: Optional[float] = None) -> None:
        """
        Start real-time packet capture and classification.
        
        Args:
            duration: Optional duration to run capture (seconds)
        """
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy not available. Please install: pip install scapy")
        
        if self.is_running:
            logger.warning("Capture already running")
            return
        
        logger.info("Starting real-time traffic capture and classification")
        
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        # Start capture thread
        self.capture_thread = threading.Thread(
            target=self._capture_packets,
            args=(duration,),
            daemon=True
        )
        self.capture_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._process_packets,
            daemon=True
        )
        self.processing_thread.start()
        
        # Set up signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Run until duration expires or interrupted
            if duration:
                time.sleep(duration)
                self.stop_capture()
            else:
                # Run indefinitely
                while self.is_running:
                    time.sleep(1)
                    self._log_stats()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
            self.stop_capture()
    
    def stop_capture(self) -> None:
        """Stop packet capture and processing."""
        if not self.is_running:
            return
        
        logger.info("Stopping traffic capture")
        self.is_running = False
        
        # Wait for threads to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        
        # Final statistics
        self._log_final_stats()
        
        logger.info("Traffic capture stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}")
        self.stop_capture()
        sys.exit(0)
    
    def _capture_packets(self, duration: Optional[float] = None):
        """
        Capture packets in background thread.
        
        Args:
            duration: Optional duration to capture (seconds)
        """
        try:
            # Determine interface
            if self.interface == "auto":
                interface = self._get_default_interface()
            else:
                interface = self.interface
            
            logger.info(f"Starting packet capture on interface: {interface}")
            
            # Calculate stop time if duration specified
            stop_time = None
            if duration:
                stop_time = time.time() + duration
            
            def packet_handler(packet):
                """Handle individual packets."""
                if not self.is_running:
                    return False  # Stop capture
                
                if stop_time and time.time() >= stop_time:
                    self.is_running = False
                    return False
                
                try:
                    # Add packet to queue for processing
                    self.packet_queue.put(packet, timeout=0.1)
                    self.stats['total_packets'] += 1
                except:
                    # Queue full, skip packet
                    pass
                
                return self.is_running
            
            # Start packet capture
            scapy.sniff(
                iface=interface,
                prn=packet_handler,
                stop_filter=lambda x: not self.is_running,
                store=False
            )
            
        except Exception as e:
            logger.error(f"Error in packet capture: {e}")
            self.stats['errors'] += 1
            self.is_running = False
    
    def _process_packets(self):
        """Process captured packets in background thread."""
        try:
            while self.is_running:
                try:
                    # Get packet from queue
                    packet = self.packet_queue.get(timeout=1.0)
                    
                    # Add to buffer
                    self.packet_buffer.append({
                        'timestamp': time.time(),
                        'packet': packet
                    })
                    
                    self.stats['processed_packets'] += 1
                    
                    # Check if it's time for classification
                    current_time = time.time()
                    if current_time - self.last_classification_time >= self.window_size:
                        self._classify_window()
                        self.last_classification_time = current_time
                    
                    # Limit buffer size
                    if len(self.packet_buffer) > self.max_packets_per_window * 2:
                        # Remove oldest packets
                        cutoff_time = current_time - self.window_size * 2
                        self.packet_buffer = [
                            p for p in self.packet_buffer 
                            if p['timestamp'] > cutoff_time
                        ]
                
                except Empty:
                    # No packets available, check for classification
                    current_time = time.time()
                    if (current_time - self.last_classification_time >= self.window_size 
                        and self.packet_buffer):
                        self._classify_window()
                        self.last_classification_time = current_time
                    continue
                
                except Exception as e:
                    logger.error(f"Error processing packet: {e}")
                    self.stats['errors'] += 1
                    continue
        
        except Exception as e:
            logger.error(f"Error in packet processing thread: {e}")
            self.stats['errors'] += 1
    
    def _classify_window(self):
        """Classify current packet window."""
        try:
            if not self.packet_buffer:
                return
            
            # Extract features from current window
            current_time = time.time()
            window_start = current_time - self.window_size
            
            # Filter packets in current window
            window_packets = [
                p for p in self.packet_buffer 
                if p['timestamp'] >= window_start
            ]
            
            if not window_packets:
                return
            
            # Convert packets to feature format
            packet_data = []
            for p in window_packets:
                packet_info = self._extract_packet_info(p['packet'])
                if packet_info:
                    packet_info['timestamp'] = p['timestamp']
                    packet_data.append(packet_info)
            
            if not packet_data:
                return
            
            # Extract features
            features = self.feature_extractor.extract_features_from_packets(packet_data)
            
            if features is not None and not features.empty:
                # Make prediction
                prediction = self.inference_engine.predict(features)
                confidence = self.inference_engine.predict_confidence(features)
                
                # Create result
                result = {
                    'timestamp': current_time,
                    'window_start': window_start,
                    'window_end': current_time,
                    'packet_count': len(window_packets),
                    'prediction': int(prediction[0]) if prediction is not None else 0,
                    'confidence': float(confidence[0]) if confidence is not None else 0.0,
                    'reel_traffic_percentage': float(prediction[0] * confidence[0] * 100) if prediction is not None and confidence is not None else 0.0
                }
                
                # Send result to output queue
                try:
                    self.output_queue.put(result, timeout=0.1)
                except:
                    pass  # Queue full, skip result
                
                self.stats['classifications'] += 1
                self.stats['last_classification'] = current_time
                
                logger.debug(
                    f"Classification: {result['prediction']} "
                    f"(confidence: {result['confidence']:.3f}, "
                    f"packets: {result['packet_count']})"
                )
        
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            self.stats['errors'] += 1
    
    def _extract_packet_info(self, packet) -> Optional[Dict]:
        """
        Extract relevant information from packet for feature extraction.
        
        Args:
            packet: Scapy packet object
            
        Returns:
            Dictionary with packet information or None if not relevant
        """
        try:
            info = {
                'packet_size': len(packet),
                'protocol': 'OTHER'
            }
            
            # Check for IP layer
            if IP in packet:
                ip_layer = packet[IP]
                info.update({
                    'src_ip': ip_layer.src,
                    'dst_ip': ip_layer.dst,
                    'ttl': ip_layer.ttl,
                    'flags': ip_layer.flags
                })
                
                # Determine direction (simplified)
                # In real implementation, you'd have better logic for this
                if ip_layer.src.startswith('192.168.') or ip_layer.src.startswith('10.'):
                    info['direction'] = 'up'
                else:
                    info['direction'] = 'down'
            else:
                info['direction'] = 'unknown'
            
            # Check for TCP layer
            if TCP in packet:
                tcp_layer = packet[TCP]
                info.update({
                    'protocol': 'TCP',
                    'src_port': tcp_layer.sport,
                    'dst_port': tcp_layer.dport,
                    'flags': tcp_layer.flags,
                    'window_size': tcp_layer.window
                })
                
                # Check for HTTPS (port 443)
                if tcp_layer.sport == 443 or tcp_layer.dport == 443:
                    info['is_https'] = True
                    info['application'] = 'HTTPS'
                else:
                    info['is_https'] = False
            
            # Check for UDP layer
            elif UDP in packet:
                udp_layer = packet[UDP]
                info.update({
                    'protocol': 'UDP',
                    'src_port': udp_layer.sport,
                    'dst_port': udp_layer.dport
                })
            
            # Check for HTTP
            if HTTPRequest in packet or HTTPResponse in packet:
                info['application'] = 'HTTP'
            
            return info
        
        except Exception as e:
            logger.debug(f"Error extracting packet info: {e}")
            return None
    
    def _get_default_interface(self) -> str:
        """Get default network interface."""
        try:
            # Try to get default route interface
            conf = scapy.conf
            if hasattr(conf, 'iface') and conf.iface:
                return conf.iface
            
            # Fallback to first available interface
            interfaces = scapy.get_if_list()
            for iface in interfaces:
                if iface != 'lo' and not iface.startswith('docker'):
                    return iface
            
            return interfaces[0] if interfaces else 'any'
        
        except Exception as e:
            logger.warning(f"Could not determine default interface: {e}")
            return 'any'
    
    def _log_stats(self):
        """Log current statistics."""
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
            packets_per_sec = self.stats['total_packets'] / runtime if runtime > 0 else 0
            
            logger.info(
                f"Runtime: {runtime:.1f}s, "
                f"Packets: {self.stats['total_packets']} ({packets_per_sec:.1f}/s), "
                f"Classifications: {self.stats['classifications']}, "
                f"Errors: {self.stats['errors']}"
            )
    
    def _log_final_stats(self):
        """Log final statistics."""
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
            
            logger.info("=" * 50)
            logger.info("FINAL STATISTICS")
            logger.info("=" * 50)
            logger.info(f"Total Runtime: {runtime:.2f} seconds")
            logger.info(f"Total Packets Captured: {self.stats['total_packets']}")
            logger.info(f"Packets Processed: {self.stats['processed_packets']}")
            logger.info(f"Classifications Made: {self.stats['classifications']}")
            logger.info(f"Errors Encountered: {self.stats['errors']}")
            
            if runtime > 0:
                logger.info(f"Capture Rate: {self.stats['total_packets'] / runtime:.2f} packets/sec")
                logger.info(f"Processing Rate: {self.stats['processed_packets'] / runtime:.2f} packets/sec")
                logger.info(f"Classification Rate: {self.stats['classifications'] / runtime:.2f} classifications/sec")
            
            logger.info("=" * 50)
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        stats = self.stats.copy()
        if stats['start_time']:
            stats['runtime'] = time.time() - stats['start_time']
        return stats


async def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Real-time traffic capture and classification")
    
    parser.add_argument(
        '--interface', '-i',
        type=str,
        default='auto',
        help='Network interface to capture from (default: auto)'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=float,
        help='Capture duration in seconds (default: infinite)'
    )
    
    parser.add_argument(
        '--window-size', '-w',
        type=float,
        default=3.0,
        help='Feature extraction window size in seconds (default: 3.0)'
    )
    
    parser.add_argument(
        '--max-packets',
        type=int,
        default=1000,
        help='Maximum packets per window (default: 1000)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for classification results (JSON format)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dashboard-url',
        type=str,
        help='WebSocket URL to send results to dashboard'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(level=log_level)
    
    logger.info("Starting real-time traffic capture system")
    
    # Create output queue for results
    result_queue = Queue()
    
    # Initialize capture system
    capture = RealTimeCapture(
        interface=args.interface,
        window_size=args.window_size,
        max_packets_per_window=args.max_packets,
        output_queue=result_queue
    )
    
    # Start result handler if output file specified
    result_handler = None
    if args.output or args.dashboard_url:
        result_handler = threading.Thread(
            target=handle_results,
            args=(result_queue, args.output, args.dashboard_url),
            daemon=True
        )
        result_handler.start()
    
    try:
        # Start capture
        capture.start_capture(duration=args.duration)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return 1
    
    finally:
        capture.stop_capture()
        
        # Stop result handler
        if result_handler and result_handler.is_alive():
            result_handler.join(timeout=2)
    
    return 0


def handle_results(result_queue: Queue, output_file: Optional[str], dashboard_url: Optional[str]):
    """
    Handle classification results from the capture system.
    
    Args:
        result_queue: Queue containing classification results
        output_file: Optional file to write results to
        dashboard_url: Optional WebSocket URL for dashboard
    """
    results = []
    
    try:
        while True:
            try:
                result = result_queue.get(timeout=1.0)
                results.append(result)
                
                # Log result
                logger.info(
                    f"Classification: {'REEL' if result['prediction'] else 'REGULAR'} "
                    f"(confidence: {result['confidence']:.3f}, "
                    f"packets: {result['packet_count']})"
                )
                
                # Send to dashboard if URL provided
                if dashboard_url:
                    try:
                        import websockets
                        import asyncio
                        
                        async def send_to_dashboard():
                            async with websockets.connect(dashboard_url) as websocket:
                                await websocket.send(json.dumps(result))
                        
                        asyncio.run(send_to_dashboard())
                    except Exception as e:
                        logger.debug(f"Could not send to dashboard: {e}")
                
                # Write to file periodically
                if output_file and len(results) % 10 == 0:
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2)
            
            except Empty:
                continue
            
            except Exception as e:
                logger.error(f"Error handling result: {e}")
                break
    
    finally:
        # Write final results to file
        if output_file and results:
            try:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Wrote {len(results)} results to {output_file}")
            except Exception as e:
                logger.error(f"Error writing results to file: {e}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
