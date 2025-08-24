#!/usr/bin/env python3
"""
PC Traffic Capture Module

Captures network traffic on PC/Linux systems using Scapy and tcpdump.
Supports both live capture and offline processing of packet captures.
"""

import os
import sys
import time
import logging
import argparse
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
from collections import deque

import scapy.all as scapy
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.http import HTTP
import psutil

from ..utils.logging import setup_logger
from ..utils.network import get_interface_info, is_interface_available


class PCCapture:
    """
    PC-based network traffic capture using Scapy.
    
    Provides real-time packet capture with filtering for social media applications
    and privacy-preserving metadata extraction.
    """
    
    def __init__(self, interface: str = None, output_dir: str = "data/raw"):
        """
        Initialize PC capture instance.
        
        Args:
            interface: Network interface to capture from (auto-detect if None)
            output_dir: Directory to store capture files
        """
        # Create the logger FIRST, so all other methods can use it
        # Note: We can't use self.interface in the name yet, so we use a generic one
        self.logger = setup_logger("pc_capture")
        
        self.interface = interface or self._get_default_interface()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update the logger's name now that we have the interface
        self.logger.name = f"pc_capture_{self.interface}"
        self.is_capturing = False
        self.capture_thread = None
        self.packets_captured = 0
        
        # Social media domains for filtering
        self.target_domains = {
            'instagram.com', 'cdninstagram.com', 'facebook.com', 'fbcdn.net',
            'tiktok.com', 'tiktokcdn.com', 'musical.ly',
            'youtube.com', 'youtubei.googleapis.com', 'ytimg.com',
            'twitter.com', 'twimg.com', 'x.com',
            'snapchat.com', 'sc-cdn.net'
        }
        
        # Packet buffer for real-time processing
        self.max_buffer_size = 200000  # Increased from 10000 to handle more packets
        self.packet_buffer = deque(maxlen=self.max_buffer_size)  # More efficient than list
        self.buffer_lock = threading.Lock()
        
    def _get_default_interface(self) -> str:
        """Get the default network interface for capture."""
        try:
            # Get network interfaces
            interfaces = psutil.net_if_addrs()
            
            # Prefer active ethernet/wifi interfaces
            for iface_name, addresses in interfaces.items():
                if any(addr.family.name == 'AF_INET' for addr in addresses):
                    if 'eth' in iface_name.lower() or 'wlan' in iface_name.lower() or 'wifi' in iface_name.lower():
                        if is_interface_available(iface_name):
                            return iface_name
            
            # Fallback to first available interface
            for iface_name in interfaces.keys():
                if is_interface_available(iface_name):
                    return iface_name
                    
            raise RuntimeError("No suitable network interface found")
            
        except Exception as e:
            self.logger.error(f"Error detecting interface: {e}")
            return "eth0"  # Fallback
    
    def _create_filter(self, apps: List[str] = None) -> str:
        """
        Create BPF filter for target applications.
        A simple filter is more reliable than one that depends on DNS lookups.
        """
        # This filter captures all standard web traffic and is much more robust.
        return "tcp port 80 or tcp port 443"
    
    def _packet_handler(self, packet):
        """
        Handle captured packets - extract metadata only.
        
        Args:
            packet: Scapy packet object
        """
        try:
            # Extract metadata without accessing payload
            metadata = self._extract_packet_metadata(packet)
            
            if metadata:
                with self.buffer_lock:
                    self.packet_buffer.append(metadata)
                    self.packets_captured += 1
                    # Note: deque with maxlen automatically handles size limit efficiently
                
                # Log progress
                if self.packets_captured % 1000 == 0:
                    self.logger.info(f"Captured {self.packets_captured} packets")
                    
        except Exception as e:
            self.logger.error(f"Error processing packet: {e}")
    
    def _extract_packet_metadata(self, packet) -> Optional[Dict[str, Any]]:
        """
        Extract privacy-preserving metadata from packet.
        
        Args:
            packet: Scapy packet object
            
        Returns:
            Dictionary of packet metadata or None if not relevant
        """
        try:
            # Basic packet info
            metadata = {
                'timestamp': float(packet.time),
                'size': len(packet),
                'protocol': None,
                'src_port': None,
                'dst_port': None,
                'direction': 'unknown',
                'tls_record': False,
                'tcp_flags': None
            }
            
            # IP layer analysis
            if IP in packet:
                ip_layer = packet[IP]
                
                # Determine direction (simplified heuristic)
                # In real deployment, you'd have more sophisticated logic
                if ip_layer.src.startswith('192.168.') or ip_layer.src.startswith('10.'):
                    metadata['direction'] = 'outbound'
                else:
                    metadata['direction'] = 'inbound'
            
            # TCP layer analysis
            if TCP in packet:
                tcp_layer = packet[TCP]
                metadata['protocol'] = 'TCP'
                metadata['src_port'] = tcp_layer.sport
                metadata['dst_port'] = tcp_layer.dport
                metadata['tcp_flags'] = str(tcp_layer.flags)
                
                # Check for TLS traffic (port 443 or TLS handshake patterns)
                if tcp_layer.dport == 443 or tcp_layer.sport == 443:
                    metadata['tls_record'] = True
                    
                    # TLS record detection (simplified)
                    if len(packet.payload.payload.payload) > 5:
                        # Check for TLS record header patterns
                        payload_start = bytes(packet.payload.payload.payload)[:5]
                        if payload_start[0] in [0x16, 0x17, 0x14, 0x15]:  # TLS record types
                            metadata['tls_record'] = True
            
            # UDP layer analysis
            elif UDP in packet:
                udp_layer = packet[UDP]
                metadata['protocol'] = 'UDP'
                metadata['src_port'] = udp_layer.sport
                metadata['dst_port'] = udp_layer.dport
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {e}")
            return None
    
    def start_capture(self, duration: Optional[int] = None, apps: List[str] = None, output_file_path: Optional[str] = None) -> str:
        """
        Start packet capture.
        
        Args:
            duration: Capture duration in seconds (None for indefinite)
            apps: List of specific apps to capture
            output_file_path: Specific output file path (overrides default naming)
            
        Returns:
            Path to output file
        """
        if self.is_capturing:
            raise RuntimeError("Capture already in progress")
        
        # This logic correctly uses the --output argument
        if output_file_path:
            output_file = Path(output_file_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            apps_str = "_".join(apps) if apps else "all"
            output_file = self.output_dir / f"capture_{apps_str}_{timestamp}.pcap"
        
        self.logger.info(f"Starting capture on interface {self.interface}")
        self.logger.info(f"Output file: {output_file}")
        
        try:
            # Create BPF filter
            packet_filter = self._create_filter(apps)
            self.logger.info(f"Using filter: {packet_filter}")
            
            self.is_capturing = True
            
            # Start capture in separate thread
            self.capture_thread = threading.Thread(
                target=self._run_capture,
                args=(output_file, packet_filter, duration)
            )
            self.capture_thread.start()
            
            return str(output_file)
            
        except Exception as e:
            self.is_capturing = False
            self.logger.error(f"Error starting capture: {e}")
            raise
    
    def _run_capture(self, output_file: Path, packet_filter: str, duration: Optional[int]):
        """
        Run the actual packet capture.
        
        Args:
            output_file: Path to save capture
            packet_filter: BPF filter string
            duration: Capture duration in seconds
        """
        try:
            # Start Scapy capture
            self.logger.info("Starting packet capture...")
            
            scapy.sniff(
                iface=self.interface,
                filter=packet_filter,
                prn=self._packet_handler,
                timeout=duration,
                store=False  # Don't store packets in memory to save RAM
            )
            
        except KeyboardInterrupt:
            self.logger.info("Capture interrupted by user")
        except Exception as e:
            self.logger.error(f"Capture error: {e}")
        finally:
            self.is_capturing = False
            self.logger.info(f"Capture completed. Total packets: {self.packets_captured}")
            
            # Save packet metadata to file
            self._save_metadata(output_file)
    
    def _save_metadata(self, output_file: Path):
        """
        Save captured packet metadata to file.
        
        Args:
            output_file: Output file path
        """
        try:
            import json
            
            # Save as JSON for easy processing
            metadata_file = output_file.with_suffix('.json')
            
            with self.buffer_lock:
                # Convert the deque to a standard list right here
                metadata_list = list(self.packet_buffer)
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata_list, f, indent=2)  # Now it's a list, and json can handle it
            
            self.logger.info(f"Saved {len(metadata_list)} packet metadata records to {metadata_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    def stop_capture(self):
        """Stop ongoing packet capture."""
        if not self.is_capturing:
            self.logger.warning("No capture in progress")
            return
        
        self.logger.info("Stopping capture...")
        self.is_capturing = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get capture statistics.
        
        Returns:
            Dictionary of capture statistics
        """
        with self.buffer_lock:
            buffer_size = len(self.packet_buffer)
        
        return {
            'packets_captured': self.packets_captured,
            'buffer_size': buffer_size,
            'is_capturing': self.is_capturing,
            'interface': self.interface
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="PC Network Traffic Capture")
    parser.add_argument('--interface', '-i', help="Network interface to use")
    parser.add_argument('--duration', '-d', type=int, help="Capture duration in seconds")
    parser.add_argument('--apps', '-a', nargs='+', help="Specific apps to capture")
    parser.add_argument('--output', '-o', default="data/raw", help="Output directory")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize capture
        capture = PCCapture(interface=args.interface)  # output_dir is no longer needed here
        
        # Start capture
        output_file = capture.start_capture(duration=args.duration, apps=args.apps, output_file_path=args.output)
        print(f"Capture started. Output: {output_file}")
        
        # Wait for completion or user interrupt
        try:
            if args.duration:
                time.sleep(args.duration)
            else:
                print("Press Ctrl+C to stop capture...")
                while capture.is_capturing:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping capture...")
        
        # Stop capture
        capture.stop_capture()
        
        # Print statistics
        stats = capture.get_statistics()
        print(f"Capture completed. Packets captured: {stats['packets_captured']}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
