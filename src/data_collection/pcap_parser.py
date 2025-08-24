#!/usr/bin/env python3
"""
PCAP Parser Module

Parses captured .pcap files and extracts privacy-preserving network metadata
for training and inference. Supports both PyShark and Scapy backends.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional, Iterator
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import pyshark
    PYSHARK_AVAILABLE = True
except ImportError:
    PYSHARK_AVAILABLE = False

try:
    import scapy.all as scapy
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.http import HTTP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

from ..utils.logging import setup_logger


class PcapParser:
    """
    PCAP file parser for extracting network traffic metadata.
    
    Extracts privacy-preserving features from packet captures without
    accessing payload content.
    """
    
    def __init__(self, backend: str = "auto"):
        """
        Initialize PCAP parser.
        
        Args:
            backend: Parser backend ('pyshark', 'scapy', 'auto')
        """
        self.logger = setup_logger("pcap_parser")
        
        # Determine backend
        if backend == "auto":
            if PYSHARK_AVAILABLE:
                self.backend = "pyshark"
            elif SCAPY_AVAILABLE:
                self.backend = "scapy"
            else:
                raise RuntimeError("Neither PyShark nor Scapy available")
        else:
            self.backend = backend
            
        # Validate backend availability
        if self.backend == "pyshark" and not PYSHARK_AVAILABLE:
            raise RuntimeError("PyShark not available")
        elif self.backend == "scapy" and not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy not available")
        
        self.logger.info(f"Using backend: {self.backend}")
        
        # Feature extraction configuration
        self.features_config = {
            'extract_basic': True,      # Basic packet features
            'extract_temporal': True,   # Timing features
            'extract_tls': True,        # TLS-specific features
            'extract_flow': True,       # Flow-level features
            'sequence_length': 100,     # Max sequence length for CNN
        }
        
        # Social media port mappings
        self.known_ports = {
            80: 'http',
            443: 'https',
            53: 'dns',
            80: 'http_alt',
            8080: 'http_proxy'
        }
    
    def parse_pcap(self, pcap_file: str, output_format: str = "csv") -> str:
        """
        Parse PCAP file and extract features.
        
        Args:
            pcap_file: Path to PCAP file
            output_format: Output format ('csv', 'json')
            
        Returns:
            Path to output file
        """
        pcap_path = Path(pcap_file)
        if not pcap_path.exists():
            raise FileNotFoundError(f"PCAP file not found: {pcap_file}")
        
        self.logger.info(f"Parsing PCAP file: {pcap_file}")
        self.logger.info(f"File size: {pcap_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Extract features based on backend
        if self.backend == "pyshark":
            features = self._parse_with_pyshark(pcap_file)
        else:
            features = self._parse_with_scapy(pcap_file)
        
        # Generate output filename
        output_file = pcap_path.with_suffix(f'.{output_format}')
        
        # Save features
        if output_format == "csv":
            self._save_csv(features, output_file)
        else:
            self._save_json(features, output_file)
        
        self.logger.info(f"Extracted {len(features)} feature records")
        self.logger.info(f"Output saved to: {output_file}")
        
        return str(output_file)
    
    def _parse_with_pyshark(self, pcap_file: str) -> List[Dict[str, Any]]:
        """
        Parse PCAP using PyShark backend.
        
        Args:
            pcap_file: Path to PCAP file
            
        Returns:
            List of feature dictionaries
        """
        features = []
        session_flows = {}
        
        try:
            # Open PCAP file
            cap = pyshark.FileCapture(pcap_file)
            
            packet_count = 0
            for packet in cap:
                packet_count += 1
                
                try:
                    # Extract basic features
                    basic_features = self._extract_basic_features_pyshark(packet)
                    if basic_features:
                        features.append(basic_features)
                        
                        # Update flow tracking
                        flow_key = self._get_flow_key(basic_features)
                        if flow_key not in session_flows:
                            session_flows[flow_key] = []
                        session_flows[flow_key].append(basic_features)
                    
                    # Progress logging
                    if packet_count % 10000 == 0:
                        self.logger.info(f"Processed {packet_count} packets")
                        
                except Exception as e:
                    self.logger.warning(f"Error processing packet {packet_count}: {e}")
                    continue
            
            cap.close()
            
            # Add flow-level features
            if self.features_config['extract_flow']:
                features = self._add_flow_features(features, session_flows)
            
        except Exception as e:
            self.logger.error(f"Error parsing with PyShark: {e}")
            raise
        
        return features
    
    def _parse_with_scapy(self, pcap_file: str) -> List[Dict[str, Any]]:
        """
        Parse PCAP using Scapy backend.
        
        Args:
            pcap_file: Path to PCAP file
            
        Returns:
            List of feature dictionaries
        """
        features = []
        session_flows = {}
        
        try:
            # Read PCAP file
            packets = scapy.rdpcap(pcap_file)
            
            self.logger.info(f"Loaded {len(packets)} packets")
            
            for i, packet in enumerate(packets):
                try:
                    # Extract basic features
                    basic_features = self._extract_basic_features_scapy(packet)
                    if basic_features:
                        features.append(basic_features)
                        
                        # Update flow tracking
                        flow_key = self._get_flow_key(basic_features)
                        if flow_key not in session_flows:
                            session_flows[flow_key] = []
                        session_flows[flow_key].append(basic_features)
                    
                    # Progress logging
                    if (i + 1) % 10000 == 0:
                        self.logger.info(f"Processed {i + 1} packets")
                        
                except Exception as e:
                    self.logger.warning(f"Error processing packet {i}: {e}")
                    continue
            
            # Add flow-level features
            if self.features_config['extract_flow']:
                features = self._add_flow_features(features, session_flows)
            
        except Exception as e:
            self.logger.error(f"Error parsing with Scapy: {e}")
            raise
        
        return features
    
    def _extract_basic_features_pyshark(self, packet) -> Optional[Dict[str, Any]]:
        """Extract basic features using PyShark packet."""
        try:
            features = {
                'timestamp': float(packet.sniff_timestamp),
                'length': int(packet.length),
                'protocol': None,
                'src_port': None,
                'dst_port': None,
                'direction': 'unknown',
                'tcp_flags': None,
                'tls_record': False,
                'frame_number': int(packet.number)
            }
            
            # IP layer
            if hasattr(packet, 'ip'):
                ip_layer = packet.ip
                
                # Simple direction heuristic
                if str(ip_layer.src).startswith(('192.168.', '10.', '172.')):
                    features['direction'] = 'outbound'
                else:
                    features['direction'] = 'inbound'
            
            # TCP layer
            if hasattr(packet, 'tcp'):
                tcp_layer = packet.tcp
                features['protocol'] = 'tcp'
                features['src_port'] = int(tcp_layer.srcport)
                features['dst_port'] = int(tcp_layer.dstport)
                features['tcp_flags'] = str(tcp_layer.flags)
                
                # TLS detection
                if int(tcp_layer.dstport) == 443 or int(tcp_layer.srcport) == 443:
                    features['tls_record'] = True
            
            # UDP layer
            elif hasattr(packet, 'udp'):
                udp_layer = packet.udp
                features['protocol'] = 'udp'
                features['src_port'] = int(udp_layer.srcport)
                features['dst_port'] = int(udp_layer.dstport)
            
            return features
            
        except Exception as e:
            self.logger.debug(f"Error extracting PyShark features: {e}")
            return None
    
    def _extract_basic_features_scapy(self, packet) -> Optional[Dict[str, Any]]:
        """Extract basic features using Scapy packet."""
        try:
            features = {
                'timestamp': float(packet.time),
                'length': len(packet),
                'protocol': None,
                'src_port': None,
                'dst_port': None,
                'direction': 'unknown',
                'tcp_flags': None,
                'tls_record': False,
                'frame_number': None  # Not available in Scapy by default
            }
            
            # IP layer
            if IP in packet:
                ip_layer = packet[IP]
                
                # Simple direction heuristic
                if str(ip_layer.src).startswith(('192.168.', '10.', '172.')):
                    features['direction'] = 'outbound'
                else:
                    features['direction'] = 'inbound'
            
            # TCP layer
            if TCP in packet:
                tcp_layer = packet[TCP]
                features['protocol'] = 'tcp'
                features['src_port'] = tcp_layer.sport
                features['dst_port'] = tcp_layer.dport
                features['tcp_flags'] = str(tcp_layer.flags)
                
                # TLS detection
                if tcp_layer.dport == 443 or tcp_layer.sport == 443:
                    features['tls_record'] = True
            
            # UDP layer
            elif UDP in packet:
                udp_layer = packet[UDP]
                features['protocol'] = 'udp'
                features['src_port'] = udp_layer.sport
                features['dst_port'] = udp_layer.dport
            
            return features
            
        except Exception as e:
            self.logger.debug(f"Error extracting Scapy features: {e}")
            return None
    
    def _get_flow_key(self, features: Dict[str, Any]) -> str:
        """Generate flow key for grouping packets."""
        src_port = features.get('src_port', 0)
        dst_port = features.get('dst_port', 0)
        protocol = features.get('protocol', 'unknown')
        
        # Create consistent flow key (sort ports for bidirectional flows)
        ports = sorted([src_port, dst_port])
        return f"{protocol}_{ports[0]}_{ports[1]}"
    
    def _add_flow_features(self, features: List[Dict[str, Any]], 
                          flows: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Add flow-level features to packet features."""
        
        # Calculate flow statistics
        flow_stats = {}
        for flow_key, flow_packets in flows.items():
            if len(flow_packets) < 2:
                continue
                
            # Sort by timestamp
            flow_packets.sort(key=lambda x: x['timestamp'])
            
            # Calculate flow features
            flow_stats[flow_key] = self._calculate_flow_statistics(flow_packets)
        
        # Add flow features to each packet
        for feature_record in features:
            flow_key = self._get_flow_key(feature_record)
            if flow_key in flow_stats:
                feature_record.update(flow_stats[flow_key])
        
        return features
    
    def _calculate_flow_statistics(self, flow_packets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistical features for a flow."""
        
        # Basic statistics
        timestamps = [p['timestamp'] for p in flow_packets]
        lengths = [p['length'] for p in flow_packets]
        
        # Temporal features
        inter_arrival_times = []
        for i in range(1, len(timestamps)):
            inter_arrival_times.append(timestamps[i] - timestamps[i-1])
        
        # Direction-based statistics
        outbound_packets = [p for p in flow_packets if p['direction'] == 'outbound']
        inbound_packets = [p for p in flow_packets if p['direction'] == 'inbound']
        
        outbound_bytes = sum(p['length'] for p in outbound_packets)
        inbound_bytes = sum(p['length'] for p in inbound_packets)
        
        return {
            'flow_duration': timestamps[-1] - timestamps[0],
            'flow_packet_count': len(flow_packets),
            'flow_total_bytes': sum(lengths),
            'flow_avg_packet_size': np.mean(lengths),
            'flow_std_packet_size': np.std(lengths),
            'flow_outbound_packets': len(outbound_packets),
            'flow_inbound_packets': len(inbound_packets),
            'flow_outbound_bytes': outbound_bytes,
            'flow_inbound_bytes': inbound_bytes,
            'flow_bytes_ratio': outbound_bytes / max(inbound_bytes, 1),
            'flow_mean_iat': np.mean(inter_arrival_times) if inter_arrival_times else 0,
            'flow_std_iat': np.std(inter_arrival_times) if inter_arrival_times else 0,
            'flow_burst_rate': len(flow_packets) / max(timestamps[-1] - timestamps[0], 1)
        }
    
    def extract_sequences(self, pcap_file: str, output_file: str = None) -> str:
        """
        Extract packet sequences for CNN training.
        
        Args:
            pcap_file: Path to PCAP file
            output_file: Output file path (auto-generated if None)
            
        Returns:
            Path to sequence file
        """
        if output_file is None:
            output_file = Path(pcap_file).with_suffix('.sequences.csv')
        
        self.logger.info(f"Extracting sequences from: {pcap_file}")
        
        # Parse PCAP for basic features
        if self.backend == "pyshark":
            features = self._parse_with_pyshark(pcap_file)
        else:
            features = self._parse_with_scapy(pcap_file)
        
        # Group by flows and extract sequences
        flows = {}
        for feature in features:
            flow_key = self._get_flow_key(feature)
            if flow_key not in flows:
                flows[flow_key] = []
            flows[flow_key].append(feature)
        
        # Extract sequences
        sequences = []
        for flow_key, flow_packets in flows.items():
            if len(flow_packets) < 10:  # Minimum sequence length
                continue
            
            # Sort by timestamp
            flow_packets.sort(key=lambda x: x['timestamp'])
            
            # Extract packet size and timing sequences
            packet_sizes = [p['length'] for p in flow_packets]
            inter_arrival_times = [0]  # First packet has no IAT
            
            for i in range(1, len(flow_packets)):
                iat = flow_packets[i]['timestamp'] - flow_packets[i-1]['timestamp']
                inter_arrival_times.append(iat * 1000)  # Convert to milliseconds
            
            directions = [1 if p['direction'] == 'outbound' else 0 for p in flow_packets]
            
            # Truncate or pad to fixed length
            seq_len = self.features_config['sequence_length']
            packet_sizes = self._pad_or_truncate(packet_sizes, seq_len)
            inter_arrival_times = self._pad_or_truncate(inter_arrival_times, seq_len)
            directions = self._pad_or_truncate(directions, seq_len)
            
            sequences.append({
                'flow_id': flow_key,
                'packet_sizes': packet_sizes,
                'inter_arrival_times': inter_arrival_times,
                'directions': directions,
                'flow_length': len(flow_packets)
            })
        
        # Save sequences
        self._save_sequences(sequences, output_file)
        
        self.logger.info(f"Extracted {len(sequences)} sequences")
        self.logger.info(f"Sequences saved to: {output_file}")
        
        return str(output_file)
    
    def _pad_or_truncate(self, sequence: List, target_length: int, pad_value: int = 0) -> List:
        """Pad or truncate sequence to target length."""
        if len(sequence) >= target_length:
            return sequence[:target_length]
        else:
            return sequence + [pad_value] * (target_length - len(sequence))
    
    def _save_csv(self, features: List[Dict[str, Any]], output_file: Path):
        """Save features to CSV file."""
        if not features:
            self.logger.warning("No features to save")
            return
        
        df = pd.DataFrame(features)
        df.to_csv(output_file, index=False)
    
    def _save_json(self, features: List[Dict[str, Any]], output_file: Path):
        """Save features to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(features, f, indent=2)
    
    def _save_sequences(self, sequences: List[Dict[str, Any]], output_file: Path):
        """Save sequences to CSV file."""
        if not sequences:
            self.logger.warning("No sequences to save")
            return
        
        # Convert to DataFrame with list columns
        df = pd.DataFrame(sequences)
        
        # Convert lists to strings for CSV storage
        df['packet_sizes'] = df['packet_sizes'].apply(lambda x: ','.join(map(str, x)))
        df['inter_arrival_times'] = df['inter_arrival_times'].apply(lambda x: ','.join(map(str, x)))
        df['directions'] = df['directions'].apply(lambda x: ','.join(map(str, x)))
        
        df.to_csv(output_file, index=False)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="PCAP Parser for Traffic Analysis")
    parser.add_argument('input', help="Input PCAP file")
    parser.add_argument('--output', '-o', help="Output file path")
    parser.add_argument('--format', '-f', choices=['csv', 'json'], default='csv',
                       help="Output format")
    parser.add_argument('--backend', '-b', choices=['pyshark', 'scapy', 'auto'], 
                       default='auto', help="Parser backend")
    parser.add_argument('--sequences', '-s', action='store_true',
                       help="Extract sequences for CNN training")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize parser
        parser = PcapParser(backend=args.backend)
        
        if args.sequences:
            # Extract sequences
            output_file = parser.extract_sequences(args.input, args.output)
        else:
            # Extract features
            if args.output:
                # Set format based on output file extension
                if args.output.endswith('.json'):
                    format_type = 'json'
                else:
                    format_type = args.format
            else:
                format_type = args.format
            
            output_file = parser.parse_pcap(args.input, format_type)
        
        print(f"Parsing completed. Output: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
