#!/usr/bin/env python3
"""
Feature Extractor Module

Extracts comprehensive features from network traffic data for training
machine learning models to classify reel/video vs non-reel traffic.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from scipy import stats
from collections import defaultdict
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logger import setup_logger


class FeatureExtractor:
    """
    Extract features from network traffic data for ML training.
    
    Computes comprehensive feature sets including basic traffic statistics,
    temporal patterns, TLS characteristics, and flow-level metrics.
    """
    
    def __init__(self, window_size: float = 3.0):
        """
        Initialize feature extractor.
        
        Args:
            window_size: Time window size in seconds for grouping packets
        """
        self.window_size = window_size
        self.logger = setup_logger("feature_extractor")
        
        # Feature categories to extract
        self.feature_categories = {
            'basic': True,      # Basic packet statistics
            'temporal': True,   # Timing-based features
            'tls': True,        # TLS-specific features
            'flow': True,       # Flow-level features
            'statistical': True # Statistical moments and distributions
        }
        
        # Port mappings for application identification
        self.app_ports = {
            443: 'https',
            80: 'http', 
            53: 'dns',
            993: 'imaps',
            995: 'pop3s',
            465: 'smtps'
        }
    
    def extract_features_from_pcap(self, pcap_data: pd.DataFrame, 
                                 labels: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Extract features from parsed PCAP data.
        
        Args:
            pcap_data: DataFrame with parsed packet data
            labels: Optional labels for supervised learning
            
        Returns:
            DataFrame with extracted features
        """
        self.logger.info(f"Extracting features from {len(pcap_data)} packets")
        
        # Group packets into time windows
        windows = self._create_time_windows(pcap_data)
        self.logger.info(f"Created {len(windows)} time windows")
        
        # Extract features for each window
        feature_records = []
        for window_id, window_data in windows.items():
            features = self._extract_window_features(window_data, window_id)
            feature_records.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(feature_records)
        
        # Add labels if provided
        if labels is not None:
            features_df = self._align_labels(features_df, labels)
        
        self.logger.info(f"Extracted {len(features_df)} feature vectors with {len(features_df.columns)} features")
        
        return features_df
    
    def _create_time_windows(self, pcap_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Group packets into time windows.
        
        Args:
            pcap_data: Packet data with timestamp column
            
        Returns:
            Dictionary mapping window_id to packet data
        """
        # Ensure timestamp column exists
        if 'timestamp' not in pcap_data.columns:
            raise ValueError("PCAP data must contain 'timestamp' column")
        
        # Sort by timestamp
        pcap_data = pcap_data.sort_values('timestamp').reset_index(drop=True)
        
        windows = {}
        start_time = pcap_data['timestamp'].min()
        end_time = pcap_data['timestamp'].max()
        
        window_start = start_time
        window_id = 0
        
        while window_start < end_time:
            window_end = window_start + self.window_size
            
            # Get packets in this window
            window_mask = (
                (pcap_data['timestamp'] >= window_start) & 
                (pcap_data['timestamp'] < window_end)
            )
            window_packets = pcap_data[window_mask].copy()
            
            if len(window_packets) > 0:
                windows[f"window_{window_id}"] = window_packets
            
            window_start = window_end
            window_id += 1
        
        return windows
    
    def _extract_window_features(self, window_data: pd.DataFrame, window_id: str) -> Dict[str, Any]:
        """
        Extract features from a single time window.
        
        Args:
            window_data: Packet data for this window
            window_id: Window identifier
            
        Returns:
            Dictionary of extracted features
        """
        features = {'window_id': window_id}
        
        # Basic features
        if self.feature_categories['basic']:
            features.update(self._extract_basic_features(window_data))
        
        # Temporal features
        if self.feature_categories['temporal']:
            features.update(self._extract_temporal_features(window_data))
        
        # TLS features
        if self.feature_categories['tls']:
            features.update(self._extract_tls_features(window_data))
        
        # Flow features
        if self.feature_categories['flow']:
            features.update(self._extract_flow_features(window_data))
        
        # Statistical features
        if self.feature_categories['statistical']:
            features.update(self._extract_statistical_features(window_data))
        
        return features
    
    def _extract_basic_features(self, window_data: pd.DataFrame) -> Dict[str, float]:
        """Extract basic traffic statistics."""
        features = {}
        
        # Total counts
        features['total_packets'] = len(window_data)
        features['total_bytes'] = window_data['length'].sum()
        
        # Direction-based features
        if 'direction' in window_data.columns:
            outbound = window_data[window_data['direction'] == 'outbound']
            inbound = window_data[window_data['direction'] == 'inbound']
            
            features['packets_outbound'] = len(outbound)
            features['packets_inbound'] = len(inbound)
            features['bytes_outbound'] = outbound['length'].sum()
            features['bytes_inbound'] = inbound['length'].sum()
            
            # Ratios
            total_packets = max(len(window_data), 1)
            total_bytes = max(window_data['length'].sum(), 1)
            
            features['packet_ratio_out_in'] = features['packets_outbound'] / max(features['packets_inbound'], 1)
            features['bytes_ratio_out_in'] = features['bytes_outbound'] / max(features['bytes_inbound'], 1)
            features['outbound_fraction'] = features['packets_outbound'] / total_packets
        else:
            # Default values if direction not available
            features['packets_outbound'] = 0
            features['packets_inbound'] = features['total_packets']
            features['bytes_outbound'] = 0
            features['bytes_inbound'] = features['total_bytes']
            features['packet_ratio_out_in'] = 0
            features['bytes_ratio_out_in'] = 0
            features['outbound_fraction'] = 0
        
        # Packet size statistics
        if len(window_data) > 0:
            packet_sizes = window_data['length']
            features['avg_packet_size'] = packet_sizes.mean()
            features['std_packet_size'] = packet_sizes.std()
            features['min_packet_size'] = packet_sizes.min()
            features['max_packet_size'] = packet_sizes.max()
            features['median_packet_size'] = packet_sizes.median()
        else:
            features.update({
                'avg_packet_size': 0, 'std_packet_size': 0,
                'min_packet_size': 0, 'max_packet_size': 0,
                'median_packet_size': 0
            })
        
        # Protocol distribution
        if 'protocol' in window_data.columns:
            protocol_counts = window_data['protocol'].value_counts(normalize=True)
            features['tcp_fraction'] = protocol_counts.get('tcp', 0)
            features['udp_fraction'] = protocol_counts.get('udp', 0)
        else:
            features['tcp_fraction'] = 0
            features['udp_fraction'] = 0
        
        return features
    
    def _extract_temporal_features(self, window_data: pd.DataFrame) -> Dict[str, float]:
        """Extract temporal patterns and timing features."""
        features = {}
        
        if len(window_data) < 2:
            # Not enough packets for temporal analysis
            return {
                'duration': 0, 'packet_rate': 0, 'mean_iat': 0,
                'std_iat': 0, 'min_iat': 0, 'max_iat': 0,
                'burst_rate': 0, 'throughput_slope': 0
            }
        
        timestamps = window_data['timestamp'].values
        timestamps = np.sort(timestamps)
        
        # Basic temporal metrics
        features['duration'] = timestamps[-1] - timestamps[0]
        features['packet_rate'] = len(window_data) / max(features['duration'], 0.001)
        
        # Inter-arrival times
        inter_arrival_times = np.diff(timestamps)
        features['mean_iat'] = np.mean(inter_arrival_times)
        features['std_iat'] = np.std(inter_arrival_times)
        features['min_iat'] = np.min(inter_arrival_times)
        features['max_iat'] = np.max(inter_arrival_times)
        
        # Burst detection
        burst_threshold = features['mean_iat'] / 2  # Packets arriving faster than half the mean IAT
        burst_packets = np.sum(inter_arrival_times < burst_threshold)
        features['burst_rate'] = burst_packets / len(inter_arrival_times)
        
        # Throughput analysis
        if features['duration'] > 0:
            # Calculate cumulative bytes over time
            cumulative_bytes = window_data.sort_values('timestamp')['length'].cumsum().values
            time_points = timestamps - timestamps[0]  # Relative time
            
            # Linear regression for throughput slope
            if len(time_points) > 1:
                slope, _, _, _, _ = stats.linregress(time_points, cumulative_bytes)
                features['throughput_slope'] = slope
            else:
                features['throughput_slope'] = 0
        else:
            features['throughput_slope'] = 0
        
        return features
    
    def _extract_tls_features(self, window_data: pd.DataFrame) -> Dict[str, float]:
        """Extract TLS-specific features."""
        features = {}
        
        # TLS traffic identification
        if 'tls_record' in window_data.columns:
            tls_packets = window_data[window_data['tls_record'] == True]
        elif 'dst_port' in window_data.columns:
            # Fallback: identify by port 443
            tls_packets = window_data[
                (window_data['dst_port'] == 443) | 
                (window_data.get('src_port', 0) == 443)
            ]
        else:
            tls_packets = pd.DataFrame()
        
        features['tls_packet_count'] = len(tls_packets)
        features['tls_fraction'] = len(tls_packets) / max(len(window_data), 1)
        
        if len(tls_packets) > 0:
            features['tls_avg_size'] = tls_packets['length'].mean()
            features['tls_std_size'] = tls_packets['length'].std()
            
            # TLS record periodicity (simplified)
            if len(tls_packets) > 1:
                tls_timestamps = tls_packets['timestamp'].values
                tls_iats = np.diff(np.sort(tls_timestamps))
                
                # Coefficient of variation as periodicity measure
                if np.mean(tls_iats) > 0:
                    features['tls_periodicity'] = np.std(tls_iats) / np.mean(tls_iats)
                else:
                    features['tls_periodicity'] = 0
            else:
                features['tls_periodicity'] = 0
        else:
            features['tls_avg_size'] = 0
            features['tls_std_size'] = 0
            features['tls_periodicity'] = 0
        
        return features
    
    def _extract_flow_features(self, window_data: pd.DataFrame) -> Dict[str, float]:
        """Extract flow-level features."""
        features = {}
        
        # Group packets by flows (simplified by port pairs)
        if 'src_port' in window_data.columns and 'dst_port' in window_data.columns:
            # Create flow identifiers
            flows = defaultdict(list)
            
            for _, packet in window_data.iterrows():
                # Create bidirectional flow key
                ports = sorted([packet.get('src_port', 0), packet.get('dst_port', 0)])
                flow_key = f"{ports[0]}_{ports[1]}"
                flows[flow_key].append(packet)
            
            features['unique_flows'] = len(flows)
            
            if flows:
                flow_sizes = [len(packets) for packets in flows.values()]
                features['avg_flow_size'] = np.mean(flow_sizes)
                features['std_flow_size'] = np.std(flow_sizes)
                features['max_flow_size'] = np.max(flow_sizes)
                
                # Dominant flow analysis
                max_flow_size = max(flow_sizes)
                features['dominant_flow_fraction'] = max_flow_size / len(window_data)
            else:
                features.update({
                    'avg_flow_size': 0, 'std_flow_size': 0,
                    'max_flow_size': 0, 'dominant_flow_fraction': 0
                })
        else:
            features.update({
                'unique_flows': 1, 'avg_flow_size': len(window_data),
                'std_flow_size': 0, 'max_flow_size': len(window_data),
                'dominant_flow_fraction': 1.0
            })
        
        return features
    
    def _extract_statistical_features(self, window_data: pd.DataFrame) -> Dict[str, float]:
        """Extract statistical distribution features."""
        features = {}
        
        if len(window_data) == 0:
            return {
                'skewness_packet_size': 0, 'kurtosis_packet_size': 0,
                'entropy_packet_size': 0, 'cv_packet_size': 0
            }
        
        packet_sizes = window_data['length'].values
        
        # Statistical moments
        features['skewness_packet_size'] = stats.skew(packet_sizes)
        features['kurtosis_packet_size'] = stats.kurtosis(packet_sizes)
        
        # Coefficient of variation
        mean_size = np.mean(packet_sizes)
        if mean_size > 0:
            features['cv_packet_size'] = np.std(packet_sizes) / mean_size
        else:
            features['cv_packet_size'] = 0
        
        # Entropy of packet size distribution
        hist, _ = np.histogram(packet_sizes, bins=20)
        hist = hist + 1e-10  # Avoid log(0)
        prob = hist / np.sum(hist)
        features['entropy_packet_size'] = -np.sum(prob * np.log2(prob))
        
        return features
    
    def _align_labels(self, features_df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """
        Align labels with feature windows.
        
        This is a placeholder implementation - in practice, you would need
        to map labels to time windows based on your labeling strategy.
        """
        # Simple approach: replicate labels for each window
        # In practice, you'd have more sophisticated label alignment
        
        num_windows = len(features_df)
        
        if len(labels) == 1:
            # Single label for all windows
            features_df['label'] = labels.iloc[0]
        elif len(labels) == num_windows:
            # Direct mapping
            features_df['label'] = labels.values
        else:
            # Interpolate or replicate labels
            label_indices = np.linspace(0, len(labels) - 1, num_windows).astype(int)
            features_df['label'] = labels.iloc[label_indices].values
        
        return features_df
    
    def extract_features_from_file(self, input_file: str, output_file: str = None) -> str:
        """
        Extract features from a CSV file of parsed packet data.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file (auto-generated if None)
            
        Returns:
            Path to output features file
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if output_file is None:
            output_file = input_path.with_suffix('.features.csv')
        
        self.logger.info(f"Extracting features from file: {input_file}")
        
        # Load packet data
        pcap_data = pd.read_csv(input_file)
        
        # Extract features
        features_df = self.extract_features_from_pcap(pcap_data)
        
        # Save features
        features_df.to_csv(output_file, index=False)
        
        self.logger.info(f"Features saved to: {output_file}")
        
        return str(output_file)
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all possible feature names.
        
        Returns:
            List of feature names
        """
        # This returns the complete list of features that could be extracted
        feature_names = [
            # Basic features
            'total_packets', 'total_bytes', 'packets_outbound', 'packets_inbound',
            'bytes_outbound', 'bytes_inbound', 'packet_ratio_out_in', 'bytes_ratio_out_in',
            'outbound_fraction', 'avg_packet_size', 'std_packet_size', 'min_packet_size',
            'max_packet_size', 'median_packet_size', 'tcp_fraction', 'udp_fraction',
            
            # Temporal features
            'duration', 'packet_rate', 'mean_iat', 'std_iat', 'min_iat', 'max_iat',
            'burst_rate', 'throughput_slope',
            
            # TLS features
            'tls_packet_count', 'tls_fraction', 'tls_avg_size', 'tls_std_size',
            'tls_periodicity',
            
            # Flow features
            'unique_flows', 'avg_flow_size', 'std_flow_size', 'max_flow_size',
            'dominant_flow_fraction',
            
            # Statistical features
            'skewness_packet_size', 'kurtosis_packet_size', 'entropy_packet_size',
            'cv_packet_size'
        ]
        
        return feature_names
