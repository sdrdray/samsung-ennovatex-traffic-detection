#!/usr/bin/env python3
"""
Real-time feature extraction for inference pipeline.

This module provides real-time feature extraction capabilities
optimized for low-latency inference in the traffic classification system.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
import threading
from queue import Queue, Empty

# Local imports
from src.features.extractor import FeatureExtractor
# from src.features.network_analyzer import NetworkAnalyzer  # File doesn't exist - commented out
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


class RealTimeFeatureExtractor:
    """
    Real-time feature extraction optimized for streaming data.
    
    This class maintains rolling windows of network data and efficiently
    extracts features for real-time classification.
    """
    
    def __init__(
        self,
        window_size: float = 3.0,
        update_interval: float = 0.5,
        max_buffer_size: int = 10000,
        feature_cache_size: int = 100
    ):
        """
        Initialize real-time feature extractor.
        
        Args:
            window_size: Time window for feature aggregation (seconds)
            update_interval: How often to extract features (seconds)
            max_buffer_size: Maximum packets to keep in buffer
            feature_cache_size: Size of feature cache for performance
        """
        self.window_size = window_size
        self.update_interval = update_interval
        self.max_buffer_size = max_buffer_size
        self.feature_cache_size = feature_cache_size
        
        # Core components
        self.feature_extractor = FeatureExtractor(window_size=window_size)
        # self.network_analyzer = NetworkAnalyzer()  # NetworkAnalyzer class doesn't exist - commented out
        
        # Data buffers
        self.packet_buffer = deque(maxlen=max_buffer_size)
        self.feature_cache = deque(maxlen=feature_cache_size)
        
        # State tracking
        self.last_extraction_time = 0
        self.extraction_count = 0
        self.buffer_lock = threading.Lock()
        
        # Performance tracking
        self.stats = {
            'packets_processed': 0,
            'features_extracted': 0,
            'cache_hits': 0,
            'extraction_times': deque(maxlen=100),
            'buffer_sizes': deque(maxlen=100)
        }
        
        logger.info(f"Initialized real-time feature extractor (window: {window_size}s)")
    
    def add_packet(self, packet_info: Dict) -> None:
        """
        Add a packet to the buffer for feature extraction.
        
        Args:
            packet_info: Dictionary containing packet metadata
        """
        with self.buffer_lock:
            # Add timestamp if not present
            if 'timestamp' not in packet_info:
                packet_info['timestamp'] = time.time()
            
            self.packet_buffer.append(packet_info)
            self.stats['packets_processed'] += 1
            
            # Clean old packets
            self._clean_old_packets()
    
    def extract_features_now(self, force: bool = False) -> Optional[pd.DataFrame]:
        """
        Extract features from current buffer state.
        
        Args:
            force: Force extraction even if update interval not reached
            
        Returns:
            DataFrame with extracted features or None if no data
        """
        current_time = time.time()
        
        # Check if extraction is needed
        if not force and (current_time - self.last_extraction_time) < self.update_interval:
            return None
        
        start_time = time.time()
        
        try:
            with self.buffer_lock:
                if not self.packet_buffer:
                    return None
                
                # Get packets in current window
                window_start = current_time - self.window_size
                window_packets = [
                    p for p in self.packet_buffer 
                    if p.get('timestamp', 0) >= window_start
                ]
                
                if not window_packets:
                    return None
                
                # Convert to DataFrame for processing
                df = pd.DataFrame(window_packets)
                
                # Extract features
                features = self._extract_window_features(df)
                
                if features is not None:
                    self.last_extraction_time = current_time
                    self.extraction_count += 1
                    self.stats['features_extracted'] += 1
                    
                    # Cache features
                    self.feature_cache.append({
                        'timestamp': current_time,
                        'features': features,
                        'packet_count': len(window_packets)
                    })
        
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
        
        finally:
            # Track extraction time
            extraction_time = time.time() - start_time
            self.stats['extraction_times'].append(extraction_time)
            self.stats['buffer_sizes'].append(len(self.packet_buffer))
        
        return features
    
    def get_latest_features(self) -> Optional[Dict]:
        """
        Get the most recently extracted features.
        
        Returns:
            Dictionary with latest features and metadata or None
        """
        if not self.feature_cache:
            return None
        
        self.stats['cache_hits'] += 1
        return self.feature_cache[-1]
    
    def get_features_in_range(self, start_time: float, end_time: float) -> List[Dict]:
        """
        Get cached features within a time range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of feature dictionaries in time range
        """
        features_in_range = [
            f for f in self.feature_cache
            if start_time <= f['timestamp'] <= end_time
        ]
        
        return features_in_range
    
    def _extract_window_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Extract features from a window of packets.
        
        Args:
            df: DataFrame containing packet data
            
        Returns:
            DataFrame with extracted features
        """
        try:
            if df.empty:
                return None
            
            # Initialize feature dictionary
            features = {}
            
            # Basic packet statistics
            features.update(self._extract_basic_stats(df))
            
            # Timing features
            features.update(self._extract_timing_features(df))
            
            # Size features
            features.update(self._extract_size_features(df))
            
            # Protocol features
            features.update(self._extract_protocol_features(df))
            
            # Direction features
            features.update(self._extract_direction_features(df))
            
            # Application-specific features
            features.update(self._extract_application_features(df))
            
            # Flow features
            features.update(self._extract_flow_features(df))
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            return feature_df
        
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            return None
    
    def _extract_basic_stats(self, df: pd.DataFrame) -> Dict:
        """Extract basic packet statistics."""
        features = {}
        
        try:
            features['packet_count'] = len(df)
            features['total_bytes'] = df.get('packet_size', 0).sum()
            features['avg_packet_size'] = df.get('packet_size', 0).mean()
            features['std_packet_size'] = df.get('packet_size', 0).std()
            
            # Duration
            if 'timestamp' in df.columns and len(df) > 1:
                timestamps = df['timestamp'].sort_values()
                features['duration'] = timestamps.iloc[-1] - timestamps.iloc[0]
                features['packet_rate'] = len(df) / max(features['duration'], 0.001)
            else:
                features['duration'] = 0.0
                features['packet_rate'] = 0.0
        
        except Exception as e:
            logger.debug(f"Error extracting basic stats: {e}")
            # Set default values
            features.update({
                'packet_count': 0,
                'total_bytes': 0,
                'avg_packet_size': 0,
                'std_packet_size': 0,
                'duration': 0.0,
                'packet_rate': 0.0
            })
        
        return features
    
    def _extract_timing_features(self, df: pd.DataFrame) -> Dict:
        """Extract timing-based features."""
        features = {}
        
        try:
            if 'timestamp' in df.columns and len(df) > 1:
                timestamps = df['timestamp'].sort_values()
                inter_arrival_times = timestamps.diff().dropna()
                
                if len(inter_arrival_times) > 0:
                    features['mean_inter_arrival_time'] = inter_arrival_times.mean()
                    features['std_inter_arrival_time'] = inter_arrival_times.std()
                    features['min_inter_arrival_time'] = inter_arrival_times.min()
                    features['max_inter_arrival_time'] = inter_arrival_times.max()
                else:
                    features.update({
                        'mean_inter_arrival_time': 0.0,
                        'std_inter_arrival_time': 0.0,
                        'min_inter_arrival_time': 0.0,
                        'max_inter_arrival_time': 0.0
                    })
            else:
                features.update({
                    'mean_inter_arrival_time': 0.0,
                    'std_inter_arrival_time': 0.0,
                    'min_inter_arrival_time': 0.0,
                    'max_inter_arrival_time': 0.0
                })
        
        except Exception as e:
            logger.debug(f"Error extracting timing features: {e}")
            features.update({
                'mean_inter_arrival_time': 0.0,
                'std_inter_arrival_time': 0.0,
                'min_inter_arrival_time': 0.0,
                'max_inter_arrival_time': 0.0
            })
        
        return features
    
    def _extract_size_features(self, df: pd.DataFrame) -> Dict:
        """Extract packet size features."""
        features = {}
        
        try:
            sizes = df.get('packet_size', pd.Series([0]))
            
            if len(sizes) > 0:
                features['min_packet_size'] = sizes.min()
                features['max_packet_size'] = sizes.max()
                features['size_variance'] = sizes.var()
                
                # Size distribution
                small_packets = (sizes <= 64).sum()
                medium_packets = ((sizes > 64) & (sizes <= 1500)).sum()
                large_packets = (sizes > 1500).sum()
                
                total_packets = len(sizes)
                features['small_packet_ratio'] = small_packets / total_packets
                features['medium_packet_ratio'] = medium_packets / total_packets
                features['large_packet_ratio'] = large_packets / total_packets
            else:
                features.update({
                    'min_packet_size': 0,
                    'max_packet_size': 0,
                    'size_variance': 0.0,
                    'small_packet_ratio': 0.0,
                    'medium_packet_ratio': 0.0,
                    'large_packet_ratio': 0.0
                })
        
        except Exception as e:
            logger.debug(f"Error extracting size features: {e}")
            features.update({
                'min_packet_size': 0,
                'max_packet_size': 0,
                'size_variance': 0.0,
                'small_packet_ratio': 0.0,
                'medium_packet_ratio': 0.0,
                'large_packet_ratio': 0.0
            })
        
        return features
    
    def _extract_protocol_features(self, df: pd.DataFrame) -> Dict:
        """Extract protocol-related features."""
        features = {}
        
        try:
            if 'protocol' in df.columns:
                protocol_counts = df['protocol'].value_counts()
                total_packets = len(df)
                
                features['tcp_ratio'] = protocol_counts.get('TCP', 0) / total_packets
                features['udp_ratio'] = protocol_counts.get('UDP', 0) / total_packets
                features['other_protocol_ratio'] = protocol_counts.get('OTHER', 0) / total_packets
            else:
                features.update({
                    'tcp_ratio': 0.0,
                    'udp_ratio': 0.0,
                    'other_protocol_ratio': 0.0
                })
            
            # HTTPS ratio
            if 'is_https' in df.columns:
                features['https_ratio'] = df['is_https'].sum() / len(df)
            else:
                features['https_ratio'] = 0.0
        
        except Exception as e:
            logger.debug(f"Error extracting protocol features: {e}")
            features.update({
                'tcp_ratio': 0.0,
                'udp_ratio': 0.0,
                'other_protocol_ratio': 0.0,
                'https_ratio': 0.0
            })
        
        return features
    
    def _extract_direction_features(self, df: pd.DataFrame) -> Dict:
        """Extract traffic direction features."""
        features = {}
        
        try:
            if 'direction' in df.columns:
                direction_counts = df['direction'].value_counts()
                total_packets = len(df)
                
                up_packets = direction_counts.get('up', 0)
                down_packets = direction_counts.get('down', 0)
                
                features['upload_packet_ratio'] = up_packets / total_packets
                features['download_packet_ratio'] = down_packets / total_packets
                
                # Bytes by direction if available
                if 'packet_size' in df.columns:
                    up_bytes = df[df['direction'] == 'up']['packet_size'].sum()
                    down_bytes = df[df['direction'] == 'down']['packet_size'].sum()
                    total_bytes = df['packet_size'].sum()
                    
                    if total_bytes > 0:
                        features['upload_byte_ratio'] = up_bytes / total_bytes
                        features['download_byte_ratio'] = down_bytes / total_bytes
                        features['download_upload_ratio'] = down_bytes / max(up_bytes, 1)
                    else:
                        features.update({
                            'upload_byte_ratio': 0.0,
                            'download_byte_ratio': 0.0,
                            'download_upload_ratio': 0.0
                        })
                else:
                    features.update({
                        'upload_byte_ratio': 0.0,
                        'download_byte_ratio': 0.0,
                        'download_upload_ratio': 0.0
                    })
            else:
                features.update({
                    'upload_packet_ratio': 0.0,
                    'download_packet_ratio': 0.0,
                    'upload_byte_ratio': 0.0,
                    'download_byte_ratio': 0.0,
                    'download_upload_ratio': 0.0
                })
        
        except Exception as e:
            logger.debug(f"Error extracting direction features: {e}")
            features.update({
                'upload_packet_ratio': 0.0,
                'download_packet_ratio': 0.0,
                'upload_byte_ratio': 0.0,
                'download_byte_ratio': 0.0,
                'download_upload_ratio': 0.0
            })
        
        return features
    
    def _extract_application_features(self, df: pd.DataFrame) -> Dict:
        """Extract application-specific features."""
        features = {}
        
        try:
            if 'application' in df.columns:
                app_counts = df['application'].value_counts()
                total_packets = len(df)
                
                features['http_ratio'] = app_counts.get('HTTP', 0) / total_packets
                features['https_ratio'] = app_counts.get('HTTPS', 0) / total_packets
            else:
                features.update({
                    'http_ratio': 0.0,
                    'https_ratio': 0.0
                })
            
            # Port analysis
            if 'dst_port' in df.columns:
                common_ports = [80, 443, 53, 22, 21, 25, 110, 143, 993, 995]
                port_counts = df['dst_port'].value_counts()
                
                features['common_port_ratio'] = sum(
                    port_counts.get(port, 0) for port in common_ports
                ) / len(df)
                
                # Video streaming ports (heuristic)
                video_ports = [443, 80, 1935, 8080]  # HTTPS, HTTP, RTMP, Alt HTTP
                features['video_port_ratio'] = sum(
                    port_counts.get(port, 0) for port in video_ports
                ) / len(df)
            else:
                features.update({
                    'common_port_ratio': 0.0,
                    'video_port_ratio': 0.0
                })
        
        except Exception as e:
            logger.debug(f"Error extracting application features: {e}")
            features.update({
                'http_ratio': 0.0,
                'https_ratio': 0.0,
                'common_port_ratio': 0.0,
                'video_port_ratio': 0.0
            })
        
        return features
    
    def _extract_flow_features(self, df: pd.DataFrame) -> Dict:
        """Extract flow-based features."""
        features = {}
        
        try:
            # Burst detection
            if 'timestamp' in df.columns and 'packet_size' in df.columns:
                bursts = self._detect_bursts(df)
                features['burst_count'] = len(bursts)
                features['avg_burst_size'] = np.mean([b['size'] for b in bursts]) if bursts else 0
                features['avg_burst_duration'] = np.mean([b['duration'] for b in bursts]) if bursts else 0
            else:
                features.update({
                    'burst_count': 0,
                    'avg_burst_size': 0.0,
                    'avg_burst_duration': 0.0
                })
            
            # Connection patterns
            if 'src_ip' in df.columns and 'dst_ip' in df.columns:
                unique_connections = df[['src_ip', 'dst_ip']].drop_duplicates()
                features['unique_connections'] = len(unique_connections)
                features['connection_diversity'] = len(unique_connections) / len(df)
            else:
                features.update({
                    'unique_connections': 0,
                    'connection_diversity': 0.0
                })
        
        except Exception as e:
            logger.debug(f"Error extracting flow features: {e}")
            features.update({
                'burst_count': 0,
                'avg_burst_size': 0.0,
                'avg_burst_duration': 0.0,
                'unique_connections': 0,
                'connection_diversity': 0.0
            })
        
        return features
    
    def _detect_bursts(self, df: pd.DataFrame, min_size: int = 1000, max_gap: float = 0.1) -> List[Dict]:
        """
        Detect burst patterns in packet data.
        
        Args:
            df: DataFrame with packet data
            min_size: Minimum bytes for a burst
            max_gap: Maximum time gap within a burst (seconds)
            
        Returns:
            List of burst dictionaries
        """
        bursts = []
        
        try:
            if len(df) < 2:
                return bursts
            
            # Sort by timestamp
            sorted_df = df.sort_values('timestamp')
            
            current_burst = []
            last_timestamp = None
            
            for _, packet in sorted_df.iterrows():
                timestamp = packet.get('timestamp', 0)
                size = packet.get('packet_size', 0)
                
                if last_timestamp is None or (timestamp - last_timestamp) <= max_gap:
                    current_burst.append(packet)
                else:
                    # End current burst, start new one
                    if current_burst:
                        burst_info = self._analyze_burst(current_burst)
                        if burst_info['size'] >= min_size:
                            bursts.append(burst_info)
                    
                    current_burst = [packet]
                
                last_timestamp = timestamp
            
            # Process final burst
            if current_burst:
                burst_info = self._analyze_burst(current_burst)
                if burst_info['size'] >= min_size:
                    bursts.append(burst_info)
        
        except Exception as e:
            logger.debug(f"Error detecting bursts: {e}")
        
        return bursts
    
    def _analyze_burst(self, burst_packets: List) -> Dict:
        """Analyze a burst of packets."""
        if not burst_packets:
            return {'size': 0, 'duration': 0, 'packet_count': 0}
        
        try:
            sizes = [p.get('packet_size', 0) for p in burst_packets]
            timestamps = [p.get('timestamp', 0) for p in burst_packets]
            
            return {
                'size': sum(sizes),
                'duration': max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0,
                'packet_count': len(burst_packets),
                'avg_packet_size': np.mean(sizes)
            }
        except Exception:
            return {'size': 0, 'duration': 0, 'packet_count': 0}
    
    def _clean_old_packets(self):
        """Remove packets older than the window size."""
        if not self.packet_buffer:
            return
        
        current_time = time.time()
        cutoff_time = current_time - (self.window_size * 2)  # Keep extra buffer
        
        # Remove old packets from the left
        while (self.packet_buffer and 
               self.packet_buffer[0].get('timestamp', 0) < cutoff_time):
            self.packet_buffer.popleft()
    
    def get_statistics(self) -> Dict:
        """Get performance statistics."""
        stats = self.stats.copy()
        
        if self.stats['extraction_times']:
            stats['avg_extraction_time'] = np.mean(list(self.stats['extraction_times']))
            stats['max_extraction_time'] = np.max(list(self.stats['extraction_times']))
        
        if self.stats['buffer_sizes']:
            stats['avg_buffer_size'] = np.mean(list(self.stats['buffer_sizes']))
            stats['max_buffer_size'] = np.max(list(self.stats['buffer_sizes']))
        
        stats['buffer_current_size'] = len(self.packet_buffer)
        stats['cache_current_size'] = len(self.feature_cache)
        
        return stats
    
    def reset_statistics(self):
        """Reset performance statistics."""
        self.stats = {
            'packets_processed': 0,
            'features_extracted': 0,
            'cache_hits': 0,
            'extraction_times': deque(maxlen=100),
            'buffer_sizes': deque(maxlen=100)
        }
    
    def clear_buffers(self):
        """Clear all buffers and caches."""
        with self.buffer_lock:
            self.packet_buffer.clear()
            self.feature_cache.clear()
        
        logger.info("Cleared all buffers and caches")


if __name__ == "__main__":
    """Test the real-time feature extractor."""
    import json
    import random
    
    # Setup logging
    setup_logger(level=logging.INFO)
    
    # Create extractor
    extractor = RealTimeFeatureExtractor(window_size=2.0, update_interval=0.5)
    
    # Generate test packets
    logger.info("Generating test packets...")
    
    base_time = time.time()
    for i in range(100):
        packet = {
            'timestamp': base_time + i * 0.1,
            'packet_size': random.randint(64, 1500),
            'protocol': random.choice(['TCP', 'UDP']),
            'direction': random.choice(['up', 'down']),
            'src_port': random.randint(1024, 65535),
            'dst_port': random.choice([80, 443, 53, 22]),
            'is_https': random.choice([True, False])
        }
        
        extractor.add_packet(packet)
        
        # Extract features periodically
        if i % 20 == 0:
            features = extractor.extract_features_now()
            if features is not None:
                logger.info(f"Extracted features at packet {i}: {features.shape}")
    
    # Final extraction
    final_features = extractor.extract_features_now(force=True)
    if final_features is not None:
        logger.info(f"Final features: {final_features.shape}")
        logger.info(f"Feature columns: {list(final_features.columns)}")
    
    # Show statistics
    stats = extractor.get_statistics()
    logger.info(f"Performance statistics: {json.dumps(stats, indent=2)}")
