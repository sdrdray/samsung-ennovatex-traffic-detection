#!/usr/bin/env python3
"""
Dataset Preparation Script

Combines multiple JSON packet capture files, extracts machine learning features,
and creates a labeled dataset for training the traffic classification model.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURE THIS SECTION ---
# IMPORTANT: List the JSON file names you created for each category.
REEL_VIDEO_FILES = [
    'reels_01.json',
    'youtube_shorts_01.json'
    # Add any other video files you made
]

NON_REEL_FILES = [
    'twitter_scrolling_01.json',
    'web_browsing_01.json'
    # Add any other non-video files you made
]

# Data paths
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DATA_DIR / "final_labeled_features.csv"

# Feature extraction parameters
WINDOW_SIZE_SECONDS = 3  # Group packets into 3-second windows
# --- END CONFIGURATION ---


def load_packet_data(file_path: Path) -> List[Dict]:
    """Load packet data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} packets from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []


def group_packets_by_time_window(packets: List[Dict], window_size: int = 3) -> List[List[Dict]]:
    """Group packets into time windows."""
    if not packets:
        return []
    
    # Sort packets by timestamp
    sorted_packets = sorted(packets, key=lambda x: x.get('timestamp', 0))
    
    windows = []
    current_window = []
    window_start_time = sorted_packets[0].get('timestamp', 0)
    
    for packet in sorted_packets:
        packet_time = packet.get('timestamp', 0)
        
        # If packet is within current window
        if packet_time - window_start_time <= window_size:
            current_window.append(packet)
        else:
            # Start new window
            if current_window:
                windows.append(current_window)
            current_window = [packet]
            window_start_time = packet_time
    
    # Add the last window
    if current_window:
        windows.append(current_window)
    
    return windows


def extract_features_from_window(window: List[Dict]) -> Dict[str, float]:
    """Extract statistical features from a packet window."""
    if not window:
        return {}
    
    # Packet sizes
    packet_sizes = [p.get('packet_size', 0) for p in window]
    
    # Inter-arrival times
    timestamps = [p.get('timestamp', 0) for p in window]
    inter_arrival_times = []
    for i in range(1, len(timestamps)):
        inter_arrival_times.append(timestamps[i] - timestamps[i-1])
    
    # Protocol distribution
    protocols = [p.get('protocol', 'unknown') for p in window]
    tcp_count = protocols.count('TCP')
    udp_count = protocols.count('UDP')
    
    # Port analysis
    src_ports = [p.get('src_port', 0) for p in window if p.get('src_port')]
    dst_ports = [p.get('dst_port', 0) for p in window if p.get('dst_port')]
    
    # Calculate features
    features = {
        # Basic statistics
        'packet_count': len(window),
        'total_bytes': sum(packet_sizes),
        'avg_packet_size': np.mean(packet_sizes) if packet_sizes else 0,
        'std_packet_size': np.std(packet_sizes) if len(packet_sizes) > 1 else 0,
        'min_packet_size': min(packet_sizes) if packet_sizes else 0,
        'max_packet_size': max(packet_sizes) if packet_sizes else 0,
        
        # Timing features
        'window_duration': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
        'avg_inter_arrival': np.mean(inter_arrival_times) if inter_arrival_times else 0,
        'std_inter_arrival': np.std(inter_arrival_times) if len(inter_arrival_times) > 1 else 0,
        
        # Protocol features
        'tcp_ratio': tcp_count / len(window) if window else 0,
        'udp_ratio': udp_count / len(window) if window else 0,
        
        # Port features
        'unique_src_ports': len(set(src_ports)) if src_ports else 0,
        'unique_dst_ports': len(set(dst_ports)) if dst_ports else 0,
        
        # Traffic patterns
        'bytes_per_second': (sum(packet_sizes) / (timestamps[-1] - timestamps[0])) if len(timestamps) > 1 and timestamps[-1] != timestamps[0] else 0,
        'packets_per_second': (len(window) / (timestamps[-1] - timestamps[0])) if len(timestamps) > 1 and timestamps[-1] != timestamps[0] else 0,
        
        # Burst detection
        'burst_ratio': len([iat for iat in inter_arrival_times if iat < 0.1]) / len(inter_arrival_times) if inter_arrival_times else 0,
    }
    
    return features


def process_file(file_path: Path, label: int) -> pd.DataFrame:
    """Process a single JSON file and extract features."""
    logger.info(f"Processing {file_path} with label {label}")
    
    # Load packet data
    packets = load_packet_data(file_path)
    if not packets:
        return pd.DataFrame()
    
    # Group into time windows
    windows = group_packets_by_time_window(packets, WINDOW_SIZE_SECONDS)
    logger.info(f"Created {len(windows)} time windows from {len(packets)} packets")
    
    # Extract features from each window
    feature_list = []
    for i, window in enumerate(windows):
        features = extract_features_from_window(window)
        if features:  # Only add if features were extracted
            features['label'] = label
            features['file_source'] = file_path.stem
            features['window_id'] = i
            feature_list.append(features)
    
    df = pd.DataFrame(feature_list)
    logger.info(f"Extracted features for {len(df)} windows")
    return df


def main():
    """Main processing function."""
    logger.info("Starting dataset preparation...")
    
    # Create output directory
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    all_dataframes = []
    
    # Process video/reel files (label = 1)
    logger.info("Processing reel/video files...")
    for filename in REEL_VIDEO_FILES:
        file_path = RAW_DATA_DIR / filename
        if file_path.exists():
            df = process_file(file_path, label=1)  # 1 = Reel/Video traffic
            if not df.empty:
                all_dataframes.append(df)
        else:
            logger.warning(f"File not found: {file_path}")
    
    # Process non-reel files (label = 0)
    logger.info("Processing non-reel files...")
    for filename in NON_REEL_FILES:
        file_path = RAW_DATA_DIR / filename
        if file_path.exists():
            df = process_file(file_path, label=0)  # 0 = Regular traffic
            if not df.empty:
                all_dataframes.append(df)
        else:
            logger.warning(f"File not found: {file_path}")
    
    if not all_dataframes:
        logger.error("No data processed! Check your file paths and data.")
        return
    
    # Combine all dataframes
    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Basic data validation
    logger.info(f"Final dataset shape: {final_df.shape}")
    logger.info(f"Label distribution:\n{final_df['label'].value_counts()}")
    
    # Remove any rows with NaN values
    initial_rows = len(final_df)
    final_df = final_df.dropna()
    if len(final_df) < initial_rows:
        logger.info(f"Removed {initial_rows - len(final_df)} rows with NaN values")
    
    # Save the final dataset
    final_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved final dataset to {OUTPUT_FILE}")
    logger.info(f"Dataset contains {len(final_df)} samples with {len(final_df.columns)-3} features")  # -3 for label, file_source, window_id
    
    # Display feature summary
    feature_columns = [col for col in final_df.columns if col not in ['label', 'file_source', 'window_id']]
    logger.info(f"Features: {', '.join(feature_columns)}")


if __name__ == "__main__":
    main()
