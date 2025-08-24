# Dataset Documentation

## Overview

This document describes the data collection methodology, dataset structure, and privacy compliance measures for the Real-time Reel/Video Traffic Detection System.

## Data Collection Methodology

### Controlled Experiment Setup

All data was collected through controlled experiments using the researcher's own devices to ensure privacy compliance and data quality.

#### Collection Environment
- **Device Types**: Personal Android smartphones, Windows/Linux PCs
- **Network Conditions**: WiFi and mobile data connections
- **Applications Tested**: Instagram, YouTube, Twitter , Whatsapp , Wiki
- **Duration**: 5+ mins of traffic across different scenarios , more data can be collected by either increasing the scrolling duration during packet capture 

#### Traffic Scenarios

1. **Reel/Video Traffic**:
   - Instagram Reels browsing and watching
   - TikTok video streaming
   - YouTube Shorts consumption
   - Facebook video posts
   - Story viewing on various platforms

2. **Non-Reel/Video Traffic**:
   - Text-based social media browsing
   - Image loading (static posts)
   - Chat/messaging activities
   - Profile browsing
   - Search and discovery (text-based)
   - News feed scrolling (non-video content)

### Data Collection Tools

#### PC-Based Collection
```bash
# Primary tool: tcpdump with specific filters
sudo tcpdump -i eth0 -w capture.pcap 'host instagram.com or host tiktok.com'

# Alternative: Wireshark programmatic capture
tshark -i eth0 -w capture.pcap -f "tcp port 443"
```

#### Android Collection
- **Method 1**: VPN Service approach (requires custom Android app)
- **Method 2**: USB tethering + PC-based capture
- **Method 3**: WiFi hotspot monitoring

### Privacy and Compliance

#### Data Sanitization
- **No Payload Inspection**: Only packet headers and metadata analyzed
- **IP Address Anonymization**: All IP addresses replaced with generic identifiers
- **Timestamp Normalization**: Relative timestamps used instead of absolute
- **MAC Address Removal**: Hardware identifiers stripped from data

#### Ethical Considerations
- All data collected from researcher's personal devices only
- No third-party traffic intercepted
- Clear consent and purpose limitation
- Data minimization principles followed
- Regular data purging (30-day retention)

## Dataset Structure

### Raw Data Format (.json)
```
data/raw/
├── reels_01.json                # Instagram Reels traffic patterns
├── youtube_shorts_01.json       # YouTube Shorts viewing sessions
├── twitter_scrolling_01.json    # Twitter/X social media browsing
└── web_browsing_01.json         # Regular web browsing traffic
```

### Processed Feature Data (.csv)

#### Primary Features Dataset
```
data/processed/features.csv
Columns:
- session_id: Unique session identifier
- timestamp: Relative timestamp (seconds from start)
- total_bytes_up: Total uplink bytes
- total_bytes_down: Total downlink bytes
- packet_count_up: Uplink packet count
- packet_count_down: Downlink packet count
- avg_packet_size_up: Average uplink packet size
- avg_packet_size_down: Average downlink packet size
- std_packet_size_up: Std dev of uplink packet sizes
- std_packet_size_down: Std dev of downlink packet sizes
- mean_iat_up: Mean inter-arrival time uplink (ms)
- mean_iat_down: Mean inter-arrival time downlink (ms)
- var_iat_up: Variance of inter-arrival times uplink
- var_iat_down: Variance of inter-arrival times downlink
- burst_rate_up: Uplink burst rate (packets/second)
- burst_rate_down: Downlink burst rate (packets/second)
- bytes_ratio: Uplink:downlink byte ratio
- throughput_slope_up: Uplink throughput slope
- throughput_slope_down: Downlink throughput slope
- tls_record_periodicity: TLS record timing patterns
- application: Source application (anonymized)
- traffic_type: Label (reel_video=1, non_reel=0)
```

#### Sequence Data for CNN
```
data/processed/sequences.csv
Columns:
- session_id: Unique session identifier
- packet_sizes: Sequence of packet sizes (up to 100 packets)
- packet_times: Sequence of inter-arrival times
- direction: Packet direction sequence (1=up, 0=down)
- traffic_type: Label (reel_video=1, non_reel=0)
```

### Dataset Statistics

#### Dataset Size
- **Total Feature Windows**: 440
- **Video Traffic (Label 1)**: 250 (56.8%)
- **Non-Video Traffic (Label 0)**: 190 (43.2%)
- **Total Features**: 19 per window
- **Data Sources**: 4 capture files

#### Data Sources Distribution
```
reels_01              153 windows (34.8%)
youtube_shorts_01      97 windows (22.0%)
twitter_scrolling_01   95 windows (21.6%)
web_browsing_01        95 windows (21.6%)
```

#### Feature Statistics
```
Feature                    Mean      Min       Max
packet_count              850       3         5,773
avg_inter_arrival         0.018s    0.000s    0.357s
tcp_ratio                 1.000     1.000     1.000
packets_per_second        354       3         23,432
burst_ratio               0.959     0.429     1.000
```

**Note**: The dataset captures packet-level metadata only. Byte-level analysis shows all entries have zero bytes, indicating the focus is on packet timing and frequency patterns rather than payload size analysis.

#### Class Distribution by Application
```
Data Source           Video Windows    Non-Video Windows    Total
reels_01             153              0                   153
youtube_shorts_01    97               0                   97
twitter_scrolling_01 0                95                  95
web_browsing_01      0                95                  95
Total                250              190                 440
```

## Data Quality Assurance

### Validation Checks
- **Temporal Consistency**: Packet timestamps in chronological order
- **Packet Count Validation**: Packet counts within expected ranges (3-5,773 per window)
- **Session Integrity**: Complete capture sessions without gaps
- **Label Accuracy**: Manual verification of traffic type labels

### Preprocessing Steps
1. **Outlier Removal**: Statistical outliers beyond 3 standard deviations
2. **Window Filtering**: Variable window sizes based on traffic patterns
3. **Feature Normalization**: Z-score normalization for continuous features
4. **Missing Value Handling**: Forward-fill for temporal features

## Anonymization Process

### Identifier Replacement
```python
# Example anonymization mapping
Original IP          Anonymous ID
157.240.7.35    →   SERVER_001
31.13.82.1      →   SERVER_002
192.168.1.100   →   CLIENT_001
```

### Temporal Anonymization
- Absolute timestamps → Relative offsets
- Date information → Day-of-week only
- Time zones → UTC normalized

## Usage Guidelines

### Academic Research
- Citation required for any publications
- Data available for reproducibility studies
- Contact information for questions

### Commercial Use
- Apache-2.0 license compliance required
- Attribution to original research
- No warranty or liability assumed

## Data Collection Scripts

### Quick Start Collection
```bash
# Collect 5 minutes of Instagram traffic
python src/data_collection/capture_pc.py \
    --app instagram \
    --duration 300 \
    --output data/raw/instagram_sample.pcap

# Parse and extract features
python src/data_collection/pcap_parser.py \
    --input data/raw/instagram_sample.pcap \
    --output data/processed/instagram_features.csv
```

### Batch Collection
```bash
# Run comprehensive data collection
./scripts/collect_all_scenarios.sh
```

## Data Updates

- **Version**: 1.0.0
- **Last Updated**: August 14, 2025
- **Update Frequency**: Quarterly
- **Validation**: Continuous monitoring for data drift

## Contact Information

For questions about the dataset or to report issues:
- **Email**: subhradipdray@gmail.com
- **GitHub Issues**: [Project Repository Issues](https://github.com/yourusername/samsung-traffic-detection/issues)
- **Documentation**: [Full Documentation](docs/)
