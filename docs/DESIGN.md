# System Design Document

## Architecture Overview

The Real-time Reel/Video Traffic Detection System follows a multi-stage pipeline architecture designed for scalability, real-time performance, and accuracy. The system processes network traffic through four main stages: capture, feature extraction, inference, and visualization.

## Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Capture  │───▶│ Feature Extract │───▶│   Live Analysis │
│   (psutil)      │    │ (prepare_dataset)│    │(live_dashboard) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Network Stats  │    │  Feature Vector │    │  Web Dashboard  │
│   (System API)  │    │ (19 features)   │    │   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Component Architecture

### 1. Data Capture Layer

#### Network Monitoring (psutil-based)
```python
# Architecture: System-level monitoring
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Network Stats  │───▶│   Processing    │───▶│  Classification │
│    (psutil)     │    │   (Rolling)     │    │   (Heuristic)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Key Components:**
- **System API**: psutil for network interface monitoring
- **No Admin Rights**: Works without special privileges
- **Real-time**: 3-second analysis intervals
- **Traffic Pattern Analysis**: Based on bandwidth and packet rates

#### Platform Support
```
Windows: psutil + native APIs
Linux:   psutil + /proc/net monitoring  
Cross-platform: Standard Python libraries
```

### 2. Feature Engineering Layer

#### Rolling Window Architecture
```python
# Time-based sliding window: 2-3 second intervals
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Packet T1     │    │   Packet T2     │    │   Packet T3     │
│   ┌─────────┐   │    │   ┌─────────┐   │    │   ┌─────────┐   │
│   │Window 1 │   │───▶│   │Window 2 │   │───▶│   │Window 3 │   │
│   └─────────┘   │    │   └─────────┘   │    │   └─────────┘   │
│   [t-3, t-1]    │    │   [t-2, t]      │    │   [t-1, t+1]    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2. Feature Engineering Layer

#### Real Dataset Features (19 total)

1. **Basic Traffic Features**
   ```python
   packet_count            # Total packets in window
   total_bytes            # Total bytes (currently 0 in dataset)
   avg_packet_size        # Average packet size
   std_packet_size        # Packet size variability
   min_packet_size        # Minimum packet size
   max_packet_size        # Maximum packet size
   ```

2. **Temporal Features**
   ```python
   window_duration        # Duration of capture window
   avg_inter_arrival      # Average time between packets
   std_inter_arrival      # Inter-arrival time variance
   packets_per_second     # Packet frequency rate
   bytes_per_second      # Byte rate (currently 0)
   ```

3. **Protocol & Pattern Features**
   ```python
   tcp_ratio             # TCP vs other protocols
   udp_ratio            # UDP protocol ratio
   unique_src_ports     # Number of unique source ports
   unique_dst_ports     # Number of unique destination ports
   burst_ratio          # Traffic burst patterns
   ```

4. **Metadata**
   ```python
   label               # 0=Non-Video, 1=Video
   file_source        # Source file identifier
   window_id          # Window sequence number
   ```

### 3. Model Architecture

#### Single Model Approach

```
Input Features (19)
     │
     ▼
┌─────────────────┐
│   XGBoost       │
│   Classifier    │
│ (or heuristic)  │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Binary Output   │
│ 0=Non-Video     │
│ 1=Video         │
└─────────────────┘
```

#### Model Specifications

**XGBoost Model** (when available):
```python
XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
```

**Heuristic Fallback**:
```python
# Based on bandwidth and packet patterns
if bandwidth > 500KB/s and packets > 800:
    return "VIDEO" (95% confidence)
elif bandwidth > 50KB/s:
    return "SOCIAL_MEDIA" (85% confidence)  
else:
    return "REGULAR_TRAFFIC" (70% confidence)
```

### 4. Real-time Processing

#### Simple Pipeline Architecture
```python
# System monitoring with live classification
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Network Monitor │    │   Classifier    │    │  Web Dashboard  │
│ (live_dashboard)│───▶│  (heuristic)    │───▶│    (FastAPI)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  psutil stats   │    │ Traffic pattern │    │ WebSocket data  │
│  (3s intervals) │    │  classification │    │ (live updates)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

#### Processing Approach

1. **System API**: Using psutil for efficiency
2. **Heuristic Classification**: Fast pattern matching
3. **Simple Processing**: Minimal computational overhead
4. **Background Operation**: Non-intrusive monitoring

### 5. Dashboard and Visualization

#### Web Dashboard Architecture
```
Browser Client (HTML/CSS/JS)
    │
    ▼ WebSocket
┌─────────────────┐
│   FastAPI App   │
│(live_dashboard) │
└─────────────────┘
    │
    ▼ Direct Integration
┌─────────────────┐
│ Network Monitor │
│   (psutil)      │
└─────────────────┘
```

#### Key Metrics Displayed
- **Real-time Classification**: Video vs Non-video traffic
- **Confidence Scores**: Heuristic certainty levels  
- **Traffic Statistics**: Bandwidth, packet counts, rates
- **Network Activity**: Live system monitoring
- **Simple Visualization**: Clean, professional interface

## Data Flow

### Training Phase
```
Raw PCAP Files
     │
     ▼ Parse & Extract
Feature CSV Files
     │
     ▼ Train Models
ONNX Model Files
     │
     ▼ Evaluate
Performance Metrics
```

### Inference Phase
```
Live Network Traffic
     │
     ▼ Capture (2-3s windows)
Packet Metadata
     │
     ▼ Feature Engineering
Feature Vectors
     │
     ▼ Model Inference
Predictions + Confidence
     │
     ▼ WebSocket
Dashboard Updates
```

## Scalability Considerations

### Horizontal Scaling
- **Load Balancing**: Multiple capture nodes
- **Distributed Processing**: Feature computation across nodes
- **Model Serving**: ONNX Runtime with multiple instances

### Vertical Scaling
- **Memory Management**: Efficient buffer allocation
- **CPU Optimization**: Multi-threading for I/O bound operations
- **GPU Acceleration**: Optional CUDA support for CNN models

## Advanced Pipeline Architecture (Future Implementation)

### Mobile & Cross-Platform Traffic Detection

The system architecture supports extension to mobile and multi-device traffic detection through advanced packet capture mechanisms. While the current implementation uses system-level monitoring, the codebase includes infrastructure for comprehensive network analysis.

#### Advanced Data Capture Pipeline
```python
# Complex pipeline for mobile and multi-device detection
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Packet Capture │───▶│ Feature Extract │───▶│   Inference     │───▶│   Dashboard     │
│  (real_capture) │    │  (features.py)  │    │   (infer.py)    │    │ (dashboard.py)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Raw Packets    │    │  Feature Vector │    │  ML Predictions │    │  Live Metrics   │
│  (Scapy/WinPcap)│    │ (37+ features)  │    │ (XGBoost/CNN)   │    │  (WebSocket)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

#### Mobile Traffic Detection Capabilities

**Platform Support:**
```
Windows:  WinPcap/Npcap + Scapy for deep packet inspection
Linux:    libpcap + PyShark for network interface monitoring
Android:  WiFi hotspot tethering + packet capture via PC
iOS:      Network proxy setup + traffic tunneling
Router:   PCAP capture at network gateway level
```

**Advanced Features (Available in codebase):**
- **Deep Packet Inspection**: Using `real_capture.py` with Scapy
- **Mobile Device Detection**: Through MAC address and traffic patterns
- **Cross-Device Correlation**: Matching traffic patterns across devices
- **Advanced Feature Engineering**: Direction-specific metrics, TLS analysis
- **Complex ML Models**: CNN for sequence analysis, ensemble methods

#### Implementation Files Ready for Mobile Detection

```
src/
├── data_collection/
│   ├── capture_pc.py        # Advanced packet capture with Scapy
│   └── real_capture.py      # Production packet capture system
├── inference/
│   ├── features.py          # Advanced feature engineering
│   ├── infer.py            # ML model inference pipeline
│   └── dashboard.py        # Advanced visualization
└── models/
    ├── xgboost_model.py    # Tree-based classification
    ├── cnn_model.py        # Sequence pattern analysis
    └── ensemble.py         # Multi-model combination
```

**Mobile Detection Workflow:**
1. **Device Tethering**: Mobile connects via WiFi hotspot to monitoring PC
2. **Packet Capture**: `real_capture.py` captures all mobile traffic using Scapy
3. **Traffic Labeling**: Automatic detection of mobile app signatures
4. **Advanced Features**: Extract mobile-specific patterns (app fingerprints, burst patterns)
5. **ML Classification**: Use ensemble models for high-accuracy detection
6. **Real-time Monitoring**: Live dashboard shows per-device reel vs non-reel usage

**Current Status:**
- ✅ **Infrastructure Ready**: All necessary files and modules implemented
- ✅ **Packet Capture**: Advanced capture system with `real_capture.py`
- ✅ **ML Pipeline**: Complete training and inference pipeline
- ⏳ **Not Implemented**: Mobile device integration (can be activated)
- ⏳ **Future Work**: Cross-platform deployment and mobile app development

### Why Current Implementation Uses Simple Approach

For the Samsung EnnovateX 2025 demonstration, we implemented the simpler psutil-based approach because:
1. **No Admin Rights Required**: Works out-of-the-box on any system
2. **Cross-Platform Compatibility**: Runs on Windows/Linux/Mac without drivers
3. **Demonstration Focus**: Shows core ML capabilities without setup complexity
4. **Rapid Deployment**: Can be run immediately for hackathon presentation

The advanced pipeline with mobile detection can be activated by using the existing `real_capture.py` and advanced inference modules, but requires administrator privileges and network configuration setup.

## Performance Targets

### Realistic Latency
- **Network Monitoring**: 3-second intervals
- **Classification**: Real-time heuristic analysis
- **Dashboard Update**: 1-second WebSocket updates
- **Browser Response**: <500ms for UI updates

### Actual Throughput
- **System Monitoring**: Continuous background process
- **Analysis Windows**: ~20 classifications per minute
- **Concurrent Users**: 1-5 dashboard viewers
- **Data Processing**: 440 sample dataset for training

## Security and Privacy

### Data Protection
- **No Payload Access**: Headers and metadata only
- **Encryption**: TLS for dashboard connections
- **Access Control**: API key authentication
- **Audit Logging**: All access attempts logged

### Privacy Preservation
- **On-device Processing**: No cloud dependencies
- **Data Minimization**: Only necessary features extracted
- **Temporal Limits**: Rolling window approach prevents long-term storage

## Error Handling and Resilience

### Fault Tolerance
- **Graceful Degradation**: System continues with reduced accuracy if models fail
- **Circuit Breaker**: Prevent cascade failures
- **Health Checks**: Continuous component monitoring
- **Automatic Recovery**: Self-healing capabilities

### Error Recovery
```python
try:
    prediction = model.predict(features)
except Exception as e:
    logging.error(f"Model inference failed: {e}")
    prediction = fallback_classifier(features)  # Simple rule-based backup
```

## Technology Stack

### Current Implementation (Simple Pipeline)
- **System Monitoring**: psutil for network statistics
- **Web Framework**: FastAPI, WebSockets for dashboard
- **Data Processing**: NumPy, Pandas for feature analysis
- **Classification**: Heuristic rules + optional XGBoost
- **Frontend**: HTML/CSS/JavaScript for visualization

### Advanced Pipeline (Mobile Detection Ready)
- **Packet Capture**: Scapy, PyShark for deep inspection
- **Machine Learning**: XGBoost, PyTorch, ensemble methods
- **Advanced Processing**: Complex feature engineering pipeline
- **Model Serving**: ONNX Runtime for optimized inference
- **Mobile Support**: WinPcap/Npcap for cross-platform capture

### Development Tools
- **Version Control**: Git
- **Testing**: pytest, unittest
- **Documentation**: Markdown
- **Packaging**: setuptools, pip

## Deployment Options

### Standalone Deployment
```bash
# Single machine deployment
python src/inference/dashboard.py &
python src/inference/capture.py | python src/inference/infer.py
```

### Docker Deployment
```yaml
# docker-compose.yml
version: '3.8'
services:
  capture:
    build: .
    command: python src/inference/capture.py
    network_mode: host
  inference:
    build: .
    command: python src/inference/infer.py
    depends_on: [capture]
  dashboard:
    build: .
    command: python src/inference/dashboard.py
    ports: ["8080:8080"]
```
