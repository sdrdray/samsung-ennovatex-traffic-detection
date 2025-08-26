# Samsung EnnovateX 2025 AI Challenge Submission

**Problem Statement** - Real-time Detection of Reel Traffic vs Non-reel Traffic in a Social-networking Application

**Team name** - Sdrdray

**Team members (Names)** - Subhradip Debray

**Demo Video Link** - [[click here - demo video]](https://youtu.be/rnXVPW4V3fQ)

## Project Artefacts

**Technical Documentation** - [Docs](docs/) (All technical details are written in markdown files inside the docs folder in the repo)

**Source Code** - [Source](src/) (All source code is added to the src folder in the repo. The code is capable of being successfully installed/executed and runs consistently on the intended platforms.)

**Models Used** - XGBoost Classifier (scikit-learn implementation), No external Hugging Face models used

**Models Published** - [Will be uploaded to Hugging Face if required]

**Datasets Used** - Custom dataset created from real network traffic captures (Instagram Reels, Twitter scrolling, YouTube Shorts, Web browsing)

**Datasets Published** - [Will be published on Hugging Face under Creative Commons license if required]

## Features

- **Real Data-Based**: Uses actual network traffic captured from controlled experiments
- **Privacy-Preserving**: Analyzes only network metadata (no payload decryption)
- **Real-time Classification**: Live traffic analysis with 2-3 second rolling windows
- **XGBoost Model**: High-performance gradient boosting classifier
- **Interactive Dashboard**: FastAPI + WebSocket real-time monitoring
- **Cross-Platform**: Works on Linux, Windows, and macOS
- **Performance Testing**: Network condition simulation and robustness evaluation

## Project Structure

```
samsung/
├── LICENSE                    # Apache-2.0 license
├── README.md                 # This file
├── DATASET.md               # Data collection methodology
├── DESIGN.md                # System architecture
├── ETHICS.md                # Privacy and fairness considerations
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
├── data/                    # Data storage
│   ├── raw/                # Raw .pcap files
│   ├── processed/          # Extracted CSV features
│   └── examples/           # Sample sanitized datasets
├── src/                     # Main source code
│   ├── __init__.py
│   ├── data_collection/    # Traffic capture scripts
│   │   ├── __init__.py
│   │   ├── capture_android.py
│   │   ├── capture_pc.py
│   │   └── pcap_parser.py
│   ├── features/           # Feature engineering
│   │   ├── __init__.py
│   │   ├── extractor.py
│   │   └── preprocessor.py
│   ├── models/             # ML models
│   │   ├── __init__.py
│   │   ├── xgboost_model.py
│   │   ├── cnn_model.py
│   │   └── ensemble.py
│   ├── inference/          # Real-time inference
│   │   ├── __init__.py
│   │   ├── capture.py
│   │   ├── features.py
│   │   ├── infer.py
│   │   └── dashboard.py
│   └── utils/              # Utility functions
│       ├── __init__.py
│       ├── network.py
│       └── logging.py
├── models/                  # Trained models
│   ├── xgboost_model.onnx
│   ├── cnn_model.onnx
│   └── ensemble_model.onnx
├── tests/                   # Test scripts
│   ├── __init__.py
│   ├── test_features.py
│   ├── test_models.py
│   └── performance_test.py
├── scripts/                 # Utility scripts
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── simulate_conditions.py
├── docs/                    # Documentation
│   ├── installation.md
│   ├── usage.md
│   └── api_reference.md
```

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/samsung-traffic-detection.git
cd samsung-traffic-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

### Data Collection

#### On PC (Windows/Linux):
```bash
# Start traffic capture
python src/data_collection/capture_pc.py --interface eth0 --duration 300

# Parse captured packets
python src/data_collection/pcap_parser.py --input data/raw/capture.pcap --output data/processed/features.csv
```

#### On Android (via VPN Service):
```bash
# Use Android app or tethering method
python src/data_collection/capture_android.py --method tethering
```

### Model Training

```bash
# Train XGBoost model
python scripts/train_models.py --model xgboost --data data/processed/features.csv

# Train CNN model
python scripts/train_models.py --model cnn --data data/processed/sequences.csv

# Train ensemble model
python scripts/train_models.py --model ensemble
```

### Real-time Inference

The primary method for running the system is via the integrated live dashboard. This script handles both the real-time analysis and the web server.

```bash
# Run with administrator/sudo privileges
python live_dashboard.py
```

Then, visit http://localhost:8003 in your web browser.

## Performance

- **Accuracy**: 76.14% on labeled dataset
- **Real-time processing**: <2 second analysis windows
- **Memory Usage**: <200MB during operation

## Samsung EnnovateX 2025

This project was developed for the Samsung EnnovateX 2025 AI Challenge Phase 1, focusing on real-time network traffic classification for social media applications.
