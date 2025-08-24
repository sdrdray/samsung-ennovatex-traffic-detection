# Installation Guide

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 4GB minimum free space
- **Network**: Internet connection for package downloads

### Recommended Requirements
- **RAM**: 8GB for optimal performance
- **CPU**: Multi-core processor for faster processing
- **Network**: Stable internet connection for real-time monitoring

## Quick Start Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sdrdray/samsung-ennovatex-traffic-detection.git
cd samsung-ennovatex-traffic-detection
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv samsung-env

# Activate virtual environment
# Windows:
samsung-env\Scripts\activate
# Linux/macOS:
source samsung-env/bin/activate
```

### 3. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

## Running the Project

### Option 1: Live Dashboard (Recommended for Demo)

The main demo application that shows real-time traffic classification:

```bash
python live_dashboard.py
```

Then open your browser to: **http://localhost:8080**

This will:
- Start real-time network monitoring using `psutil`
- Load the trained XGBoost model
- Classify traffic as "Video/Reel" or "Regular Traffic"
- Display results in a live web dashboard

### Option 2: Data Collection (For Research)

To capture your own network traffic data:

```bash
# Capture network traffic (requires admin privileges)
python src/data_collection/capture_pc.py --interface "Wi-Fi" --duration 60
```

### Option 3: Model Training

To train the model with your own data:

```bash
# Prepare dataset from raw JSON files
python scripts/prepare_dataset.py

# Train the XGBoost model
python scripts/train_models.py
```

## Project Structure Overview

```
samsung-ennovatex-traffic-detection/
├── live_dashboard.py          # Main demo application (START HERE)
├── src/
│   ├── data_collection/       # Network traffic capture
│   ├── features/             # Feature extraction
│   ├── models/               # ML model implementations
│   └── utils/                # Helper utilities
├── scripts/
│   ├── prepare_dataset.py    # Data preprocessing
│   └── train_models.py       # Model training
├── models/
│   └── xgboost_model.pkl     # Trained model (ready to use)
├── data/
│   ├── raw/                  # Raw network capture files
│   └── processed/            # Processed feature datasets
└── docs/                     # Documentation
```

## Verification

Test if everything is working:

```bash
python -c "
import xgboost as xgb
import pandas as pd
import psutil
print('✓ All core packages installed successfully')
print('✓ Ready to run the demo!')
"
```

## Troubleshooting

### Common Issues

#### "Permission Denied" for Network Monitoring
- **Windows**: Run Command Prompt as Administrator
- **Linux/macOS**: Use `sudo python live_dashboard.py`

#### "Module not found" errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Dashboard not accessible
- Check if port 8080 is free
- Try: `python live_dashboard.py --port 8081`

### Quick Fixes

```bash
# Update pip first
pip install --upgrade pip

---

## Advanced Setup (For Future Development)

The following sections are for advanced users who want to extend the project with additional capabilities:

### Deep Packet Capture Setup

For advanced packet analysis using Scapy (requires admin privileges):

#### Windows Setup
1. **Install Npcap for packet capture:**
   - Download from: https://nmap.org/npcap/
   - Run installer with "WinPcap API-compatible mode" enabled

#### Linux Setup
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev libpcap-dev tcpdump

# Set capabilities for packet capture
sudo setcap cap_net_raw,cap_net_admin=eip $(which python)
```

#### macOS Setup
```bash
# Install Homebrew dependencies
brew install libpcap
```

### GPU Support for CNN Models

For accelerated CNN training (future enhancement):

1. **Install CUDA Toolkit:**
   - Download from: https://developer.nvidia.com/cuda-downloads

2. **Install PyTorch with CUDA:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Android Traffic Capture

For mobile traffic analysis (future feature):

1. **Install ADB (Android Debug Bridge):**
   ```bash
   # Windows (via Chocolatey)
   choco install adb

   # macOS (via Homebrew)
   brew install android-platform-tools

   # Linux
   sudo apt-get install android-tools-adb
   ```

2. **Enable Developer Options on Android device:**
   - Go to Settings > About Phone
   - Tap "Build Number" 7 times
   - Enable "USB Debugging" in Developer Options

### Advanced Configuration

Create `~/.samsung/config.json` for advanced settings:
```json
{
  "data_dir": "./data",
  "models_dir": "./models",
  "log_level": "INFO",
  "capture": {
    "interface": "auto",
    "buffer_size": 10000
  },
  "models": {
    "xgboost": {
      "n_estimators": 200,
      "max_depth": 8,
      "learning_rate": 0.1
    }
  },
  "dashboard": {
    "host": "localhost",
    "port": 8080
  }
}
```

### Performance Optimization

For high-traffic environments:

```bash
# Use multiple CPU cores for processing
export OMP_NUM_THREADS=4

# Increase system buffer sizes
ulimit -n 65536
```

### Advanced Testing

```bash
# Test advanced packet capture
python src/data_collection/capture_pc.py --duration 10 --verbose

# Test Android device detection
python src/data_collection/capture_android.py --list-devices

# Test CNN model training
python scripts/train_models.py --model cnn --verbose
```

### Environment Variables

Set these for advanced configurations:

```bash
# Windows
set PYTHONPATH=%PYTHONPATH%;%CD%\src
set SAMSUNG_DATA_DIR=.\data

# Linux/macOS
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export SAMSUNG_DATA_DIR=./data
```

### Log Files

Advanced logging locations:
- `~/.samsung/logs/capture.log`
- `~/.samsung/logs/training.log`
- `~/.samsung/logs/dashboard.log`

### Uninstallation

To completely remove the installation:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf samsung-env

# Remove configuration
rm -rf ~/.samsung

# Remove cloned repository
rm -rf samsung-ennovatex-traffic-detection
```
