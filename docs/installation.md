# Installation Guide

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Network**: Internet connection for package downloads

### Recommended Requirements
- **RAM**: 16GB for large dataset processing
- **CPU**: Multi-core processor for faster training
- **GPU**: NVIDIA GPU with CUDA support (optional, for CNN acceleration)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/samsung-traffic-detection.git
cd samsung-traffic-detection
```

### 2. Create Virtual Environment

#### Using venv (Recommended)
```bash
python -m venv samsung-env
```

#### Activate Virtual Environment

**Windows:**
```cmd
samsung-env\Scripts\activate
```

**Linux/macOS:**
```bash
source samsung-env/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 4. Platform-Specific Setup

#### Windows Setup

1. **Install Npcap for packet capture:**
   - Download from: https://nmap.org/npcap/
   - Run installer with "WinPcap API-compatible mode" enabled

2. **Install Visual C++ Build Tools (if needed):**
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

#### Linux Setup

1. **Install system dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install python3-dev libpcap-dev tcpdump

   # CentOS/RHEL/Fedora
   sudo yum install python3-devel libpcap-devel tcpdump
   ```

2. **Set capabilities for packet capture (optional):**
   ```bash
   sudo setcap cap_net_raw,cap_net_admin=eip $(which python)
   ```

#### macOS Setup

1. **Install Homebrew (if not already installed):**
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install dependencies:**
   ```bash
   brew install libpcap
   ```

### 5. Verify Installation

Run the verification script:
```bash
python -c "
import sys
print('Python version:', sys.version)

# Test core imports
try:
    import numpy, pandas, sklearn
    print('✓ Core scientific packages installed')
except ImportError as e:
    print('✗ Missing core packages:', e)

try:
    import xgboost, lightgbm
    print('✓ ML packages installed')
except ImportError as e:
    print('✗ Missing ML packages:', e)

try:
    import scapy, pyshark
    print('✓ Network packages installed')
except ImportError as e:
    print('✗ Missing network packages:', e)

try:
    import fastapi, uvicorn
    print('✓ Web framework installed')
except ImportError as e:
    print('✗ Missing web packages:', e)

print('Installation verification complete!')
"
```
## How to Run the Project

After completing installation and verification, follow these steps to run the real-time traffic classification system:

### 1. Activate Your Python Environment
Activate your virtual environment if not already active:
- **Windows (PowerShell):**
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
- **Linux/macOS (bash/zsh):**
   ```bash
   source .venv/bin/activate
   ```

### 2. Start the Real-Time Inference Pipeline
Run the real-time packet capture and inference engine:
```powershell
python src/inference/capture.py
```
This will start packet capture, feature extraction, and classification. Results are streamed to the dashboard.

### 3. Launch the Dashboard
Start the FastAPI dashboard server:
```powershell
uvicorn src.dashboard.main:app --reload
```
The dashboard will be available at [http://localhost:8000](http://localhost:8000).

### 4. View Results
Open your browser and navigate to [http://localhost:8000](http://localhost:8000) to view real-time traffic classification results.

### 5. Stopping the System
Press `Ctrl+C` in the terminal to stop packet capture or dashboard server.

---

**Note:**
- For custom model paths or configuration, edit `src/inference/infer.py` and `src/dashboard/main.py` as needed.
- For troubleshooting, see the FAQ section in this guide.

## Optional Components

### GPU Support (NVIDIA)

For accelerated CNN training:

1. **Install CUDA Toolkit:**
   - Download from: https://developer.nvidia.com/cuda-downloads

2. **Install PyTorch with CUDA:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install ONNX Runtime GPU:**
   ```bash
   pip install onnxruntime-gpu
   ```

### Android Development

For Android traffic capture:

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
   - Go back to Settings > Developer Options
   - Enable "USB Debugging"

## Configuration

### 1. Create Configuration Directory

```bash
mkdir -p ~/.samsung
```

### 2. Create Basic Configuration

Create `~/.samsung/config.json`:
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

## Testing Installation

### 1. Test Data Collection

```bash
# Test PC capture (requires admin/root privileges)
python src/data_collection/capture_pc.py --duration 10 --verbose

# Test Android detection
python src/data_collection/capture_android.py --list-devices
```

### 2. Test Model Training

```bash
# Generate sample data and train a model
python scripts/train_models.py --generate-sample --data data/processed/sample_features.csv --verbose
```

### 3. Test Dashboard

```bash
# Start the dashboard
python src/inference/dashboard.py --verbose

# Open browser to http://localhost:8080
```

## Troubleshooting

### Common Issues

#### 1. Permission Denied for Packet Capture

**Linux/macOS:**
```bash
# Run with sudo
sudo python src/data_collection/capture_pc.py

# Or set capabilities (Linux only)
sudo setcap cap_net_raw,cap_net_admin=eip $(which python)
```

**Windows:**
- Run Command Prompt as Administrator
- Ensure Npcap is installed correctly

#### 2. ImportError: No module named 'scapy'

```bash
# Reinstall with specific version
pip install scapy==2.4.5
```

#### 3. XGBoost Installation Issues

**Windows:**
```bash
# Install Microsoft C++ Build Tools first
pip install xgboost
```

**Linux:**
```bash
# Install build dependencies
sudo apt-get install build-essential
pip install xgboost
```

#### 4. ONNX Export Issues

```bash
# Install specific ONNX versions
pip install onnx==1.12.0 onnxmltools==1.11.1
```

#### 5. Dashboard Not Accessible

- Check firewall settings
- Ensure port 8080 is not in use:
  ```bash
  # Check port usage
  netstat -tulpn | grep 8080
  
  # Use different port
  python src/inference/dashboard.py --port 8081
  ```

### Environment Variables

Set these environment variables if needed:

```bash
# Windows
set PYTHONPATH=%PYTHONPATH%;%CD%\src
set SAMSUNG_DATA_DIR=.\data

# Linux/macOS
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export SAMSUNG_DATA_DIR=./data
```

### Log Files

Check log files for detailed error information:
- `~/.samsung/logs/capture.log`
- `~/.samsung/logs/training.log`
- `~/.samsung/logs/dashboard.log`

## Performance Optimization

### 1. Increase Buffer Sizes

For high-traffic environments:
```json
{
  "capture": {
    "buffer_size": 50000,
    "max_packets_per_second": 10000
  }
}
```

### 2. Enable Parallel Processing

```bash
# Use multiple CPU cores for training
export OMP_NUM_THREADS=4
python scripts/train_models.py --data data/processed/features.csv
```

### 3. Memory Optimization

For large datasets:
```json
{
  "preprocessing": {
    "chunk_size": 10000,
    "low_memory_mode": true
  }
}
```

## Updating

To update to the latest version:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Reinstall package
pip install -e .
```

## Uninstallation

To completely remove the installation:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf samsung-env

# Remove configuration
rm -rf ~/.samsung

# Remove cloned repository
rm -rf samsung-traffic-detection
```

## Support

For installation issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Search existing issues on GitHub
3. Create a new issue with:
   - Operating system and version
   - Python version
   - Error messages and full traceback
   - Steps to reproduce the issue
