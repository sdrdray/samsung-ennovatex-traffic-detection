# System Design Document

## 1. Architecture Overview

The system is designed as a multi-stage pipeline to address the challenge of real-time video traffic classification. The architecture separates the project into four logical stages: Data Collection, Feature Engineering, Model Training, and finally, Real-Time Inference & Visualization. This modular design allows for independent development and improvement of each component.

### High-Level Architecture

The overall workflow of the project can be visualized as follows:

1. **Data Collection** (`capture_pc.py`)
2. **Feature Engineering** (`prepare_dataset.py`) 
3. **Model Training** (`train_models.py`)
4. **Real-Time System** (`live_dashboard.py`)

```
┌─────────────────────┐
│   1. Data           │
│   Collection        │
│  (capture_pc.py)    │
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│   2. Feature        │
│   Engineering       │
│(prepare_dataset.py) │
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│   3. Model          │
│   Training          │
│ (train_models.py)   │
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│   4. Real-Time      │
│   System            │
│(live_dashboard.py)  │
└─────────────────────┘
```

## 2. Component Breakdown

### 2.1. Data Collection (`src/data_collection/capture_pc.py`)

The foundation of the AI model is a high-quality dataset. We developed a robust data collection script to capture real-world network traffic.

- **Technology**: The script uses the **Scapy** library in Python, a powerful tool for packet manipulation and sniffing.
- **Method**: It captures live network traffic from a specified network interface (e.g., "Wi-Fi"). To ensure the privacy of the user, the script is designed to only record packet **metadata** (like timestamps, packet sizes, protocol information, and port numbers) while explicitly ignoring all packet payloads.
- **Output**: The raw captures from different user activities (e.g., watching Reels, browsing Twitter) are saved as structured `.json` files in the `data/raw/` directory.

### 2.2. Feature Engineering (`scripts/prepare_dataset.py`)

Raw packet data is not suitable for direct use in a machine learning model. This stage transforms the raw data into a meaningful, structured format.

- **Process**: The preparation script reads the raw `.json` files and groups the thousands of individual packets into 3-second time windows. This aggregation transforms a stream of packets into a series of distinct observations.
- **Features**: For each 3-second window, a vector of 16 statistical features is calculated. These features are designed to be robust to varying network speeds and capture the underlying *pattern* of the traffic. Key features include:
  - `packet_count`: The volume of packets in the window.
  - `avg_packet_size` & `std_packet_size`: The average size and variability of packets.
  - `avg_inter_arrival` & `std_inter_arrival`: The average time between packets and its consistency.
  - `packets_per_second`: The frequency of packets.
  - `burst_ratio`: A measure of how "bursty" the traffic is, a key indicator for streaming.
- **Output**: The script produces a final, labeled `.csv` file (`MASTER_DATASET.csv`), which serves as the ground truth for training the AI model.

### 2.3. Model Training (`src/models/` & `scripts/train_models.py`)

This stage involves creating the "brain" of the system.

- **Model Choice**: We selected the **XGBoost Classifier**, a highly effective gradient boosting algorithm. It is well-suited for the tabular, statistical feature data we generated and is known for its high performance and accuracy.
- **Training Process**: The `scripts/train_models.py` script automates the training pipeline. It loads the master dataset, splits it into training and testing sets (to prevent overfitting), trains the XGBoost model, and evaluates its performance.
- **Output**: The final trained and validated model is saved as **`models/xgboost_model.pkl`**. This file is a portable, self-contained AI that can now be used for live predictions.

### 2.4. Real-Time Inference & Visualization (`live_dashboard.py`)

This is the final, user-facing component that brings all the previous stages together for a live demonstration.

- **Engine**: This script is an all-in-one application that handles both live analysis and visualization. For maximum stability and to avoid requiring administrator privileges during the live demo, it uses the `psutil` library to monitor high-level network statistics (like total bandwidth and packet rate) in a continuous background thread.
- **Inference**: The live statistics are fed into the loaded `xgboost_model.pkl`. The model makes a prediction in real-time, classifying the current network activity as either "Reel/Video" or "Regular Traffic" and providing a confidence score.
- **Dashboard Technology**: A web server is created using **FastAPI**. It serves a single-page HTML/CSS/JavaScript interface to the user.
- **Live Updates**: Communication between the backend analyzer and the front-end dashboard is handled by **WebSockets**. This allows the backend to push new classification results to the web browser instantly, creating a smooth, real-time visualization of the AI's decisions.

## 3. Data Flow Summary

### Training Pipeline
```
Raw .json Files ───> scripts/prepare_dataset.py ───> Labeled .csv ───> scripts/train_models.py ───> xgboost_model.pkl
```

### Real-Time Inference Pipeline (Live Demo)
```
Live Network Traffic ───> psutil Monitor ───> Loaded AI Model ───> Prediction ───> WebSocket ───> Web Dashboard
```

## 4. Technology Stack

- **Data Collection**: Scapy, psutil
- **Data Manipulation**: pandas, numpy
- **Machine Learning**: XGBoost, scikit-learn
- **Web Backend & API**: FastAPI, uvicorn
- **Real-Time Communication**: WebSockets
- **Web Frontend**: HTML, CSS, JavaScript

## 5. Future Enhancements

The project was designed with a modular structure that allows for significant future expansion.

- **Implement Advanced Models**: The codebase includes frameworks for a `cnn_model.py` and an `ensemble.py` model. These can be trained on the existing dataset to potentially improve classification accuracy by capturing sequential patterns in the data.
- **Enable Mobile Capture**: The `src/data_collection/capture_android.py` module provides the necessary functions to extend the data collection process to Android devices via USB tethering, allowing the model to be trained on mobile-specific traffic patterns.
- **Integrate Deep Packet Inspection for Inference**: The `real_capture.py` script provides a more advanced, Scapy-based capture method that can be used in place of `psutil` for the live demo. This would allow for the use of more granular, real-time features, potentially increasing accuracy at the cost of requiring administrator privileges.
