# Changelog

## [0.1.0] - 2025-08-24

### Added
- Initial project setup with modules for data collection, feature engineering, and modeling.
- Developed a robust data collection script (`src/data_collection/capture_pc.py`) using Scapy to capture live packet metadata.
- Implemented a feature extraction pipeline (`scripts/prepare_dataset.py`) to create a labeled dataset from raw JSON captures.
- Trained an XGBoost model on the collected data, achieving high accuracy on the test set.
- Created a real-time, all-in-one dashboard (`live_dashboard.py`) with FastAPI and WebSockets to visualize live classification results.
