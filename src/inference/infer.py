#!/usr/bin/env python3
"""
Real-time inference engine for traffic classification.

This module provides the main inference engine that coordinates
model loading, feature processing, and prediction generation for
real-time traffic classification.
"""

import logging
import time
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import threading
from queue import Queue, Empty
import json
import sys

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Model imports
from src.models.xgboost_model import XGBoostModel
from src.models.cnn_model import CNNModel
from src.models.ensemble import EnsembleModel

# Feature processing
from src.features.preprocessor import FeaturePreprocessor

# Utils
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Main inference engine for real-time traffic classification.
    
    This class handles model loading, feature preprocessing, prediction
    generation, and result post-processing for the complete inference pipeline.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_ensemble: bool = True,
        confidence_threshold: float = 0.8,
        preprocessing_config: Optional[Dict] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to saved models directory
            use_ensemble: Whether to use ensemble approach
            confidence_threshold: Minimum confidence for positive predictions
            preprocessing_config: Configuration for data preprocessing
        """
        self.model_path = model_path or "models/"
        self.use_ensemble = use_ensemble
        self.confidence_threshold = confidence_threshold
        self.preprocessing_config = preprocessing_config or {}
        
        # Model components
        self.xgboost_model = None
        self.cnn_model = None
        self.ensemble_model = None
        self.preprocessor = FeaturePreprocessor()
        
        # State management
        self.is_initialized = False
        self.models_loaded = False
        self.last_prediction_time = 0
        
        # Performance tracking
        self.stats = {
            'predictions_made': 0,
            'total_inference_time': 0.0,
            'avg_inference_time': 0.0,
            'max_inference_time': 0.0,
            'confidence_scores': [],
            'prediction_history': []
        }
        
        # Thread safety
        self.inference_lock = threading.Lock()
        
        logger.info("Initialized inference engine")
    
    def initialize(self) -> bool:
        """
        Initialize the inference engine and load models.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing inference engine...")
            
            # Load models
            if not self._load_models():
                logger.error("Failed to load models")
                return False
            
            # Initialize preprocessor
            if not self._initialize_preprocessor():
                logger.error("Failed to initialize preprocessor")
                return False
            
            self.is_initialized = True
            logger.info("Inference engine initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing inference engine: {e}")
            return False
    
    def predict(self, features: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Make predictions on input features.
        
        Args:
            features: DataFrame with extracted features
            
        Returns:
            Array of predictions (0 = regular traffic, 1 = reel/video traffic)
        """
        if not self.is_initialized:
            logger.warning("Inference engine not initialized")
            return None
        
        start_time = time.time()
        
        try:
            with self.inference_lock:
                # Preprocess features
                processed_features = self._preprocess_features(features)
                
                if processed_features is None or processed_features.empty:
                    logger.warning("Feature preprocessing failed")
                    return None
                
                # Make prediction
                if self.use_ensemble and self.ensemble_model is not None:
                    predictions = self.ensemble_model.predict(processed_features)
                elif self.xgboost_model is not None:
                    predictions = self.xgboost_model.predict(processed_features)
                else:
                    logger.error("No model available for prediction")
                    return None
                
                # Update statistics
                inference_time = time.time() - start_time
                self._update_stats(inference_time, predictions)
                
                return predictions
        
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def predict_proba(self, features: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Predict class probabilities for input features.
        
        Args:
            features: DataFrame with extracted features
            
        Returns:
            Array of class probabilities
        """
        if not self.is_initialized:
            logger.warning("Inference engine not initialized")
            return None
        
        try:
            with self.inference_lock:
                # Preprocess features
                processed_features = self._preprocess_features(features)
                
                if processed_features is None or processed_features.empty:
                    return None
                
                # Get probabilities
                if self.use_ensemble and self.ensemble_model is not None:
                    probabilities = self.ensemble_model.predict_proba(processed_features)
                elif self.xgboost_model is not None:
                    probabilities = self.xgboost_model.predict_proba(processed_features)
                else:
                    logger.error("No model available for probability prediction")
                    return None
                
                return probabilities
        
        except Exception as e:
            logger.error(f"Error predicting probabilities: {e}")
            return None
    
    def predict_confidence(self, features: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Predict confidence scores for input features.
        
        Args:
            features: DataFrame with extracted features
            
        Returns:
            Array of confidence scores (0-1)
        """
        probabilities = self.predict_proba(features)
        
        if probabilities is None:
            return None
        
        # Calculate confidence as maximum probability
        if probabilities.ndim == 1:
            # Single probability (binary classification)
            confidence = np.maximum(probabilities, 1 - probabilities)
        else:
            # Multiple probabilities
            confidence = np.max(probabilities, axis=1)
        
        return confidence
    
    def classify_with_confidence(self, features: pd.DataFrame) -> Optional[Dict]:
        """
        Classify traffic with confidence scores and metadata.
        
        Args:
            features: DataFrame with extracted features
            
        Returns:
            Dictionary with prediction, confidence, and metadata
        """
        try:
            # Get predictions and probabilities
            predictions = self.predict(features)
            probabilities = self.predict_proba(features)
            confidence = self.predict_confidence(features)
            
            if predictions is None or probabilities is None or confidence is None:
                return None
            
            # Create result dictionary
            result = {
                'prediction': int(predictions[0]),
                'confidence': float(confidence[0]),
                'probabilities': probabilities[0].tolist() if probabilities.ndim > 1 else [1-probabilities[0], probabilities[0]],
                'is_reel_traffic': bool(predictions[0] == 1 and confidence[0] >= self.confidence_threshold),
                'timestamp': time.time(),
                'model_type': 'ensemble' if self.use_ensemble else 'xgboost'
            }
            
            # Add model-specific information
            if self.use_ensemble and self.ensemble_model is not None:
                result['model_contributions'] = self._get_model_contributions(features)
            
            # Store in history
            self.stats['prediction_history'].append(result)
            
            # Keep only recent history
            if len(self.stats['prediction_history']) > 1000:
                self.stats['prediction_history'] = self.stats['prediction_history'][-1000:]
            
            return result
        
        except Exception as e:
            logger.error(f"Error in classification with confidence: {e}")
            return None
    
    def batch_predict(self, features_list: List[pd.DataFrame]) -> List[Optional[Dict]]:
        """
        Process multiple feature sets in batch.
        
        Args:
            features_list: List of feature DataFrames
            
        Returns:
            List of prediction results
        """
        results = []
        
        for features in features_list:
            result = self.classify_with_confidence(features)
            results.append(result)
        
        return results
    
    def _load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            logger.info("Loading models...")
            
            # Load XGBoost model
            xgboost_path = os.path.join(self.model_path, "xgboost_model.pkl")
            if os.path.exists(xgboost_path):
                self.xgboost_model = XGBoostModel.load_model(xgboost_path)
                logger.info("Loaded XGBoost model")
            else:
                logger.warning(f"XGBoost model not found at {xgboost_path}")
            
            # Load CNN model
            cnn_path = os.path.join(self.model_path, "cnn_model.pkl")
            if os.path.exists(cnn_path):
                try:
                    self.cnn_model = CNNModel.load_model(cnn_path)
                    logger.info("Loaded CNN model")
                except Exception as e:
                    logger.warning(f"Could not load CNN model: {e}")
            else:
                logger.warning(f"CNN model not found at {cnn_path}")
            
            # Load ensemble model if both base models available
            if self.use_ensemble and self.xgboost_model is not None:
                ensemble_path = os.path.join(self.model_path, "ensemble")
                if os.path.exists(ensemble_path):
                    try:
                        self.ensemble_model = EnsembleModel.load_ensemble(ensemble_path)
                        logger.info("Loaded ensemble model")
                    except Exception as e:
                        logger.warning(f"Could not load ensemble model: {e}")
                        # Create ensemble with available models
                        self._create_runtime_ensemble()
                else:
                    logger.info("Ensemble not found, creating runtime ensemble")
                    self._create_runtime_ensemble()
            
            # Check if at least one model is loaded
            if (self.xgboost_model is None and 
                self.cnn_model is None and 
                self.ensemble_model is None):
                logger.error("No models could be loaded")
                return False
            
            self.models_loaded = True
            return True
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def _create_runtime_ensemble(self):
        """Create ensemble model at runtime with available models."""
        try:
            if self.xgboost_model is None:
                logger.warning("Cannot create ensemble without XGBoost model")
                return
            
            self.ensemble_model = EnsembleModel(ensemble_method="weighted_voting")
            
            # Add XGBoost model
            self.ensemble_model.add_model(
                name="xgboost",
                model=self.xgboost_model,
                weight=0.7,
                confidence_threshold=0.8
            )
            
            # Add CNN model if available
            if self.cnn_model is not None:
                self.ensemble_model.add_model(
                    name="cnn",
                    model=self.cnn_model,
                    weight=0.3,
                    confidence_threshold=0.8
                )
            
            logger.info("Created runtime ensemble model")
        
        except Exception as e:
            logger.error(f"Error creating runtime ensemble: {e}")
    
    def _initialize_preprocessor(self) -> bool:
        """Initialize data preprocessor."""
        try:
            # Load preprocessing parameters if available
            preprocessor_path = os.path.join(self.model_path, "preprocessor.pkl")
            if os.path.exists(preprocessor_path):
                import joblib
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info("Loaded preprocessor parameters")
            else:
                logger.info("Using default preprocessor")
            
            return True
        
        except Exception as e:
            logger.error(f"Error initializing preprocessor: {e}")
            return False
    
    def _preprocess_features(self, features: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Preprocess features for model input.
        
        Args:
            features: Raw feature DataFrame
            
        Returns:
            Preprocessed features ready for model input
        """
        try:
            if features is None or features.empty:
                return None
            
            # Handle missing values
            processed = self.preprocessor.handle_missing_values(features)
            
            # Normalize features (if preprocessor is fitted)
            if hasattr(self.preprocessor, 'scaler') and self.preprocessor.scaler is not None:
                processed = self.preprocessor.normalize_features(processed)
            
            return processed
        
        except Exception as e:
            logger.error(f"Error preprocessing features: {e}")
            return None
    
    def _get_model_contributions(self, features: pd.DataFrame) -> Optional[Dict]:
        """Get individual model contributions in ensemble."""
        try:
            if self.ensemble_model is None:
                return None
            
            processed_features = self._preprocess_features(features)
            if processed_features is None:
                return None
            
            contributions = self.ensemble_model.get_model_contributions(processed_features)
            
            # Convert to serializable format
            serializable_contributions = {}
            for name, contrib in contributions.items():
                if isinstance(contrib, np.ndarray):
                    serializable_contributions[name] = contrib.tolist()
                else:
                    serializable_contributions[name] = contrib
            
            return serializable_contributions
        
        except Exception as e:
            logger.debug(f"Error getting model contributions: {e}")
            return None
    
    def _update_stats(self, inference_time: float, predictions: np.ndarray):
        """Update performance statistics."""
        self.stats['predictions_made'] += len(predictions)
        self.stats['total_inference_time'] += inference_time
        self.stats['avg_inference_time'] = (
            self.stats['total_inference_time'] / self.stats['predictions_made']
        )
        self.stats['max_inference_time'] = max(
            self.stats['max_inference_time'], inference_time
        )
        self.last_prediction_time = time.time()
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        info = {
            'is_initialized': self.is_initialized,
            'models_loaded': self.models_loaded,
            'use_ensemble': self.use_ensemble,
            'confidence_threshold': self.confidence_threshold,
            'model_path': self.model_path
        }
        
        # XGBoost model info
        if self.xgboost_model is not None:
            info['xgboost_model'] = self.xgboost_model.get_model_info()
        
        # CNN model info
        if self.cnn_model is not None:
            info['cnn_model'] = self.cnn_model.get_model_info()
        
        # Ensemble model info
        if self.ensemble_model is not None:
            info['ensemble_model'] = self.ensemble_model.get_ensemble_info()
        
        return info
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        stats = self.stats.copy()
        
        # Calculate additional metrics
        if self.stats['confidence_scores']:
            stats['avg_confidence'] = np.mean(self.stats['confidence_scores'])
            stats['min_confidence'] = np.min(self.stats['confidence_scores'])
            stats['max_confidence'] = np.max(self.stats['confidence_scores'])
        
        # Recent prediction rate
        if len(self.stats['prediction_history']) > 1:
            recent_predictions = self.stats['prediction_history'][-10:]
            if len(recent_predictions) > 1:
                time_span = (recent_predictions[-1]['timestamp'] - 
                           recent_predictions[0]['timestamp'])
                stats['recent_prediction_rate'] = len(recent_predictions) / max(time_span, 1.0)
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'predictions_made': 0,
            'total_inference_time': 0.0,
            'avg_inference_time': 0.0,
            'max_inference_time': 0.0,
            'confidence_scores': [],
            'prediction_history': []
        }
        
        logger.info("Reset performance statistics")
    
    def update_confidence_threshold(self, threshold: float):
        """Update confidence threshold for predictions."""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            logger.info(f"Updated confidence threshold to {threshold}")
        else:
            logger.warning(f"Invalid confidence threshold: {threshold}")
    
    def save_prediction_history(self, output_path: str):
        """Save prediction history to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.stats['prediction_history'], f, indent=2)
            
            logger.info(f"Saved prediction history to {output_path}")
        
        except Exception as e:
            logger.error(f"Error saving prediction history: {e}")


class AsyncInferenceEngine:
    """
    Asynchronous wrapper for the inference engine.
    
    Provides non-blocking inference capabilities for high-throughput scenarios.
    """
    
    def __init__(self, inference_engine: InferenceEngine, max_queue_size: int = 1000):
        """
        Initialize async inference engine.
        
        Args:
            inference_engine: The main inference engine
            max_queue_size: Maximum size of the inference queue
        """
        self.inference_engine = inference_engine
        self.input_queue = Queue(maxsize=max_queue_size)
        self.output_queue = Queue(maxsize=max_queue_size)
        
        self.is_running = False
        self.worker_thread = None
        
        logger.info("Initialized async inference engine")
    
    def start(self):
        """Start the async inference worker."""
        if self.is_running:
            logger.warning("Async inference already running")
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        logger.info("Started async inference worker")
    
    def stop(self):
        """Stop the async inference worker."""
        self.is_running = False
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        
        logger.info("Stopped async inference worker")
    
    def submit_inference(self, features: pd.DataFrame, request_id: Optional[str] = None) -> bool:
        """
        Submit features for async inference.
        
        Args:
            features: Feature DataFrame
            request_id: Optional request identifier
            
        Returns:
            True if submitted successfully, False if queue full
        """
        try:
            request = {
                'features': features,
                'request_id': request_id or f"req_{int(time.time() * 1000)}",
                'timestamp': time.time()
            }
            
            self.input_queue.put(request, timeout=0.1)
            return True
        
        except:
            return False
    
    def get_result(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get inference result.
        
        Args:
            timeout: Maximum wait time for result
            
        Returns:
            Result dictionary or None if timeout
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def _worker(self):
        """Worker thread for async inference."""
        while self.is_running:
            try:
                # Get request from queue
                request = self.input_queue.get(timeout=1.0)
                
                # Process inference
                result = self.inference_engine.classify_with_confidence(
                    request['features']
                )
                
                # Add request metadata
                if result is not None:
                    result['request_id'] = request['request_id']
                    result['request_timestamp'] = request['timestamp']
                    result['processing_time'] = time.time() - request['timestamp']
                
                # Put result in output queue
                try:
                    self.output_queue.put(result, timeout=0.1)
                except:
                    # Output queue full, skip result
                    logger.warning("Output queue full, dropping result")
            
            except Empty:
                continue
            
            except Exception as e:
                logger.error(f"Error in async inference worker: {e}")


if __name__ == "__main__":
    """Test the inference engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test inference engine")
    parser.add_argument('--model-path', type=str, default="models/", help='Path to models')
    parser.add_argument('--test-features', type=str, help='Path to test features CSV')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(level=log_level)
    
    # Create inference engine
    engine = InferenceEngine(model_path=args.model_path)
    
    # Initialize
    if not engine.initialize():
        logger.error("Failed to initialize inference engine")
        exit(1)
    
    # Test with sample features or provided file
    if args.test_features and os.path.exists(args.test_features):
        # Load test features
        test_features = pd.read_csv(args.test_features)
        logger.info(f"Loaded test features: {test_features.shape}")
    else:
        # Generate sample features
        logger.info("Generating sample features for testing...")
        test_features = pd.DataFrame({
            'packet_count': [50],
            'total_bytes': [75000],
            'avg_packet_size': [1500],
            'duration': [3.0],
            'download_upload_ratio': [10.5],
            'https_ratio': [0.8],
            'tcp_ratio': [0.9],
            'burst_count': [3],
            'avg_burst_size': [25000]
        })
        
        # Add more sample features to reach expected count
        feature_names = [
            'std_packet_size', 'packet_rate', 'mean_inter_arrival_time',
            'std_inter_arrival_time', 'min_inter_arrival_time', 'max_inter_arrival_time',
            'min_packet_size', 'max_packet_size', 'size_variance',
            'small_packet_ratio', 'medium_packet_ratio', 'large_packet_ratio',
            'udp_ratio', 'other_protocol_ratio', 'upload_packet_ratio',
            'download_packet_ratio', 'upload_byte_ratio', 'download_byte_ratio',
            'http_ratio', 'common_port_ratio', 'video_port_ratio',
            'avg_burst_duration', 'unique_connections', 'connection_diversity'
        ]
        
        for name in feature_names:
            test_features[name] = [np.random.random()]
    
    # Make prediction
    logger.info("Making prediction...")
    result = engine.classify_with_confidence(test_features)
    
    if result is not None:
        logger.info("Prediction result:")
        logger.info(json.dumps(result, indent=2))
    else:
        logger.error("Prediction failed")
    
    # Show model info
    model_info = engine.get_model_info()
    logger.info("Model information:")
    logger.info(json.dumps(model_info, indent=2))
    
    # Show performance stats
    perf_stats = engine.get_performance_stats()
    logger.info("Performance statistics:")
    logger.info(json.dumps(perf_stats, indent=2))
