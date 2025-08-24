"""
Ensemble model combining multiple classifiers.

This module provides ensemble methods for combining predictions from
multiple models to improve overall classification performance.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble model for combining multiple classifiers.
    
    This class provides methods for combining predictions from multiple
    models using various ensemble techniques like weighted voting, stacking,
    and adaptive weighting.
    """
    
    def __init__(self, ensemble_method: str = "weighted_voting"):
        """
        Initialize ensemble model.
        
        Args:
            ensemble_method: Method for combining predictions
                - "weighted_voting": Weighted voting based on model weights
                - "adaptive_voting": Adaptive weights based on confidence
                - "stacking": Use meta-learner for combination
                - "majority_voting": Simple majority voting
        """
        self.models = {}
        self.weights = {}
        self.model_performance = {}
        self.ensemble_method = ensemble_method
        self.is_fitted = False
        self.meta_learner = None
        
        # Performance tracking
        self.prediction_history = []
        self.confidence_thresholds = {}
        
        logger.info(f"Initialized ensemble model with method: {ensemble_method}")
    
    def add_model(self, name: str, model: Any, weight: float = 1.0, confidence_threshold: float = 0.8):
        """
        Add a model to the ensemble.
        
        Args:
            name: Model name/identifier
            model: Model instance with predict and predict_proba methods
            weight: Weight for this model in ensemble voting
            confidence_threshold: Minimum confidence threshold for this model
        """
        self.models[name] = model
        self.weights[name] = weight
        self.confidence_thresholds[name] = confidence_threshold
        
        logger.info(f"Added model '{name}' to ensemble with weight {weight}")
    
    def remove_model(self, name: str):
        """Remove a model from the ensemble."""
        if name in self.models:
            del self.models[name]
            del self.weights[name]
            if name in self.confidence_thresholds:
                del self.confidence_thresholds[name]
            logger.info(f"Removed model '{name}' from ensemble")
    
    def update_weights(self, performance_data: Dict[str, float]):
        """
        Update model weights based on performance data.
        
        Args:
            performance_data: Dictionary mapping model names to performance scores
        """
        total_performance = sum(performance_data.values())
        
        for name, performance in performance_data.items():
            if name in self.weights and total_performance > 0:
                # Update weight based on relative performance
                self.weights[name] = performance / total_performance
                self.model_performance[name] = performance
        
        logger.info(f"Updated ensemble weights based on performance: {self.weights}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Ensemble predictions
        """
        if not self.models:
            raise ValueError("No models added to ensemble")
        
        if self.ensemble_method == "weighted_voting":
            return self._weighted_voting_predict(X)
        elif self.ensemble_method == "adaptive_voting":
            return self._adaptive_voting_predict(X)
        elif self.ensemble_method == "majority_voting":
            return self._majority_voting_predict(X)
        elif self.ensemble_method == "stacking":
            return self._stacking_predict(X)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Ensemble probability predictions
        """
        if not self.models:
            raise ValueError("No models added to ensemble")
        
        if self.ensemble_method == "weighted_voting":
            return self._weighted_voting_proba(X)
        elif self.ensemble_method == "adaptive_voting":
            return self._adaptive_voting_proba(X)
        elif self.ensemble_method == "majority_voting":
            return self._majority_voting_proba(X)
        elif self.ensemble_method == "stacking":
            return self._stacking_proba(X)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _weighted_voting_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using weighted voting."""
        probabilities = self._weighted_voting_proba(X)
        return np.argmax(probabilities, axis=1)
    
    def _weighted_voting_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using weighted voting."""
        probabilities = {}
        total_weight = 0
        
        # Collect predictions from all models
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    # Ensure proba is 2D (for binary classification)
                    if proba.ndim == 1:
                        proba = np.column_stack([1 - proba, proba])
                    probabilities[name] = proba
                    total_weight += self.weights.get(name, 1.0)
                else:
                    # Convert predictions to probabilities
                    pred = model.predict(X)
                    proba = np.zeros((len(pred), 2))
                    proba[np.arange(len(pred)), pred] = 1.0
                    probabilities[name] = proba
                    total_weight += self.weights.get(name, 1.0)
            except Exception as e:
                logger.warning(f"Model '{name}' failed to predict: {e}")
                continue
        
        if not probabilities:
            raise RuntimeError("All models failed to make predictions")
        
        # Combine probabilities using weighted averaging
        weighted_proba = None
        for name, proba in probabilities.items():
            weight = self.weights.get(name, 1.0)
            
            if weighted_proba is None:
                weighted_proba = weight * proba
            else:
                weighted_proba += weight * proba
        
        # Normalize by total weight
        if total_weight > 0:
            weighted_proba /= total_weight
        
        return weighted_proba
    
    def _adaptive_voting_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using adaptive voting based on confidence."""
        predictions = {}
        confidences = {}
        
        # Get predictions and confidences from all models
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    if proba.ndim == 1:
                        proba = np.column_stack([1 - proba, proba])
                    
                    pred = np.argmax(proba, axis=1)
                    conf = np.max(proba, axis=1)
                    
                    predictions[name] = pred
                    confidences[name] = conf
                else:
                    pred = model.predict(X)
                    predictions[name] = pred
                    confidences[name] = np.ones(len(pred)) * 0.5  # Default confidence
            except Exception as e:
                logger.warning(f"Model '{name}' failed in adaptive voting: {e}")
                continue
        
        if not predictions:
            raise RuntimeError("All models failed to make predictions")
        
        # Combine predictions based on confidence
        ensemble_pred = np.zeros(len(X), dtype=int)
        
        for i in range(len(X)):
            # Calculate adaptive weights for this sample
            sample_weights = {}
            total_weight = 0
            
            for name in predictions.keys():
                confidence = confidences[name][i]
                base_weight = self.weights.get(name, 1.0)
                
                # Boost weight if confidence is above threshold
                threshold = self.confidence_thresholds.get(name, 0.8)
                if confidence >= threshold:
                    adaptive_weight = base_weight * confidence
                else:
                    adaptive_weight = base_weight * 0.1  # Reduce weight for low confidence
                
                sample_weights[name] = adaptive_weight
                total_weight += adaptive_weight
            
            # Weighted voting for this sample
            class_votes = {0: 0, 1: 0}
            for name, pred in predictions.items():
                weight = sample_weights[name] / total_weight if total_weight > 0 else 0
                class_votes[pred[i]] += weight
            
            ensemble_pred[i] = max(class_votes, key=class_votes.get)
        
        return ensemble_pred
    
    def _adaptive_voting_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using adaptive voting."""
        # For simplicity, use predictions to estimate probabilities
        predictions = self._adaptive_voting_predict(X)
        proba = np.zeros((len(predictions), 2))
        proba[np.arange(len(predictions)), predictions] = 0.9  # High confidence for ensemble
        proba[np.arange(len(predictions)), 1 - predictions] = 0.1
        return proba
    
    def _majority_voting_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using simple majority voting."""
        predictions = {}
        
        # Collect predictions from all models
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Model '{name}' failed in majority voting: {e}")
                continue
        
        if not predictions:
            raise RuntimeError("All models failed to make predictions")
        
        # Majority voting
        ensemble_pred = np.zeros(len(X), dtype=int)
        
        for i in range(len(X)):
            votes = [pred[i] for pred in predictions.values()]
            ensemble_pred[i] = max(set(votes), key=votes.count)
        
        return ensemble_pred
    
    def _majority_voting_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using majority voting."""
        predictions = self._majority_voting_predict(X)
        proba = np.zeros((len(predictions), 2))
        proba[np.arange(len(predictions)), predictions] = 0.8  # Moderate confidence
        proba[np.arange(len(predictions)), 1 - predictions] = 0.2
        return proba
    
    def _stacking_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using stacking (meta-learner)."""
        if self.meta_learner is None:
            raise ValueError("Meta-learner not trained for stacking")
        
        # Get base model predictions as features for meta-learner
        meta_features = self._get_meta_features(X)
        return self.meta_learner.predict(meta_features)
    
    def _stacking_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using stacking."""
        if self.meta_learner is None:
            raise ValueError("Meta-learner not trained for stacking")
        
        meta_features = self._get_meta_features(X)
        if hasattr(self.meta_learner, 'predict_proba'):
            return self.meta_learner.predict_proba(meta_features)
        else:
            predictions = self.meta_learner.predict(meta_features)
            proba = np.zeros((len(predictions), 2))
            proba[np.arange(len(predictions)), predictions] = 0.8
            proba[np.arange(len(predictions)), 1 - predictions] = 0.2
            return proba
    
    def _get_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Get features for meta-learner from base model predictions."""
        meta_features = []
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    if proba.ndim == 1:
                        proba = np.column_stack([1 - proba, proba])
                    meta_features.append(proba)
                else:
                    pred = model.predict(X)
                    # Convert to one-hot
                    one_hot = np.zeros((len(pred), 2))
                    one_hot[np.arange(len(pred)), pred] = 1.0
                    meta_features.append(one_hot)
            except Exception as e:
                logger.warning(f"Model '{name}' failed to provide meta-features: {e}")
                continue
        
        if not meta_features:
            raise RuntimeError("No models could provide meta-features")
        
        return np.hstack(meta_features)
    
    def train_meta_learner(self, X_train: pd.DataFrame, y_train: np.ndarray, meta_model=None):
        """
        Train meta-learner for stacking ensemble.
        
        Args:
            X_train: Training features
            y_train: Training labels
            meta_model: Model to use as meta-learner (default: LogisticRegression)
        """
        if meta_model is None:
            from sklearn.linear_model import LogisticRegression
            meta_model = LogisticRegression(random_state=42)
        
        # Get meta-features from base models
        meta_features = self._get_meta_features(X_train)
        
        # Train meta-learner
        self.meta_learner = meta_model
        self.meta_learner.fit(meta_features, y_train)
        
        logger.info("Meta-learner trained for stacking ensemble")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble performance and individual model contributions.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Ensemble predictions
        ensemble_pred = self.predict(X_test)
        ensemble_proba = self.predict_proba(X_test)
        
        # Calculate ensemble metrics
        ensemble_metrics = {
            'ensemble_accuracy': accuracy_score(y_test, ensemble_pred),
            'ensemble_precision': precision_score(y_test, ensemble_pred, average='weighted'),
            'ensemble_recall': recall_score(y_test, ensemble_pred, average='weighted'),
            'ensemble_f1': f1_score(y_test, ensemble_pred, average='weighted'),
        }
        
        # Individual model metrics
        individual_metrics = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X_test)
                individual_metrics[f'{name}_accuracy'] = accuracy_score(y_test, pred)
                individual_metrics[f'{name}_f1'] = f1_score(y_test, pred, average='weighted')
            except Exception as e:
                logger.warning(f"Could not evaluate model '{name}': {e}")
        
        # Combine metrics
        all_metrics = {**ensemble_metrics, **individual_metrics}
        
        logger.info(f"Ensemble evaluation - Accuracy: {ensemble_metrics['ensemble_accuracy']:.4f}")
        
        return all_metrics
    
    def get_model_contributions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get individual model contributions to ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary mapping model names to their prediction contributions
        """
        contributions = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    if proba.ndim == 1:
                        proba = np.column_stack([1 - proba, proba])
                    contributions[name] = proba[:, 1]  # Probability of positive class
                else:
                    pred = model.predict(X)
                    contributions[name] = pred.astype(float)
            except Exception as e:
                logger.warning(f"Could not get contributions from model '{name}': {e}")
        
        return contributions
    
    def save_ensemble(self, save_path: str):
        """
        Save ensemble configuration and models.
        
        Args:
            save_path: Directory path to save ensemble
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save ensemble configuration
        config = {
            'ensemble_method': self.ensemble_method,
            'weights': self.weights,
            'confidence_thresholds': self.confidence_thresholds,
            'model_performance': self.model_performance,
            'is_fitted': self.is_fitted
        }
        
        config_path = os.path.join(save_path, 'ensemble_config.pkl')
        joblib.dump(config, config_path)
        
        # Save individual models
        for name, model in self.models.items():
            model_path = os.path.join(save_path, f'{name}_model.pkl')
            try:
                if hasattr(model, 'save_model'):
                    model.save_model(model_path)
                else:
                    joblib.dump(model, model_path)
            except Exception as e:
                logger.warning(f"Could not save model '{name}': {e}")
        
        # Save meta-learner if exists
        if self.meta_learner is not None:
            meta_path = os.path.join(save_path, 'meta_learner.pkl')
            joblib.dump(self.meta_learner, meta_path)
        
        logger.info(f"Ensemble saved to {save_path}")
    
    @classmethod
    def load_ensemble(cls, save_path: str) -> 'EnsembleModel':
        """
        Load ensemble from saved files.
        
        Args:
            save_path: Directory path to load ensemble from
            
        Returns:
            Loaded EnsembleModel instance
        """
        # Load configuration
        config_path = os.path.join(save_path, 'ensemble_config.pkl')
        config = joblib.load(config_path)
        
        # Create ensemble instance
        ensemble = cls(ensemble_method=config['ensemble_method'])
        ensemble.weights = config['weights']
        ensemble.confidence_thresholds = config['confidence_thresholds']
        ensemble.model_performance = config['model_performance']
        ensemble.is_fitted = config['is_fitted']
        
        # Load individual models
        for name in ensemble.weights.keys():
            model_path = os.path.join(save_path, f'{name}_model.pkl')
            if os.path.exists(model_path):
                try:
                    # Try to load using custom load_model method first
                    if name == 'xgboost':
                        from .xgboost_model import XGBoostModel
                        model = XGBoostModel.load_model(model_path)
                    elif name == 'cnn':
                        from .cnn_model import CNNModel
                        model = CNNModel.load_model(model_path)
                    else:
                        model = joblib.load(model_path)
                    
                    ensemble.models[name] = model
                except Exception as e:
                    logger.warning(f"Could not load model '{name}': {e}")
        
        # Load meta-learner if exists
        meta_path = os.path.join(save_path, 'meta_learner.pkl')
        if os.path.exists(meta_path):
            ensemble.meta_learner = joblib.load(meta_path)
        
        logger.info(f"Ensemble loaded from {save_path}")
        
        return ensemble
    
    def get_ensemble_info(self) -> Dict:
        """Get information about the ensemble configuration."""
        info = {
            "ensemble_method": self.ensemble_method,
            "num_models": len(self.models),
            "model_names": list(self.models.keys()),
            "weights": self.weights,
            "confidence_thresholds": self.confidence_thresholds,
            "is_fitted": self.is_fitted
        }
        
        if self.model_performance:
            info["model_performance"] = self.model_performance
        
        if self.meta_learner is not None:
            info["has_meta_learner"] = True
        
        return info
