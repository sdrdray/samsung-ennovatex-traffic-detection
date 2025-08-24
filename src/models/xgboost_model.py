#!/usr/bin/env python3
"""
XGBoost Model Implementation

XGBoost classifier for tabular feature-based traffic classification.
Optimized for high performance on structured network traffic features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import pickle
import logging
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import onnx
import onnxmltools

from ..utils.logging import setup_logger


class XGBoostModel:
    """
    XGBoost classifier for network traffic classification.
    
    Provides training, evaluation, and ONNX export capabilities for 
    binary classification of reel/video vs non-reel traffic.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost model.
        
        Args:
            config: Model configuration parameters
        """
        self.logger = setup_logger("xgboost_model")
        
        # Default XGBoost configuration optimized for traffic classification
        default_config = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'early_stopping_rounds': 20,
            'verbose': False
        }
        
        self.config = default_config
        if config:
            self.config.update(config)
        
        # Initialize model
        self.model = xgb.XGBClassifier(**self.config)
        self.is_trained = False
        self.feature_names = None
        self.training_history = {}
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training results dictionary
        """
        self.logger.info(f"Training XGBoost model on {len(X_train)} samples")
        self.logger.info(f"Features: {X_train.shape[1]}, Classes: {len(y_train.unique())}")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Prepare evaluation sets
        eval_set = [(X_train, y_train)]
        eval_names = ['train']
        
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            eval_names.append('val')
            self.logger.info(f"Using validation set with {len(X_val)} samples")
        
        # Train model with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=self.config.get('verbose', False)
        )
        
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        
        training_results = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'train_precision': precision_score(y_train, train_pred, average='weighted'),
            'train_recall': recall_score(y_train, train_pred, average='weighted'),
            'train_f1': f1_score(y_train, train_pred, average='weighted'),
            'train_auc': roc_auc_score(y_train, train_pred_proba) if len(y_train.unique()) == 2 else None
        }
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_pred_proba = self.model.predict_proba(X_val)[:, 1]
            
            training_results.update({
                'val_accuracy': accuracy_score(y_val, val_pred),
                'val_precision': precision_score(y_val, val_pred, average='weighted'),
                'val_recall': recall_score(y_val, val_pred, average='weighted'),
                'val_f1': f1_score(y_val, val_pred, average='weighted'),
                'val_auc': roc_auc_score(y_val, val_pred_proba) if len(y_val.unique()) == 2 else None
            })
        
        # Store training history
        self.training_history = training_results
        
        # Log results
        self.logger.info(f"Training completed - Accuracy: {training_results['train_accuracy']:.4f}")
        if 'val_accuracy' in training_results:
            self.logger.info(f"Validation Accuracy: {training_results['val_accuracy']:.4f}")
        
        return training_results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure feature order matches training
        if self.feature_names:
            X = X[self.feature_names]
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure feature order matches training
        if self.feature_names:
            X = X[self.feature_names]
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics dictionary
        """
        self.logger.info(f"Evaluating model on {len(X_test)} test samples")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # AUC for binary classification
        if len(y_test.unique()) == 2:
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        # Log results
        self.logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"Test F1-Score: {metrics['f1']:.4f}")
        
        return metrics
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
            
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance_scores = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': list(importance_scores.keys()),
            'importance': list(importance_scores.values())
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, top_k: int = 20, save_path: Optional[str] = None):
        """
        Plot feature importance.
        
        Args:
            top_k: Number of top features to plot
            save_path: Optional path to save the plot
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to plot feature importance")
        
        try:
            import matplotlib.pyplot as plt
            
            importance_df = self.get_feature_importance()
            top_features = importance_df.head(top_k)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_k} Feature Importances (XGBoost)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Feature importance plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
    
    def save_model(self, filepath: str):
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        save_data = {
            'model': self.model,
            'config': self.config,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'XGBoostModel':
        """
        Load trained model from file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded XGBoost model instance
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        model = cls(save_data['config'])
        model.model = save_data['model']
        model.feature_names = save_data['feature_names']
        model.training_history = save_data['training_history']
        model.is_trained = save_data['is_trained']
        
        return model
    
    def export_to_onnx(self, filepath: str, X_sample: Optional[pd.DataFrame] = None):
        """
        Export model to ONNX format for inference.
        
        Args:
            filepath: Path to save ONNX model
            X_sample: Sample input for shape inference
        """
        if not self.is_trained:
            raise ValueError("Cannot export untrained model")
        
        try:
            # Convert to ONNX
            if X_sample is not None:
                initial_type = [('float_input', onnxmltools.convert.common.dataframes.guess_numpy_type(X_sample))]
            else:
                # Default to float32 with feature count
                n_features = len(self.feature_names) if self.feature_names else 10
                initial_type = [('float_input', onnxmltools.convert.common.data_types.FloatTensorType([None, n_features]))]
            
            onnx_model = onnxmltools.convert.xgboost.convert_xgboost(
                self.model,
                initial_types=initial_type
            )
            
            # Save ONNX model
            onnx.save_model(onnx_model, filepath)
            self.logger.info(f"Model exported to ONNX format: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export to ONNX: {e}")
            raise
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of model configuration and performance.
        
        Returns:
            Dictionary with model summary
        """
        summary = {
            'model_type': 'XGBoost',
            'config': self.config,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'training_history': self.training_history
        }
        
        if self.is_trained:
            summary['feature_names'] = self.feature_names
        
        return summary
    
    def hyperparameter_tune(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           param_grid: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            param_grid: Parameter grid for tuning
            
        Returns:
            Best parameters and performance
        """
        from sklearn.model_selection import GridSearchCV
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        
        self.logger.info("Starting hyperparameter tuning...")
        
        # Create XGBoost classifier for grid search
        xgb_classifier = xgb.XGBClassifier(
            random_state=self.config['random_state'],
            n_jobs=self.config['n_jobs']
        )
        
        # Perform grid search
        grid_search = GridSearchCV(
            xgb_classifier,
            param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.config.update(grid_search.best_params_)
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        # Evaluate on validation set
        val_score = grid_search.score(X_val, y_val)
        
        tuning_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'validation_score': val_score,
            'cv_results': grid_search.cv_results_
        }
        
        self.logger.info(f"Hyperparameter tuning completed. Best score: {grid_search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return tuning_results
