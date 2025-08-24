#!/usr/bin/env python3
"""
Feature Preprocessor Module

Preprocesses extracted features for machine learning training including
normalization, scaling, feature selection, and handling missing values.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import pickle
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from ..utils.logging import setup_logger


class FeaturePreprocessor:
    """
    Preprocess features for machine learning training.
    
    Handles missing values, feature scaling, selection, and train/test splitting
    with proper data leakage prevention.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature preprocessor.
        
        Args:
            config: Configuration dictionary for preprocessing options
        """
        self.logger = setup_logger("feature_preprocessor")
        
        # Default configuration
        default_config = {
            'imputation_strategy': 'median',  # 'mean', 'median', 'mode', 'constant'
            'scaling_method': 'standard',     # 'standard', 'minmax', 'robust', 'none'
            'feature_selection': True,        # Whether to perform feature selection
            'selection_method': 'f_classif',  # 'f_classif', 'mutual_info', 'variance'
            'n_features': 25,                 # Number of features to select
            'handle_outliers': True,          # Whether to handle outliers
            'outlier_method': 'iqr',          # 'iqr', 'zscore', 'isolation'
            'test_size': 0.2,                 # Train/test split ratio
            'random_state': 42,               # Random seed for reproducibility
            'validation_split': True,         # Whether to create validation set
            'val_size': 0.2                   # Validation set size from training data
        }
        
        self.config = default_config
        if config:
            self.config.update(config)
        
        # Fitted preprocessors (will be set during fit)
        self.imputer = None
        self.scaler = None
        self.feature_selector = None
        self.feature_names = None
        self.selected_features = None
        
        # Statistics for outlier detection
        self.outlier_bounds = None
        
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeaturePreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Feature matrix
            y: Target labels (required for supervised feature selection)
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting preprocessor on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Store original feature names
        self.feature_names = list(X.columns)
        
        # Create a copy to avoid modifying original data
        X_processed = X.copy()
        
        # Step 1: Handle missing values
        self._fit_imputer(X_processed)
        X_processed = pd.DataFrame(
            self.imputer.transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )
        
        # Step 2: Handle outliers
        if self.config['handle_outliers']:
            self._fit_outlier_detector(X_processed)
            X_processed = self._remove_outliers(X_processed)
        
        # Step 3: Fit scaler
        self._fit_scaler(X_processed)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )
        
        # Step 4: Feature selection
        if self.config['feature_selection'] and y is not None:
            self._fit_feature_selector(X_scaled, y)
        
        self.is_fitted = True
        self.logger.info("Preprocessor fitting completed")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted preprocessor.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        self.logger.debug(f"Transforming {X.shape[0]} samples")
        
        # Ensure we have the same features as training
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            self.logger.warning(f"Missing features in input: {missing_features}")
            for feature in missing_features:
                X[feature] = 0  # Add missing features with default value
        
        # Reorder columns to match training
        X = X[self.feature_names]
        
        # Apply transformations in the same order as fitting
        X_processed = X.copy()
        
        # Step 1: Impute missing values
        X_processed = pd.DataFrame(
            self.imputer.transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )
        
        # Step 2: Handle outliers (clipping, not removal for new data)
        if self.config['handle_outliers'] and self.outlier_bounds is not None:
            X_processed = self._clip_outliers(X_processed)
        
        # Step 3: Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )
        
        # Step 4: Select features
        if self.feature_selector is not None:
            X_selected = X_scaled[self.selected_features]
            return X_selected
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit preprocessor and transform features in one step.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)
    
    def _fit_imputer(self, X: pd.DataFrame):
        """Fit imputer for missing values."""
        strategy = self.config['imputation_strategy']
        
        if strategy == 'mode':
            # Use most frequent for categorical data
            self.imputer = SimpleImputer(strategy='most_frequent')
        elif strategy == 'constant':
            self.imputer = SimpleImputer(strategy='constant', fill_value=0)
        else:
            self.imputer = SimpleImputer(strategy=strategy)
        
        self.imputer.fit(X)
        
        missing_stats = X.isnull().sum()
        if missing_stats.sum() > 0:
            self.logger.info(f"Missing values found in {missing_stats[missing_stats > 0].shape[0]} features")
        
    def _fit_outlier_detector(self, X: pd.DataFrame):
        """Fit outlier detection parameters."""
        method = self.config['outlier_method']
        
        if method == 'iqr':
            # Interquartile Range method
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.outlier_bounds = {
                'lower': lower_bound,
                'upper': upper_bound
            }
            
        elif method == 'zscore':
            # Z-score method
            mean = X.mean()
            std = X.std()
            
            self.outlier_bounds = {
                'lower': mean - 3 * std,
                'upper': mean + 3 * std
            }
    
    def _remove_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from training data."""
        if self.outlier_bounds is None:
            return X
        
        method = self.config['outlier_method']
        
        if method in ['iqr', 'zscore']:
            # Create mask for non-outlier rows
            outlier_mask = pd.Series(True, index=X.index)
            
            for column in X.columns:
                if column in self.outlier_bounds['lower']:
                    lower = self.outlier_bounds['lower'][column]
                    upper = self.outlier_bounds['upper'][column]
                    
                    column_mask = (X[column] >= lower) & (X[column] <= upper)
                    outlier_mask = outlier_mask & column_mask
            
            outliers_removed = (~outlier_mask).sum()
            if outliers_removed > 0:
                self.logger.info(f"Removed {outliers_removed} outlier samples")
            
            return X[outlier_mask]
        
        return X
    
    def _clip_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers in new data instead of removing."""
        if self.outlier_bounds is None:
            return X
        
        X_clipped = X.copy()
        
        for column in X.columns:
            if column in self.outlier_bounds['lower']:
                lower = self.outlier_bounds['lower'][column]
                upper = self.outlier_bounds['upper'][column]
                
                X_clipped[column] = X_clipped[column].clip(lower=lower, upper=upper)
        
        return X_clipped
    
    def _fit_scaler(self, X: pd.DataFrame):
        """Fit feature scaler."""
        method = self.config['scaling_method']
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'none':
            # Identity scaler (no scaling)
            from sklearn.preprocessing import FunctionTransformer
            self.scaler = FunctionTransformer()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        self.scaler.fit(X)
        self.logger.info(f"Fitted {method} scaler")
    
    def _fit_feature_selector(self, X: pd.DataFrame, y: pd.Series):
        """Fit feature selector."""
        method = self.config['selection_method']
        n_features = min(self.config['n_features'], X.shape[1])
        
        if method == 'f_classif':
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        elif method == 'mutual_info':
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        self.feature_selector.fit(X, y)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()
        
        self.logger.info(f"Selected {len(self.selected_features)} features using {method}")
        self.logger.debug(f"Selected features: {self.selected_features}")
    
    def create_train_val_test_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Create train/validation/test splits with proper preprocessing.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info(f"Creating train/val/test split from {X.shape[0]} samples")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y if len(y.unique()) > 1 else None
        )
        
        # Second split: separate validation set from remaining data
        if self.config['validation_split']:
            val_size = self.config['val_size']
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size,
                random_state=self.config['random_state'],
                stratify=y_temp if len(y_temp.unique()) > 1 else None
            )
        else:
            X_train, X_val = X_temp, pd.DataFrame()
            y_train, y_val = y_temp, pd.Series()
        
        # Fit preprocessor on training data only
        self.fit(X_train, y_train)
        
        # Transform all splits
        X_train_processed = self.transform(X_train)
        X_test_processed = self.transform(X_test)
        
        if self.config['validation_split']:
            X_val_processed = self.transform(X_val)
        else:
            X_val_processed = pd.DataFrame()
        
        self.logger.info(f"Split sizes - Train: {len(X_train_processed)}, "
                        f"Val: {len(X_val_processed)}, Test: {len(X_test_processed)}")
        
        return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores from feature selector.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_selector is None:
            return None
        
        scores = self.feature_selector.scores_
        selected_mask = self.feature_selector.get_support()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'score': scores,
            'selected': selected_mask
        }).sort_values('score', ascending=False)
        
        return importance_df
    
    def save(self, filepath: str):
        """
        Save fitted preprocessor to file.
        
        Args:
            filepath: Path to save the preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        save_data = {
            'config': self.config,
            'imputer': self.imputer,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'outlier_bounds': self.outlier_bounds,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        self.logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeaturePreprocessor':
        """
        Load fitted preprocessor from file.
        
        Args:
            filepath: Path to load the preprocessor from
            
        Returns:
            Loaded preprocessor instance
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        preprocessor = cls(save_data['config'])
        preprocessor.imputer = save_data['imputer']
        preprocessor.scaler = save_data['scaler']
        preprocessor.feature_selector = save_data['feature_selector']
        preprocessor.feature_names = save_data['feature_names']
        preprocessor.selected_features = save_data['selected_features']
        preprocessor.outlier_bounds = save_data['outlier_bounds']
        preprocessor.is_fitted = save_data['is_fitted']
        
        return preprocessor
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing steps applied.
        
        Returns:
            Dictionary with preprocessing summary
        """
        summary = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'original_features': len(self.feature_names) if self.feature_names else 0,
            'selected_features': len(self.selected_features) if self.selected_features else 0,
            'scaling_method': self.config['scaling_method'],
            'imputation_strategy': self.config['imputation_strategy'],
            'feature_selection_enabled': self.config['feature_selection']
        }
        
        if self.is_fitted and self.selected_features:
            summary['selected_feature_names'] = self.selected_features
        
        return summary
