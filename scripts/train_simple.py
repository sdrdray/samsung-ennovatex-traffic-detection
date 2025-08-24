#!/usr/bin/env python3
"""
Simple Model Training Script

Trains XGBoost model directly from the prepared CSV dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_prepare_data(data_path: Path):
    """Load and prepare the dataset."""
    logger.info(f"Loading dataset from {data_path}")
    
    # Load the CSV
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    
    # Separate features and labels
    feature_columns = [col for col in df.columns if col not in ['label', 'file_source', 'window_id']]
    X = df[feature_columns]
    y = df['label']
    
    logger.info(f"Features: {feature_columns}")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_columns


def train_model(X, y):
    """Train XGBoost model."""
    logger.info("Training XGBoost model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model Accuracy: {accuracy:.4f}")
    logger.info("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Regular Traffic', 'Reel/Video Traffic']))
    
    logger.info("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return model, accuracy, feature_importance


def save_model(model, output_dir: Path, feature_columns, accuracy):
    """Save the trained model and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    model_path = output_dir / "xgboost_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")
    
    # Save feature names
    features_path = output_dir / "feature_names.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump(feature_columns, f)
    logger.info(f"Feature names saved to {features_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'XGBoost',
        'accuracy': accuracy,
        'feature_count': len(feature_columns),
        'features': feature_columns
    }
    
    metadata_path = output_dir / "model_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Metadata saved to {metadata_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train traffic classification model")
    parser.add_argument('--data', required=True, help="Path to CSV dataset")
    parser.add_argument('--output', required=True, help="Output directory for model")
    
    args = parser.parse_args()
    
    data_path = Path(args.data)
    output_dir = Path(args.output)
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
    
    try:
        # Load and prepare data
        X, y, feature_columns = load_and_prepare_data(data_path)
        
        # Train model
        model, accuracy, feature_importance = train_model(X, y)
        
        # Save model
        save_model(model, output_dir, feature_columns, accuracy)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info(f"Model saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
