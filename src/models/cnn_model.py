"""
CNN Model for sequence-based traffic classification.

This module implements a 1D Convolutional Neural Network for analyzing
sequential patterns in network traffic data. The CNN model complements
the XGBoost model in the ensemble approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Union
import os

logger = logging.getLogger(__name__)


class Conv1DNetwork(nn.Module):
    """
    1D Convolutional Neural Network for traffic sequence classification.
    
    Architecture:
    - Input: (batch_size, sequence_length, features)
    - Conv1D layers with batch normalization and dropout
    - Global average pooling
    - Fully connected layers
    - Output: Binary classification (reel/video vs regular traffic)
    """
    
    def __init__(
        self,
        input_features: int = 37,
        sequence_length: int = 60,
        conv_filters: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [3, 3, 3],
        fc_units: List[int] = [512, 256],
        dropout_rate: float = 0.3,
        num_classes: int = 2
    ):
        """
        Initialize CNN model.
        
        Args:
            input_features: Number of input features per time step
            sequence_length: Length of input sequences
            conv_filters: Number of filters for each conv layer
            kernel_sizes: Kernel sizes for each conv layer
            fc_units: Units in fully connected layers
            dropout_rate: Dropout rate for regularization
            num_classes: Number of output classes
        """
        super(Conv1DNetwork, self).__init__()
        
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        in_channels = input_features
        for i, (out_channels, kernel_size) in enumerate(zip(conv_filters, kernel_sizes)):
            # 1D Convolutional layer
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            self.conv_layers.append(conv)
            
            # Batch normalization
            self.batch_norms.append(nn.BatchNorm1d(out_channels))
            
            in_channels = out_channels
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        
        # First FC layer (from conv output)
        fc_input_size = conv_filters[-1]
        for units in fc_units:
            self.fc_layers.append(nn.Linear(fc_input_size, units))
            fc_input_size = units
        
        # Output layer
        self.output_layer = nn.Linear(fc_input_size, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        logger.info(f"Initialized CNN model with {self._count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Transpose for Conv1d: (batch_size, features, sequence_length)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        for conv, batch_norm in zip(self.conv_layers, self.batch_norms):
            x = conv(x)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)  # Remove last dimension
        
        # Fully connected layers
        for fc in self.fc_layers:
            x = fc(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def _count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNModel:
    """
    Wrapper class for CNN model training and inference.
    
    This class provides a high-level interface for training and using
    the CNN model for traffic classification.
    """
    
    def __init__(
        self,
        input_features: int = 37,
        sequence_length: int = 60,
        model_config: Optional[Dict] = None
    ):
        """
        Initialize CNN model wrapper.
        
        Args:
            input_features: Number of input features per time step
            sequence_length: Length of input sequences
            model_config: Optional configuration dictionary
        """
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.model_config = model_config or {}
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        logger.info(f"Initialized CNN model wrapper for device: {self.device}")
    
    def _create_model(self) -> Conv1DNetwork:
        """Create CNN model with current configuration."""
        model = Conv1DNetwork(
            input_features=self.input_features,
            sequence_length=self.sequence_length,
            **self.model_config
        ).to(self.device)
        
        return model
    
    def _prepare_sequences(
        self,
        features: pd.DataFrame,
        labels: Optional[np.ndarray] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare sequential data for CNN training/inference.
        
        Args:
            features: Feature dataframe
            labels: Optional labels for training
            
        Returns:
            Tensor(s) for model input
        """
        # Convert to numpy array
        feature_array = features.values
        
        # Normalize features
        if not hasattr(self.scaler, 'scale_'):
            feature_array = self.scaler.fit_transform(feature_array)
        else:
            feature_array = self.scaler.transform(feature_array)
        
        # Create sequences
        sequences = []
        sequence_labels = []
        
        for i in range(len(feature_array) - self.sequence_length + 1):
            sequence = feature_array[i:i + self.sequence_length]
            sequences.append(sequence)
            
            if labels is not None:
                # Use label from the last time step
                sequence_labels.append(labels[i + self.sequence_length - 1])
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(np.array(sequences)).to(self.device)
        
        if labels is not None:
            y_tensor = torch.LongTensor(np.array(sequence_labels)).to(self.device)
            return X_tensor, y_tensor
        else:
            return X_tensor
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the CNN model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            patience: Early stopping patience
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        logger.info("Starting CNN model training")
        
        # Create model
        self.model = self._create_model()
        
        # Prepare training data
        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train)
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
            use_validation = True
        else:
            use_validation = False
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_seq, y_train_seq)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if use_validation:
            val_dataset = TensorDataset(X_val_seq, y_val_seq)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Calculate training metrics
            train_loss_avg = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            self.training_history['train_loss'].append(train_loss_avg)
            self.training_history['train_acc'].append(train_acc)
            
            # Validation phase
            if use_validation:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                
                val_loss_avg = val_loss / len(val_loader)
                val_acc = val_correct / val_total
                
                self.training_history['val_loss'].append(val_loss_avg)
                self.training_history['val_acc'].append(val_acc)
                
                # Learning rate scheduling
                scheduler.step(val_loss_avg)
                
                # Early stopping
                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{epochs}] - "
                        f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f}, "
                        f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}"
                    )
                
                # Early stopping check
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{epochs}] - "
                        f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f}"
                    )
        
        # Load best model if validation was used
        if use_validation and hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        self.is_trained = True
        logger.info("CNN model training completed")
        
        return self.training_history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        # Prepare sequences
        X_seq = self._prepare_sequences(X)
        
        predictions = []
        with torch.no_grad():
            # Process in batches to handle memory efficiently
            batch_size = 64
            for i in range(0, len(X_seq), batch_size):
                batch = X_seq[i:i + batch_size]
                outputs = self.model(batch)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        # Prepare sequences
        X_seq = self._prepare_sequences(X)
        
        probabilities = []
        with torch.no_grad():
            # Process in batches
            batch_size = 64
            for i in range(0, len(X_seq), batch_size):
                batch = X_seq[i:i + batch_size]
                outputs = self.model(batch)
                probs = F.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_samples': len(y_test)
        }
        
        logger.info(f"CNN Model evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def save_model(self, model_path: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model components
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'input_features': self.input_features,
            'sequence_length': self.sequence_length,
            'scaler': self.scaler,
            'training_history': self.training_history,
            'device': str(self.device)
        }
        
        torch.save(save_dict, model_path)
        logger.info(f"CNN model saved to {model_path}")
    
    @classmethod
    def load_model(cls, model_path: str) -> 'CNNModel':
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to load the model from
            
        Returns:
            Loaded CNNModel instance
        """
        # Load saved components
        save_dict = torch.load(model_path, map_location='cpu')
        
        # Create model instance
        model_instance = cls(
            input_features=save_dict['input_features'],
            sequence_length=save_dict['sequence_length'],
            model_config=save_dict['model_config']
        )
        
        # Load components
        model_instance.scaler = save_dict['scaler']
        model_instance.training_history = save_dict['training_history']
        model_instance.is_trained = True
        
        # Create and load model
        model_instance.model = model_instance._create_model()
        model_instance.model.load_state_dict(save_dict['model_state_dict'])
        model_instance.model.eval()
        
        logger.info(f"CNN model loaded from {model_path}")
        
        return model_instance
    
    def export_to_onnx(self, onnx_path: str, sample_input: Optional[torch.Tensor] = None) -> None:
        """
        Export model to ONNX format for deployment.
        
        Args:
            onnx_path: Path to save ONNX model
            sample_input: Sample input tensor for tracing
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before exporting")
        
        self.model.eval()
        
        # Create sample input if not provided
        if sample_input is None:
            sample_input = torch.randn(
                1, self.sequence_length, self.input_features,
                device=self.device
            )
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            sample_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"CNN model exported to ONNX format: {onnx_path}")
    
    def get_model_info(self) -> Dict:
        """Get information about the model architecture and parameters."""
        if self.model is None:
            return {"status": "Model not initialized"}
        
        info = {
            "model_type": "CNN (1D Convolutional Neural Network)",
            "input_features": self.input_features,
            "sequence_length": self.sequence_length,
            "total_parameters": self.model._count_parameters(),
            "device": str(self.device),
            "is_trained": self.is_trained
        }
        
        if self.is_trained:
            info["training_epochs"] = len(self.training_history['train_loss'])
            if self.training_history['val_acc']:
                info["best_val_accuracy"] = max(self.training_history['val_acc'])
        
        return info
