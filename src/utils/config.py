#!/usr/bin/env python3
"""
Configuration management for the traffic classification system.
"""

import json
import os
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self._config = self._load_default_config()
        
        if config_file and os.path.exists(config_file):
            self._load_config_file(config_file)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "data_dir": "./data",
            "models_dir": "./models",
            "log_level": "INFO",
            "capture": {
                "interface": "auto",
                "buffer_size": 10000,
                "max_packets_per_second": 10000,
                "timeout": 1.0
            },
            "models": {
                "xgboost": {
                    "n_estimators": 200,
                    "max_depth": 8,
                    "learning_rate": 0.1
                },
                "cnn": {
                    "input_size": 100,
                    "hidden_sizes": [64, 32],
                    "num_classes": 2,
                    "dropout": 0.3
                },
                "ensemble": {
                    "method": "weighted",
                    "weights": {"xgboost": 0.6, "cnn": 0.4}
                }
            },
            "dashboard": {
                "host": "localhost",
                "port": 8000,
                "update_interval": 1.0
            },
            "preprocessing": {
                "chunk_size": 10000,
                "low_memory_mode": False
            }
        }
    
    def _load_config_file(self, config_file: str):
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            self._merge_config(file_config)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing."""
        def merge_dict(base: Dict[str, Any], update: Dict[str, Any]):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(self._config, new_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'models.xgboost.n_estimators')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, config_file: Optional[str] = None):
        """
        Save configuration to file.
        
        Args:
            config_file: File path to save to (uses initialized file if None)
        """
        file_path = config_file or self.config_file
        if not file_path:
            raise ValueError("No config file specified")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()


# Global configuration instance
_global_config = None


def get_config(config_file: Optional[str] = None) -> Config:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config(config_file)
    return _global_config


def set_config(config: Config):
    """Set global configuration instance."""
    global _global_config
    _global_config = config
