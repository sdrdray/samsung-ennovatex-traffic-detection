"""
Real-time Reel/Video Traffic Detection System

A comprehensive AI system for detecting video/reel traffic patterns
in social networking applications using network metadata analysis.
"""

__version__ = "1.0.0"
__author__ = "Samsung EnnovateX 2025 Challenge Team"
__license__ = "Apache-2.0"

from . import data_collection
from . import features
from . import models
from . import inference
from . import utils

__all__ = [
    "data_collection",
    "features", 
    "models",
    "inference",
    "utils",
]
