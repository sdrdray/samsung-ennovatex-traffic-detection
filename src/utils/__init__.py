"""
Utility modules for the traffic detection system.
"""

from .logging import setup_logger
from .network import get_interface_info, is_interface_available

__all__ = [
    "setup_logger",
    "get_interface_info", 
    "is_interface_available",
]
