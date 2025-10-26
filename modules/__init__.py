"""
modules/__init__.py
Module initialization file
"""

from .bin_packing import BinPacking3D, Warehouse, Box, Position
from .visualizer import Visualizer3D
from .optimizer import WarehouseOptimizer
from .ai_object_detector import AIObjectDetector

__all__ = [
    "ImageProcessor",
    "BinPacking3D",
    "Warehouse",
    "Box",
    "Position",
    "Visualizer3D",
    "WarehouseOptimizer",
    "AIObjectDetector",
]

__version__ = "1.0.0"
__author__ = "Warehouse Optimization Team"
