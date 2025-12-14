"""
JSON Comparator module for PAPI export validation

This module provides comprehensive comparison between reference and generated JSON files
with proper interpolation (without extrapolation) and detailed delta analysis.
"""

from .json_comparator import JSONComparator, compare_json_files
from .base_comparator import BaseComparator
from .interpolator import DataInterpolator
from .delta_analyzer import DeltaAnalyzer
from .units_converter import UnitsConverter
from .segments_analyzer import SegmentsAnalyzer

__all__ = [
    'JSONComparator',
    'compare_json_files',
    'BaseComparator',
    'DataInterpolator',
    'DeltaAnalyzer',
    'UnitsConverter',
    'SegmentsAnalyzer'
]

__version__ = '1.0.0'