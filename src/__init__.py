"""
DataProcessing - Antarctic turbulence and flux data processing toolkit
"""
__version__ = "0.1.0"

# Make commonly used functions available at package level
from .ec.func_read_data import read_eddypro_data, read_data
from .utils.utils import resample_with_threshold
from .utils.constants import *