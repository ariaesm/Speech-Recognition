"""Offline keyword spotting: MFCC features from scratch + DTW template matching."""

from .dtw import dtw
from .mfcc import mfcc
from .recognizer import TemplateRecognizer

__all__ = ["TemplateRecognizer", "dtw", "mfcc"]
__version__ = "0.1.0"
