"""
TurbTwin: Physics-Informed Machine Learning Digital Twin
for Thermal Turbulent Jet Temperature Prediction

This package provides an ensemble learning framework combining
XGBoost, LightGBM, Random Forest, and Physics-Informed Neural Networks
for accurate temperature prediction with uncertainty quantification.
"""

__version__ = "1.0.0"
__author__ = "Narjisse Kabbaj, Naila Marir, Mohamed F. El-Amin"

from turbtwin.ensemble.robust_ensemble import TurbTwinEnsemble
from turbtwin.data.preprocessor import DataPreprocessor

__all__ = [
    "TurbTwinEnsemble",
    "DataPreprocessor",
    "__version__",
]
