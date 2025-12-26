"""
PyStatistics: GPU-accelerated statistical computing for Python.

A unified ecosystem for regulatory-grade statistical analysis with
optional GPU acceleration for revolutionary performance.

Submodules:
    regression: Linear and generalized linear models
    mvnmle: Multivariate normal maximum likelihood estimation
    survival: Survival analysis (future)
    longitudinal: Mixed models (future)
"""

__version__ = "0.1.0"
__author__ = "Hai-Shuo"
__email__ = "contact@sgcx.org"

# Submodule imports (lazy loading in future)
from pystatistics import regression
from pystatistics import mvnmle

__all__ = [
    "__version__",
    "regression",
    "mvnmle",
]
