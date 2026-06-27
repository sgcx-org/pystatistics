"""
Multivariate analysis module.

Provides principal component analysis and factor analysis,
matching R's implementations.

Public API:
    pca(X)                - Principal Component Analysis (matches prcomp)
    factor_analysis(X)    - Maximum Likelihood Factor Analysis (matches factanal)
"""

from pystatistics.multivariate._pca import pca
from pystatistics.multivariate._factor import factor_analysis
from pystatistics.multivariate._common import (
    PCASolution, FactorSolution, PCAParams, FactorParams,
)

__all__ = [
    "pca",
    "factor_analysis",
    "PCASolution",
    "FactorSolution",
    "PCAParams",
    "FactorParams",
]
