"""
Linear and generalized linear models.

This module provides linear regression (OLS) and will eventually include
generalized linear models (GLM) with various link functions.

Public API:
    fit(X, y, ...) -> LinearSolution
    
The fit() function is the only entry point. It handles:
    - Input validation
    - Design construction
    - Backend selection
    - Result wrapping

Example:
    >>> from pystatistics.regression import fit
    >>> result = fit(X, y)
    >>> print(result.coefficients)
    >>> print(result.summary())
"""

from pystatistics.regression.design import RegressionDesign
from pystatistics.regression.solution import LinearSolution, LinearParams
from pystatistics.regression.solvers import fit

__all__ = [
    "fit",
    "RegressionDesign",
    "LinearSolution",
    "LinearParams",
]
