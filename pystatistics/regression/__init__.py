"""
Linear and generalized linear models.

Usage:
    from pystatistics import DataSource
    from pystatistics.regression import Design, fit

    # OLS from DataSource
    ds = DataSource.from_file("data.csv")
    design = Design.from_datasource(ds, y='target')
    result = fit(design)

    # OLS from arrays (convenience)
    result = fit(X, y)

    # GLM (logistic regression)
    result = fit(X, y, family='binomial')

    # GLM (Poisson regression)
    result = fit(X, y, family='poisson')
"""

from pystatistics.core.datasource import DataSource
from pystatistics.regression.design import Design
from pystatistics.regression.solution import (
    LinearSolution, LinearParams,
    GLMSolution, GLMParams,
)
from pystatistics.regression.solvers import fit
from pystatistics.regression.families import (
    Family, Gaussian, Binomial, Poisson,
)

__all__ = [
    "fit",
    "Design",
    "DataSource",
    # LM types
    "LinearSolution",
    "LinearParams",
    # GLM types
    "GLMSolution",
    "GLMParams",
    # GLM families
    "Family",
    "Gaussian",
    "Binomial",
    "Poisson",
]
