"""
Descriptive statistics module.

Provides comprehensive descriptive statistics matching R's implementations,
with optional GPU acceleration for large datasets.

Public API:
    describe(data)  - All statistics at once
    cor(x)          - Correlation matrix (Pearson, Spearman, Kendall)
    cov(x)          - Covariance matrix (Bessel-corrected)
    var(x)          - Variance (Bessel-corrected)
    quantile(x)     - Quantiles (all 9 R types)
    summary(x)      - Six-number summary (Min, Q1, Median, Mean, Q3, Max)
"""

from pystatistics.descriptive.design import DescriptiveDesign
from pystatistics.descriptive.solution import DescriptiveParams, DescriptiveSolution
from pystatistics.descriptive.solvers import (
    describe,
    cor,
    cov,
    var,
    quantile,
    summary,
)

__all__ = [
    "describe",
    "cor",
    "cov",
    "var",
    "quantile",
    "summary",
    "DescriptiveDesign",
    "DescriptiveParams",
    "DescriptiveSolution",
]
