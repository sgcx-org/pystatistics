"""
Core infrastructure for PyStatistics.
"""

from pystatistics.core.datasource import DataSource
from pystatistics.core.result import Result
from pystatistics.core.exceptions import (
    PyStatisticsError,
    ValidationError,
    DimensionError,
    NumericalError,
    SingularMatrixError,
    NotPositiveDefiniteError,
    ConvergenceError,
)

__all__ = [
    "DataSource",
    "Result",
    "PyStatisticsError",
    "ValidationError",
    "DimensionError",
    "NumericalError",
    "SingularMatrixError",
    "NotPositiveDefiniteError",
    "ConvergenceError",
]
