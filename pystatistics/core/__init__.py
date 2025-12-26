"""
Core infrastructure for PyStatistics.
"""

from pystatistics.core.datasource import DataSource
from pystatistics.core.result import Result
from pystatistics.core.capabilities import (
    CAPABILITY_MATERIALIZED,
    CAPABILITY_STREAMING,
    CAPABILITY_GPU_NATIVE,
    CAPABILITY_REPEATABLE,
    CAPABILITY_SUFFICIENT_STATISTICS,
)
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
    # Capabilities
    "CAPABILITY_MATERIALIZED",
    "CAPABILITY_STREAMING",
    "CAPABILITY_GPU_NATIVE",
    "CAPABILITY_REPEATABLE",
    "CAPABILITY_SUFFICIENT_STATISTICS",
    # Exceptions
    "PyStatisticsError",
    "ValidationError",
    "DimensionError",
    "NumericalError",
    "SingularMatrixError",
    "NotPositiveDefiniteError",
    "ConvergenceError",
]
