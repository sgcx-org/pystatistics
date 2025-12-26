"""
Core infrastructure for PyStatistics.

This module provides shared abstractions, utilities, and backend infrastructure
used by all domain-specific submodules (regression, mvnmle, survival, etc.).

Key components:
    protocols: DataSource, Backend protocols
    result: Generic Result[P] envelope
    exceptions: Exception hierarchy
    validation: Input validators
    backends: Hardware detection, timing, linear algebra primitives
"""

from pystatistics.core.protocols import DataSource, Backend
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
    # Protocols
    "DataSource",
    "Backend",
    # Result
    "Result",
    # Exceptions
    "PyStatisticsError",
    "ValidationError",
    "DimensionError",
    "NumericalError",
    "SingularMatrixError",
    "NotPositiveDefiniteError",
    "ConvergenceError",
]
