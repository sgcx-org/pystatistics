"""
Exception hierarchy for PyStatistics.

All exceptions inherit from PyStatisticsError to allow catching any
library-specific error. Domain-specific exceptions should inherit from
the appropriate base class here.

Design principles:
    - Exceptions carry diagnostic information as attributes
    - Error messages are actionable with actual vs expected values
    - Never catch and re-raise with less information
"""


class PyStatisticsError(Exception):
    """Base exception for all PyStatistics errors."""
    pass


class ValidationError(PyStatisticsError):
    """
    Input validation failed.
    
    Raised when user-provided inputs fail validation checks.
    """
    pass


class DimensionError(ValidationError):
    """
    Array dimensions are incorrect or inconsistent.
    
    Raised when array shapes don't match expected dimensions or
    when multiple arrays have inconsistent shapes.
    """
    pass


class NumericalError(PyStatisticsError):
    """
    Numerical computation failed.
    
    Base class for errors arising from numerical issues during computation.
    """
    pass


class SingularMatrixError(NumericalError):
    """
    Matrix is singular or nearly singular.
    
    Raised when a matrix operation requires invertibility but the matrix
    is singular or numerically rank-deficient.
    
    Attributes:
        matrix_name: Name/description of the problematic matrix
        condition_number: Estimated condition number, if available
        rank: Numerical rank, if computed
        expected_rank: Expected rank (typically min(n, p))
    """
    
    def __init__(
        self, 
        message: str,
        matrix_name: str | None = None,
        condition_number: float | None = None,
        rank: int | None = None,
        expected_rank: int | None = None
    ):
        super().__init__(message)
        self.matrix_name = matrix_name
        self.condition_number = condition_number
        self.rank = rank
        self.expected_rank = expected_rank


class NotPositiveDefiniteError(NumericalError):
    """
    Matrix is not positive definite.
    
    Raised when an operation requires a positive definite matrix
    (e.g., Cholesky decomposition) but the matrix fails this requirement.
    
    Attributes:
        matrix_name: Name/description of the problematic matrix
        min_eigenvalue: Minimum eigenvalue, if computed
    """
    
    def __init__(
        self, 
        message: str,
        matrix_name: str | None = None,
        min_eigenvalue: float | None = None
    ):
        super().__init__(message)
        self.matrix_name = matrix_name
        self.min_eigenvalue = min_eigenvalue


class ConvergenceError(PyStatisticsError):
    """
    Iterative algorithm failed to converge.
    
    Raised when an iterative optimization method (EM, Newton-Raphson, IRLS)
    fails to meet convergence criteria within the maximum number of iterations.
    
    Attributes:
        iterations: Number of iterations completed
        final_change: Final parameter or objective change
        reason: Why convergence failed (e.g., 'max_iterations', 'diverging')
        threshold: The convergence threshold that was not met
    """
    
    def __init__(
        self, 
        message: str, 
        iterations: int, 
        final_change: float | None = None,
        reason: str | None = None,
        threshold: float | None = None
    ):
        super().__init__(message)
        self.iterations = iterations
        self.final_change = final_change
        self.reason = reason
        self.threshold = threshold
