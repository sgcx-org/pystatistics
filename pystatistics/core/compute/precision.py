"""
Numerical precision constants and utilities.

Provides machine epsilon, safe numerical operations, and precision-related
utilities used across all backends.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Any


# Machine epsilon for float64
EPSILON_64: float = np.finfo(np.float64).eps  # ~2.22e-16

# Machine epsilon for float32
EPSILON_32: float = np.finfo(np.float32).eps  # ~1.19e-7

# Default tolerance for numerical comparisons (relative)
DEFAULT_RTOL: float = 1e-12

# Default tolerance for considering values as zero (absolute)
DEFAULT_ATOL: float = 1e-14


def machine_epsilon(dtype: np.dtype | type = np.float64) -> float:
    """
    Get machine epsilon for a given dtype.
    
    Args:
        dtype: NumPy dtype or type
        
    Returns:
        Machine epsilon for the dtype
    """
    return float(np.finfo(dtype).eps)


def safe_divide(
    numerator: NDArray[np.floating[Any]], 
    denominator: NDArray[np.floating[Any]],
    fill_value: float = 0.0
) -> NDArray[np.floating[Any]]:
    """
    Division with protection against divide-by-zero.
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        fill_value: Value to use where denominator is zero
        
    Returns:
        Result of division with fill_value where denominator is zero
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
        result = np.where(np.isfinite(result), result, fill_value)
    return result


def safe_log(
    x: NDArray[np.floating[Any]], 
    floor: float = 1e-300
) -> NDArray[np.floating[Any]]:
    """
    Natural logarithm with protection against log(0).
    
    Args:
        x: Input array
        floor: Minimum value to use (prevents -inf)
        
    Returns:
        log(max(x, floor))
    """
    return np.log(np.maximum(x, floor))


def is_close(
    a: float | NDArray[np.floating[Any]], 
    b: float | NDArray[np.floating[Any]], 
    rtol: float = DEFAULT_RTOL, 
    atol: float = DEFAULT_ATOL
) -> bool | NDArray[np.bool_]:
    """
    Check if values are numerically close.
    
    Uses the formula: |a - b| <= atol + rtol * |b|
    
    Args:
        a: First value(s)
        b: Second value(s)
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Boolean or boolean array indicating closeness
    """
    return np.abs(a - b) <= atol + rtol * np.abs(b)


def condition_number(A: NDArray[np.floating[Any]]) -> float:
    """
    Compute condition number of a matrix using SVD.
    
    Args:
        A: Input matrix
        
    Returns:
        Condition number (ratio of largest to smallest singular value)
        Returns inf if matrix is singular.
    """
    s = np.linalg.svd(A, compute_uv=False)
    if s[-1] == 0:
        return np.inf
    return float(s[0] / s[-1])
