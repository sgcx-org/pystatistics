"""
Input validation utilities for PyStatistics.

These validators follow the "fail fast, fail loud" principle. They raise
immediately with clear error messages rather than silently correcting
or making assumptions about user intent.

Design principles:
    - No silent type coercion (except np.asarray on array-likes)
    - No default handling of edge cases
    - Clear, actionable error messages with actual values
    - Each function validates ONE thing
    - Parameter names included in all error messages
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any

from pystatistics.core.exceptions import ValidationError, DimensionError


def check_array(
    array: ArrayLike,
    name: str,
) -> NDArray[np.floating[Any]]:
    """
    Validate and convert input to numpy array.
    
    Accepts any array-like and converts to numpy array. Rejects inputs
    that result in object dtype (indicating mixed types or non-numeric data).
    
    Args:
        array: Input to validate
        name: Parameter name for error messages
        
    Returns:
        numpy.ndarray with numeric dtype
        
    Raises:
        ValidationError: If input cannot be converted to numeric array
    """
    try:
        result = np.asarray(array)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"{name}: cannot convert to array: {e}") from e
    
    if result.dtype == object:
        raise ValidationError(
            f"{name}: converted to object dtype, indicating mixed types or non-numeric data"
        )

    # Reject non-numeric dtypes (strings, bytes, datetime, etc.)
    if not np.issubdtype(result.dtype, np.number):
        raise ValidationError(
            f"{name}: non-numeric dtype {result.dtype}, expected numeric data"
        )

    # Ensure floating point for numerical stability
    if not np.issubdtype(result.dtype, np.floating):
        result = result.astype(np.float64)

    return result


def check_finite(array: NDArray[np.floating[Any]], name: str) -> None:
    """
    Verify array contains no NaN or Inf values.
    
    Args:
        array: Array to check
        name: Parameter name for error messages
        
    Raises:
        ValidationError: If array contains non-finite values
    """
    if not np.all(np.isfinite(array)):
        n_nan = int(np.sum(np.isnan(array)))
        n_inf = int(np.sum(np.isinf(array)))
        raise ValidationError(
            f"{name}: contains non-finite values ({n_nan} NaN, {n_inf} Inf)"
        )


def check_ndim(array: NDArray[np.floating[Any]], ndim: int, name: str) -> None:
    """
    Verify array has exactly the specified number of dimensions.
    
    Args:
        array: Array to check
        ndim: Required number of dimensions
        name: Parameter name for error messages
        
    Raises:
        DimensionError: If array has wrong number of dimensions
    """
    if array.ndim != ndim:
        raise DimensionError(
            f"{name}: expected {ndim}D array, got {array.ndim}D with shape {array.shape}"
        )


def check_1d(array: NDArray[np.floating[Any]], name: str) -> None:
    """
    Verify array is 1-dimensional.
    
    Args:
        array: Array to check
        name: Parameter name for error messages
        
    Raises:
        DimensionError: If array is not 1D
    """
    check_ndim(array, 1, name)


def check_2d(array: NDArray[np.floating[Any]], name: str) -> None:
    """
    Verify array is 2-dimensional.
    
    Args:
        array: Array to check
        name: Parameter name for error messages
        
    Raises:
        DimensionError: If array is not 2D
    """
    check_ndim(array, 2, name)


def check_consistent_length(
    *arrays: NDArray[np.floating[Any]], 
    names: tuple[str, ...]
) -> None:
    """
    Verify all arrays have the same length (first dimension).
    
    Args:
        *arrays: Arrays to check
        names: Parameter names for error messages (must match number of arrays)
        
    Raises:
        ValueError: If number of names doesn't match number of arrays
        DimensionError: If arrays have inconsistent lengths
    """
    if len(arrays) != len(names):
        raise ValueError(
            f"Number of arrays ({len(arrays)}) must match number of names ({len(names)})"
        )
    
    if len(arrays) < 2:
        return
    
    lengths = [arr.shape[0] for arr in arrays]
    if len(set(lengths)) > 1:
        details = ", ".join(f"{name}={length}" for name, length in zip(names, lengths))
        raise DimensionError(f"Inconsistent lengths: {details}")


def check_min_samples(array: NDArray[np.floating[Any]], min_samples: int, name: str) -> None:
    """
    Verify array has at least the minimum number of samples.
    
    Args:
        array: Array to check
        min_samples: Minimum required samples (first dimension)
        name: Parameter name for error messages
        
    Raises:
        ValidationError: If array has fewer than min_samples
    """
    n = array.shape[0]
    if n < min_samples:
        raise ValidationError(
            f"{name}: requires at least {min_samples} samples, got {n}"
        )


def check_no_zero_variance_columns(X: NDArray[np.floating[Any]], name: str) -> None:
    """
    Verify matrix has no constant (zero-variance) columns.
    
    Constant columns cause singularity in X'X and typically indicate
    data problems or redundant features.
    
    Args:
        X: 2D array to check
        name: Parameter name for error messages
        
    Raises:
        ValidationError: If any column has zero variance
    """
    variances = np.var(X, axis=0)
    zero_var_cols = np.where(variances == 0)[0]
    
    if len(zero_var_cols) > 0:
        raise ValidationError(
            f"{name}: columns {zero_var_cols.tolist()} have zero variance (constant)"
        )


def check_column_rank(X: NDArray[np.floating[Any]], name: str) -> None:
    """
    Verify matrix has full column rank.
    
    A rank-deficient design matrix indicates perfect multicollinearity
    and will cause OLS to fail.
    
    Args:
        X: 2D array to check
        name: Parameter name for error messages
        
    Raises:
        ValidationError: If matrix is rank-deficient
    """
    n, p = X.shape
    rank = np.linalg.matrix_rank(X)
    expected_rank = min(n, p)
    
    if rank < p:
        raise ValidationError(
            f"{name}: rank-deficient (rank={rank}, expected={expected_rank}). "
            f"This indicates perfect multicollinearity."
        )
