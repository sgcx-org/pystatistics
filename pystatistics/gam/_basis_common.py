"""Shared validation for GAM basis constructors.

One job: the input contracts common to every spline basis — predictor sanity
and basis-dimension sanity. Fails loud on violation (Rule 1/2).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ValidationError

_MIN_K = 3


def validate_x(
    x: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Validate and flatten a 1-D predictor array.

    Args:
        x: Input array.

    Returns:
        Flattened float64 array.

    Raises:
        ValidationError: empty input or non-finite values.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.shape[0] == 0:
        raise ValidationError("x must be a non-empty 1-D array")
    if np.any(~np.isfinite(x)):
        raise ValidationError("x contains non-finite values (NaN or Inf)")
    return x


def validate_k(k: int, n_unique: int) -> None:
    """Validate the basis dimension against the number of unique x values.

    mgcv's rule: a rank-k spline basis needs at least k distinct covariate
    values (its error: "fewer unique covariate combinations than specified
    maximum degrees of freedom"). We enforce the same, loudly.

    Args:
        k: Requested basis dimension.
        n_unique: Number of unique predictor values.

    Raises:
        ValidationError: If k is invalid for this predictor.
    """
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValidationError(f"k must be an integer, got {type(k).__name__}")
    if k < _MIN_K:
        raise ValidationError(f"k must be >= {_MIN_K}, got k={k}")
    if k > n_unique:
        raise ValidationError(
            f"k={k} exceeds the number of unique x values ({n_unique}); "
            f"a rank-k spline basis needs at least k distinct values "
            f"(mgcv fails on this too)"
        )
