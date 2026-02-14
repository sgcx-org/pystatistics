"""
Missing data handling for descriptive statistics.

Implements R-compatible missing data policies:
- 'everything': propagate NaN (R default)
- 'complete.obs': listwise deletion (remove rows with any NaN)
- 'pairwise.complete.obs': per-(i,j) pair, use only shared non-NaN rows
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ValidationError


def apply_use_policy(
    data: NDArray,
    use: str,
) -> tuple[NDArray, int]:
    """
    Apply missing data policy for column-wise statistics (mean, var, sd, etc.).

    Parameters
    ----------
    data : NDArray
        (n, p) data matrix, may contain NaN.
    use : str
        'everything', 'complete.obs', or 'pairwise.complete.obs'.

    Returns
    -------
    clean_data : NDArray
        Data after applying policy. For 'everything' and 'pairwise.complete.obs',
        returns the original data unchanged (NaN handling is per-operation).
        For 'complete.obs', returns data with NaN-containing rows removed.
    n_complete : int
        Number of complete (non-NaN-containing) rows.
    """
    if use == 'everything':
        n_complete = int(np.sum(~np.any(np.isnan(data), axis=1)))
        return data, n_complete

    elif use == 'complete.obs':
        complete_mask = ~np.any(np.isnan(data), axis=1)
        n_complete = int(np.sum(complete_mask))
        if n_complete < 1:
            raise ValidationError(
                "No complete observations (all rows contain NaN). "
                "Consider using use='pairwise.complete.obs'."
            )
        return data[complete_mask], n_complete

    elif use == 'pairwise.complete.obs':
        # For column-wise statistics, pairwise behaves like 'everything'
        # (each column uses its own non-NaN values).
        # Pairwise logic is only different for bivariate operations (cov, cor).
        n_complete = int(np.sum(~np.any(np.isnan(data), axis=1)))
        return data, n_complete

    else:
        raise ValidationError(
            f"Invalid use= parameter: {use!r}. "
            f"Must be 'everything', 'complete.obs', or 'pairwise.complete.obs'."
        )


def pairwise_mask(xi: NDArray, xj: NDArray) -> NDArray:
    """
    Boolean mask where both columns are non-NaN.

    Parameters
    ----------
    xi, xj : NDArray
        1D arrays of the same length.

    Returns
    -------
    mask : NDArray[bool]
        True where both xi and xj are non-NaN.
    """
    return ~(np.isnan(xi) | np.isnan(xj))


def columnwise_clean(col: NDArray) -> NDArray:
    """
    Remove NaN values from a 1D array.

    Parameters
    ----------
    col : NDArray
        1D array, may contain NaN.

    Returns
    -------
    clean : NDArray
        1D array with NaN removed.
    """
    return col[~np.isnan(col)]
