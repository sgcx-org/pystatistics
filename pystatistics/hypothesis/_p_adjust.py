"""
Multiple testing correction matching R's p.adjust().

Implements all 8 methods: holm, hochberg, hommel, bonferroni, BH, BY, fdr, none.

This is a standalone utility function (no Design/Backend pipeline).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray, ArrayLike

from pystatistics.core.exceptions import ValidationError

VALID_METHODS = (
    "holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none"
)


def p_adjust(
    p: ArrayLike,
    method: str = "holm",
    n: int | None = None,
) -> NDArray[np.floating]:
    """
    Adjust p-values for multiple comparisons. Matches R p.adjust().

    Parameters
    ----------
    p : array-like
        Vector of p-values.
    method : str
        Adjustment method. One of: "holm" (default), "hochberg", "hommel",
        "bonferroni", "BH", "BY", "fdr" (alias for BH), "none".
    n : int or None
        Number of comparisons. Default len(p). Can be larger than len(p)
        when some p-values are omitted.

    Returns
    -------
    ndarray
        Adjusted p-values, same length as input. Clipped to [0, 1].
        NaN positions in input produce NaN in output.
    """
    if method not in VALID_METHODS:
        raise ValidationError(
            f"method must be one of {VALID_METHODS}, got {method!r}"
        )

    p_arr = np.asarray(p, dtype=np.float64).ravel()

    if n is None:
        n_tests = len(p_arr)
    else:
        if n < len(p_arr):
            raise ValidationError(
                f"n ({n}) must be >= length of p ({len(p_arr)})"
            )
        n_tests = n

    if len(p_arr) == 0:
        return np.array([], dtype=np.float64)

    # Handle NaN: work only with non-NaN values
    nan_mask = np.isnan(p_arr)
    if np.all(nan_mask):
        return p_arr.copy()

    result = p_arr.copy()

    if method == "none":
        return result

    # Get indices of non-NaN values
    valid_idx = np.where(~nan_mask)[0]
    pv = p_arr[valid_idx]
    lp = len(pv)

    if method == "bonferroni":
        adjusted = np.minimum(pv * n_tests, 1.0)

    elif method == "holm":
        adjusted = _holm(pv, n_tests)

    elif method == "hochberg":
        adjusted = _hochberg(pv, n_tests)

    elif method == "hommel":
        adjusted = _hommel(pv, n_tests)

    elif method in ("BH", "fdr"):
        adjusted = _bh(pv, n_tests)

    elif method == "BY":
        adjusted = _by(pv, n_tests)

    else:
        raise ValidationError(f"Unknown method: {method!r}")

    result[valid_idx] = np.clip(adjusted, 0.0, 1.0)
    return result


def _holm(pv: NDArray, n: int) -> NDArray:
    """Holm's step-down method (controls FWER, no assumptions)."""
    lp = len(pv)
    order = np.argsort(pv)
    sorted_p = pv[order]

    # Multiply by (n - rank + 1) where rank is 1-based
    adjusted_sorted = sorted_p * np.arange(n, n - lp, -1, dtype=np.float64)

    # Enforce monotonicity (cumulative max)
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)

    # Unsort
    result = np.empty(lp, dtype=np.float64)
    result[order] = adjusted_sorted
    return result


def _hochberg(pv: NDArray, n: int) -> NDArray:
    """Hochberg's step-up method (controls FWER, needs independence/PRDS)."""
    lp = len(pv)
    order = np.argsort(pv)[::-1]  # descending
    sorted_p = pv[order]

    # Multiply by (n - rank + 1) where rank counts from top
    multipliers = np.arange(n - lp + 1, n + 1, dtype=np.float64)
    adjusted_sorted = sorted_p * multipliers

    # Enforce monotonicity (cumulative min, since we're going descending)
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted)

    # Unsort
    result = np.empty(lp, dtype=np.float64)
    result[order] = adjusted_sorted
    return result


def _hommel(pv: NDArray, n: int) -> NDArray:
    """
    Hommel's method (controls FWER, needs independence/PRDS).

    Translated directly from R's actual p.adjust source (stats::p.adjust):

        if (n > lp) p <- c(p, rep.int(1, n - lp))
        i <- seq_len(n)
        o <- order(p)
        p <- p[o]
        ro <- order(o)
        q <- pa <- rep.int(min(n * p/i), n)
        for (j in (n - 1L):2L) {
            ij <- seq_len(n - j + 1L)
            i2 <- (n - j + 2L):n
            q1 <- min(j * p[i2]/(2L:j))
            q[ij] <- pmin(j * p[ij], q1)
            q[i2] <- q[n - j + 1L]
            pa <- pmax(pa, q)
        }
        pmax(pa, p)[if (lp < n) ro[1L:lp] else ro]
    """
    lp = len(pv)
    if lp <= 1:
        return pv.copy()

    # If n > lp, pad with 1s (R: if (n > lp) p <- c(p, rep.int(1, n - lp)))
    if n > lp:
        padded = np.ones(n, dtype=np.float64)
        padded[:lp] = pv
        work_p = padded
    else:
        work_p = pv.copy()

    nn = len(work_p)

    # Sort ascending
    o = np.argsort(work_p)
    sp = work_p[o].copy()
    ro = np.argsort(o)  # inverse permutation

    # i = 1..n
    i = np.arange(1, nn + 1, dtype=np.float64)

    # q <- pa <- rep(min(n * p / i), n)
    init_val = np.min(nn * sp / i)
    pa = np.full(nn, init_val, dtype=np.float64)
    q = np.full(nn, init_val, dtype=np.float64)

    # for (j in (n-1):2)
    for j in range(nn - 1, 1, -1):
        # R 1-based: ij = 1:(n-j+1), i2 = (n-j+2):n
        # 0-based:   ij = 0:(n-j),    i2 = (n-j+1):(n-1)
        ij_end = nn - j + 1       # exclusive end for ij (0-based)
        i2_start = nn - j + 1     # start of i2 (0-based)

        divisors = np.arange(2, j + 1, dtype=np.float64)  # R: 2:j
        p_i2 = sp[i2_start:]

        # q1 <- min(j * p[i2] / (2:j))
        q1 = np.min(j * p_i2 / divisors)

        # q[ij] <- pmin(j * p[ij], q1)
        ij_indices = np.arange(1, ij_end + 1, dtype=np.float64)  # R: 1-based ij
        q[:ij_end] = np.minimum(j * sp[:ij_end], q1)

        # q[i2] <- q[n - j + 1]  (R 1-based index n-j+1 = 0-based index n-j)
        boundary_idx = nn - j  # 0-based
        q[i2_start:] = q[boundary_idx]

        # pa <- pmax(pa, q)
        pa = np.maximum(pa, q)

    # result = pmax(pa, p)
    combined = np.maximum(pa, sp)

    # Unsort
    if lp < n:
        result = combined[ro[:lp]]
    else:
        result = combined[ro]

    return result


def _bh(pv: NDArray, n: int) -> NDArray:
    """Benjamini-Hochberg (controls FDR, needs independence/PRDS)."""
    lp = len(pv)
    order = np.argsort(pv)[::-1]  # descending
    sorted_p = pv[order]

    # p * n / rank (rank from 1 to lp, but we're descending so adjust)
    ranks = np.arange(lp, 0, -1, dtype=np.float64)
    adjusted_sorted = sorted_p * n / ranks

    # Enforce monotonicity (cumulative min going from largest to smallest)
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted)

    # Unsort
    result = np.empty(lp, dtype=np.float64)
    result[order] = adjusted_sorted
    return result


def _by(pv: NDArray, n: int) -> NDArray:
    """Benjamini-Yekutieli (controls FDR under arbitrary dependence)."""
    lp = len(pv)
    # Correction factor: c(m) = sum(1/i for i in 1..n)
    cm = np.sum(1.0 / np.arange(1, n + 1, dtype=np.float64))

    order = np.argsort(pv)[::-1]  # descending
    sorted_p = pv[order]

    ranks = np.arange(lp, 0, -1, dtype=np.float64)
    adjusted_sorted = sorted_p * cm * n / ranks

    # Enforce monotonicity
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted)

    # Unsort
    result = np.empty(lp, dtype=np.float64)
    result[order] = adjusted_sorted
    return result
