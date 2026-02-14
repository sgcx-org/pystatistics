"""
All 9 R quantile type algorithms.

Implements the Hyndman & Fan (1996) quantile definitions exactly as
R's quantile() function does.

Types 1-3 are discontinuous (step functions).
Types 4-9 are continuous (linear interpolation with varying definitions of
the plotting position p(k)).

Reference:
    Hyndman, R.J. and Fan, Y. (1996) "Sample Quantiles in Statistical
    Packages", The American Statistician, 50(4), 361-365.

R source: src/library/stats/R/quantile.R
"""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ValidationError


def r_quantile(x: NDArray, probs: NDArray, qtype: int) -> NDArray:
    """
    Compute quantiles matching R's quantile() exactly.

    Parameters
    ----------
    x : NDArray
        1D sorted array with no NaN values.
    probs : NDArray
        1D array of probabilities in [0, 1].
    qtype : int
        R quantile type 1-9.

    Returns
    -------
    NDArray
        Quantile values, one per probability.
    """
    if qtype not in range(1, 10):
        raise ValidationError(f"Quantile type must be 1-9, got {qtype}")

    n = len(x)
    if n == 0:
        return np.full(len(probs), np.nan)
    if n == 1:
        return np.full(len(probs), x[0])

    probs = np.asarray(probs, dtype=np.float64)
    result = np.empty(len(probs), dtype=np.float64)

    # Implementation following R's quantile.default exactly.
    #
    # R pads the sorted vector:  xs = c(x[1], x, x[n])  (length n+2)
    # Then computes index j and weight h, and interpolates:
    #   qs = (1-h)*xs[j+1] + h*xs[j+2]    (R 1-indexed)
    #
    # In 0-indexed Python: xs[j] and xs[j+1], where
    #   xs[k] maps to x[clamp(k-1, 0, n-1)] in the original array.

    # R fuzz factor: 4 * machine epsilon
    fuzz = 4.0 * np.finfo(np.float64).eps

    if qtype <= 3:
        # --- Discontinuous types ---
        for i, p in enumerate(probs):
            if qtype == 3:
                nppm = n * p - 0.5
            else:
                nppm = n * p  # types 1 and 2

            j = int(math.floor(nppm + fuzz))

            # h (interpolation weight) depends on type
            if qtype == 1:
                h = 1.0 if (nppm > j + fuzz) else 0.0
            elif qtype == 2:
                h = 0.5 if abs(nppm - j) < fuzz else (1.0 if nppm > j else 0.0)
            elif qtype == 3:
                # R: (nppm != j) | ((j %% 2L) == 1L)
                # h=1 unless nppm==j AND j is even
                nppm_eq_j = abs(nppm - j) < fuzz
                if nppm_eq_j and (j % 2 == 0):
                    h = 0.0
                else:
                    h = 1.0

            # R pads x: xs = [x[0], x[0], x[1], ..., x[n-1], x[n-1]]
            # xs is length n+2 (0-indexed: xs[0]=x[0], xs[1]=x[0], xs[2]=x[1], ...)
            # R uses xs[j+1] and xs[j+2] (1-indexed) = xs[j] and xs[j+1] (0-indexed)
            # xs[k] in 0-indexed maps to x[clamp(k-1, 0, n-1)]
            lo = max(0, min(j - 1, n - 1))
            hi = max(0, min(j, n - 1))
            result[i] = (1.0 - h) * x[lo] + h * x[hi]

    else:
        # --- Continuous types 4-9 ---
        # R: nppm = a + p * (n + 1 - a - b)
        if qtype == 4:
            a, b = 0.0, 1.0
        elif qtype == 5:
            a, b = 0.5, 0.5
        elif qtype == 6:
            a, b = 0.0, 0.0
        elif qtype == 7:
            a, b = 1.0, 1.0
        elif qtype == 8:
            a, b = 1.0 / 3.0, 1.0 / 3.0
        elif qtype == 9:
            a, b = 3.0 / 8.0, 3.0 / 8.0

        for i, p in enumerate(probs):
            nppm = a + p * (n + 1.0 - a - b)
            j = int(math.floor(nppm + fuzz))
            h = nppm - j

            # Small negative h from floating point → clamp to 0
            if abs(h) < fuzz:
                h = 0.0
            elif abs(h - 1.0) < fuzz:
                h = 1.0

            # Clamp j to valid range for 0-indexed x
            # j maps to x[j-1] (since nppm is 1-indexed)
            if j < 1:
                result[i] = x[0]
            elif j >= n:
                result[i] = x[n - 1]
            else:
                result[i] = (1.0 - h) * x[j - 1] + h * x[j]

    return result
