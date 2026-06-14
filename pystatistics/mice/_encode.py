"""
Predictor encoding and category-code mapping for the chained-equations sweep.

When a column is imputed, the other columns serve as predictors. Numeric columns
enter the model as-is, but a *categorical* predictor must be dummy-encoded — its
integer codes are labels, not magnitudes, so feeding the raw codes would impose a
spurious linear order. This module builds the encoded predictor matrix and maps
categorical target values between their stored codes and the consecutive
``0..K-1`` class indices the categorical methods expect.

Contrast choice: categorical predictors use treatment (drop-first) dummies. For
a model's *predictions* this is equivalent to any other full-rank contrast
(treatment, polynomial, …) — they span the same column space — so this matches
R's model matrix for imputation purposes regardless of which contrast R picks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_predictor_matrix(data: NDArray, j: int, design) -> NDArray:
    """Predictor matrix for imputing column ``j``: every other column, with
    categorical columns treatment-dummy-encoded. No intercept column (methods
    add one where they need it).

    Fast path: when no column is categorical this is a plain column slice, so the
    all-numeric sweep keeps its original cost.
    """
    p = design.p
    if not design.has_categorical:
        cols = [c for c in range(p) if c != j]
        return data[:, cols]

    blocks = []
    for c in range(p):
        if c == j:
            continue
        if design.is_categorical(c):
            blocks.append(_treatment_dummies(data[:, c], design.levels_for(c)))
        else:
            blocks.append(data[:, c].reshape(-1, 1))
    if not blocks:
        return np.empty((data.shape[0], 0), dtype=np.float64)
    return np.hstack(blocks)


def _treatment_dummies(col: NDArray, levels: NDArray) -> NDArray:
    """One-hot encode ``col`` against ``levels``, dropping the first level as the
    reference. Returns (n, len(levels)-1). Values are exact category codes (drawn
    from observed values or imputed back as codes), so equality is exact."""
    # Columns for levels[1:]; the reference level maps to an all-zero row.
    return (col[:, None] == levels[None, 1:]).astype(np.float64)


def codes_to_indices(values: NDArray, levels: NDArray) -> NDArray:
    """Map category codes to consecutive ``0..K-1`` indices (levels are sorted)."""
    return np.searchsorted(levels, values).astype(np.intp)


def indices_to_codes(indices: NDArray, levels: NDArray) -> NDArray:
    """Map ``0..K-1`` class indices back to the stored category codes."""
    return levels[np.asarray(indices, dtype=np.intp)]


def add_intercept(X: NDArray) -> NDArray:
    """Prepend a column of ones (for models that carry an explicit intercept)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    return np.hstack([ones, X])
