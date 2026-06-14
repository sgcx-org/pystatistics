"""
Example datasets and deterministic missing-data generators for MICE.

Provides a small fixed example with missing values for docs/quick tests, plus
seeded generators that knock holes (MCAR) in a known complete dataset. The
generators are deterministic given their ``seed`` (Rule 6), so tests and R
fixtures see identical inputs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# A small, fully numeric dataset with a handful of missing entries (NaN). Three
# correlated columns, 12 rows. Fixed values — no randomness.
EXAMPLE: NDArray[np.floating] = np.array(
    [
        [5.1, 3.5, 1.4],
        [4.9, np.nan, 1.4],
        [4.7, 3.2, 1.3],
        [4.6, 3.1, np.nan],
        [5.0, 3.6, 1.4],
        [5.4, 3.9, 1.7],
        [np.nan, 3.4, 1.4],
        [5.0, 3.4, 1.5],
        [4.4, 2.9, np.nan],
        [4.9, 3.1, 1.5],
        [5.4, 3.7, 1.5],
        [4.8, np.nan, 1.6],
    ],
    dtype=np.float64,
)


def make_gaussian_complete(
    n: int,
    seed: int,
    *,
    cov: NDArray[np.floating] | None = None,
    mean: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Draw ``n`` rows from a multivariate normal — a complete dataset.

    Deterministic given ``seed``. Default is a 3-variable, moderately correlated
    Gaussian.
    """
    if cov is None:
        cov = np.array(
            [
                [1.0, 0.6, 0.3],
                [0.6, 1.0, 0.5],
                [0.3, 0.5, 1.0],
            ],
            dtype=np.float64,
        )
    if mean is None:
        mean = np.zeros(cov.shape[0], dtype=np.float64)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, cov, size=n)


def make_mcar(
    complete: NDArray[np.floating],
    prop: float,
    seed: int,
    *,
    protect_columns: tuple[int, ...] = (),
) -> NDArray[np.floating]:
    """Return a copy of ``complete`` with a fraction ``prop`` of cells set NaN
    completely at random (MCAR).

    Deterministic given ``seed``. Guarantees no fully-missing row or column so
    the result is a valid :class:`MICEDesign` input.

    Parameters
    ----------
    complete : (n, p) array
        Fully observed source data (no NaN).
    prop : float
        Target fraction of missing cells, in (0, 1).
    seed : int
        RNG seed.
    protect_columns : tuple of int
        Column indices that must stay fully observed (e.g. an outcome).
    """
    if not (0.0 < prop < 1.0):
        from pystatistics.core.exceptions import ValidationError
        raise ValidationError(f"prop must be in (0, 1), got {prop}")
    complete = np.asarray(complete, dtype=np.float64)
    if np.any(np.isnan(complete)):
        from pystatistics.core.exceptions import ValidationError
        raise ValidationError("`complete` must have no missing values")

    n, p = complete.shape
    rng = np.random.default_rng(seed)
    out = complete.copy()

    eligible_cols = [j for j in range(p) if j not in protect_columns]
    mask = np.zeros((n, p), dtype=bool)
    draw = rng.random((n, len(eligible_cols)))
    for local_j, j in enumerate(eligible_cols):
        mask[:, j] = draw[:, local_j] < prop

    # Repair: never blank an entire row or an entire eligible column.
    for i in range(n):
        if np.all(mask[i]):
            keep = rng.integers(0, p)
            mask[i, keep] = False
    for j in eligible_cols:
        if np.all(mask[:, j]):
            keep = rng.integers(0, n)
            mask[keep, j] = False

    out[mask] = np.nan
    return out
