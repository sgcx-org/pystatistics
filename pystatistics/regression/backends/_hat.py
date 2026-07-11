"""Leverage (hat-matrix diagonal) for weighted least-squares / IRLS fits.

Shared by the linear (OLS/WLS) and GLM CPU backends so both expose the same
R-matching ``hatvalues`` diagnostic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular


def hat_diagonal(
    X: NDArray, w: NDArray, R: NDArray | None,
    pivot, rank: int,
) -> NDArray:
    """Hat-matrix diagonal of the (weighted) least-squares fit.

    With ``sqrt(w)·X = Q R`` (pivoted QR of the weighted design), the leverage is
    ``h_i = || row_i(sqrt(w)·X) R^{-1} ||^2`` = the row sums of ``Q^2``. For OLS
    pass unit weights. Returns NaN if the QR factor is unavailable, and clips into
    ``[0, 1]`` against round-off.
    """
    n = X.shape[0]
    if R is None:
        return np.full(n, np.nan, dtype=np.float64)
    sqrt_w = np.sqrt(np.maximum(w, 0.0))
    Xw = X * sqrt_w[:, np.newaxis]
    if pivot is not None:
        Xw = Xw[:, np.asarray(pivot)]
    R_sq = R[:rank, :rank]
    try:
        R_inv = solve_triangular(R_sq, np.eye(rank), lower=False)
    except np.linalg.LinAlgError:
        return np.full(n, np.nan, dtype=np.float64)
    Q = Xw[:, :rank] @ R_inv
    return np.clip(np.sum(Q ** 2, axis=1), 0.0, 1.0)
