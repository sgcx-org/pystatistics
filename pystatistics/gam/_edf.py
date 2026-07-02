"""Effective degrees of freedom, Ref.df and posterior covariance for GAMs.

One job: turn the two triangular factors of a stable P-IRLS fit —
``R_x`` (weighted design) and the PIVOTED ``R`` (penalized augmented system,
with its column permutation ``piv``) — into the inferential quantities mgcv
reports, without ever forming or inverting the penalized normal-equations
matrix:

    H       = A^{-1} X'WX = R^{-1} R^{-T} (X'WX)     (p x p, pivot-aware)
    edf_j   = tr(H[block_j]),  total_edf = tr(H)
    Ref.df_j= tr((2H - H H)[block_j])                (mgcv convention)
    V_beta  = phi * R^{-1} R^{-T}                    (Bayesian posterior,
                                                      mgcv's ``Vp``)
    log|A|  = 2 * sum(log|diag(R)|) over the rank block

Everything is O(p^3) on p-by-p triangles; the conditioning of ``R`` is the
square root of the old normal-equations path's. When the pivoted solve
detected dependent columns (``rank < p``), those coordinates are dropped:
their coefficients are zero, and their H / covariance rows and columns are
zero — mirroring mgcv's column-dropping, never a silent ridge.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular


def influence_matrix(
    R: NDArray[np.floating[Any]],
    R_x: NDArray[np.floating[Any]],
    piv: NDArray[np.integer[Any]],
    rank: int,
) -> NDArray[np.floating[Any]]:
    """``H = A^{-1} X'WX`` in ORIGINAL column order, via triangular solves.

    Args:
        R: Pivoted augmented triangle ``(p, p)`` from the penalized solve.
        R_x: Weighted-design triangle ``(p, p)`` (original column order).
        piv: Column permutation such that ``R``'s column ``i`` is original
            column ``piv[i]``.
        rank: Numerical rank from the pivoted solve.

    Returns:
        ``(p, p)`` influence matrix in original coordinates; rows/columns of
        dropped (dependent) coordinates are zero.
    """
    p = R.shape[0]
    K = R_x.T @ R_x                       # X'WX, original order
    K_piv = K[np.ix_(piv, piv)]
    H_piv = np.zeros((p, p), dtype=np.float64)
    Rr = R[:rank, :rank]
    G = solve_triangular(Rr.T, K_piv[:rank, :], lower=True)
    H_piv[:rank, :] = solve_triangular(Rr, G)
    H = np.zeros((p, p), dtype=np.float64)
    H[np.ix_(piv, piv)] = H_piv
    return H


def edf_per_block(
    H: NDArray[np.floating[Any]],
    blocks: list[tuple[int, int]],
) -> NDArray[np.floating[Any]]:
    """Per-smooth EDF: block traces of the influence matrix."""
    return np.array(
        [float(np.trace(H[s:e, s:e])) for s, e in blocks], dtype=np.float64,
    )


def ref_df_per_block(
    H: NDArray[np.floating[Any]],
    blocks: list[tuple[int, int]],
) -> NDArray[np.floating[Any]]:
    """mgcv's ``Ref.df``: block traces of ``2H - HH``."""
    H2 = H @ H
    return np.array(
        [float(np.trace(H[s:e, s:e]) * 2.0 - np.trace(H2[s:e, s:e]))
         for s, e in blocks],
        dtype=np.float64,
    )


def total_edf(H: NDArray[np.floating[Any]]) -> float:
    """Total effective degrees of freedom, tr(H)."""
    return float(np.trace(H))


def posterior_covariance(
    R: NDArray[np.floating[Any]],
    piv: NDArray[np.integer[Any]],
    rank: int,
    scale: float,
) -> NDArray[np.floating[Any]]:
    """Bayesian posterior covariance ``phi * (X'WX + S_lambda)^{-1}``.

    Original column order; dropped coordinates get zero rows/columns (their
    coefficients are pinned at zero).
    """
    p = R.shape[0]
    Rr = R[:rank, :rank]
    R_inv = solve_triangular(Rr, np.eye(rank))
    V_piv = np.zeros((p, p), dtype=np.float64)
    V_piv[:rank, :rank] = scale * (R_inv @ R_inv.T)
    V = np.zeros((p, p), dtype=np.float64)
    V[np.ix_(piv, piv)] = V_piv
    return V


def logdet_penalized(
    R: NDArray[np.floating[Any]], rank: int,
) -> float:
    """``log|X'WX + S_lambda|`` (pseudo-determinant over the rank block)."""
    d = np.abs(np.diag(R)[:rank])
    return float(2.0 * np.sum(np.log(d)))
