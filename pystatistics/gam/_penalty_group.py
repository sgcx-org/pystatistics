"""Joint pseudo-determinant of a smooth's summed penalty, and its gradient.

The Laplace-REML criterion carries the term ``log|S_lambda|_+`` (the log
pseudo-determinant of ``sum_j lambda_j S_j``) and the null-space dimension
``M_p = p - rank(S_lambda)``; its gradient carries
``d log|S_lambda|_+ / d rho_j`` with ``rho_j = log lambda_j``.

For an ORDINARY smooth (one penalty per coefficient block) these decompose
per penalty into the block-orthogonal shortcut

    log|lambda_j S_j|_+          = rank_j log lambda_j + logdet_pos(S_j)
    d log|lambda_j S_j|_+/d rho_j = rank_j

which is what ``_criteria`` / ``_gradient`` used before tensor smooths
existed. A TENSOR-PRODUCT smooth carries SEVERAL penalties on the SAME
coefficient block (each margin's Kronecker-embedded penalty). Those
penalties OVERLAP: the null space of ``sum_j lambda_j S_j`` is the tensor
product of the marginal null spaces, so neither the log pseudo-determinant
nor its derivative decomposes per penalty. This module computes the JOINT
quantities per determinant GROUP (``PenaltyRoot.group`` — one per ordinary
smooth, one shared across a tensor smooth's margins):

    log|S_g,lambda|_+  = log det( P_g' (sum_{j in g} lambda_j S_j) P_g )
    rank(S_g,lambda)   = dim range(sum_{j in g} S_j)   (lambda-invariant)
    d log|S_g,lambda|_+/d rho_j = lambda_j tr(S_g,lambda^+ S_j)

with ``P_g`` an orthonormal basis of the group's penalty range. The range
(hence the rank) is taken from the UNIT-weight structural sum ``sum_j S_j``,
so it is invariant to the smoothing parameters and stable when margin
lambdas differ by many orders of magnitude — the pseudo-determinant is then
a genuine determinant on that fixed subspace. For a SINGLETON group every
formula collapses back to the shortcut exactly (verified in the tests), so
non-tensor smooths are numerically identical to the pre-tensor code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pystatistics.gam._pirls import PenaltyRoot

_EIG_TOL = 1e-12  # relative eigenvalue floor for the structural range


def _group_order(roots: list[PenaltyRoot]) -> list[list[int]]:
    """Root indices partitioned by ``group``, in first-seen group order."""
    seen: dict[int, list[int]] = {}
    order: list[int] = []
    for idx, r in enumerate(roots):
        if r.group not in seen:
            seen[r.group] = []
            order.append(r.group)
        seen[r.group].append(idx)
    return [seen[g] for g in order]


def _reconstruct(root: PenaltyRoot) -> NDArray[np.floating[Any]]:
    """Full block-coordinate penalty ``S_j = rows' rows`` (k_g x k_g)."""
    return root.rows.T @ root.rows


def _joint(
    s_list: list[NDArray[np.floating[Any]]],
    lam_list: list[float],
) -> tuple[float, int, NDArray[np.floating[Any]]]:
    """Joint ``(logdet, rank, pseudo-inverse)`` of ``sum_j lam_j S_j``.

    The range (and rank) come from the unit-weight structural sum so they do
    not move with the smoothing parameters; the log-determinant and
    pseudo-inverse are then formed on that fixed range, where the weighted
    sum is positive definite.
    """
    s_struct = np.add.reduce(s_list)
    s_struct = 0.5 * (s_struct + s_struct.T)
    ev0, u0 = np.linalg.eigh(s_struct)
    thresh = ev0.max() * _EIG_TOL if ev0.size and ev0.max() > 0.0 else 0.0
    pos = ev0 > thresh
    rank = int(pos.sum())
    up = u0[:, pos]  # (k_g, rank) orthonormal basis of the penalty range

    s_lam = np.add.reduce([lam * s for s, lam in zip(s_list, lam_list)])
    s_lam = 0.5 * (s_lam + s_lam.T)
    m = up.T @ s_lam @ up  # (rank, rank), positive definite on the range
    evm, um = np.linalg.eigh(m)
    logdet = float(np.sum(np.log(evm)))
    m_inv = (um / evm) @ um.T
    s_pinv = up @ m_inv @ up.T  # pseudo-inverse of s_lam on its range
    return logdet, rank, s_pinv


def penalty_logdet(
    roots: list[PenaltyRoot],
    lambdas: NDArray[np.floating[Any]],
) -> tuple[float, int]:
    """``(log|S_lambda|_+, rank(S_lambda))`` summed over determinant groups."""
    logdet_total = 0.0
    rank_total = 0
    for idxs in _group_order(roots):
        s_list = [_reconstruct(roots[i]) for i in idxs]
        lam_list = [float(lambdas[i]) for i in idxs]
        logdet, rank, _ = _joint(s_list, lam_list)
        logdet_total += logdet
        rank_total += rank
    return logdet_total, rank_total


def penalty_logdet_grad(
    roots: list[PenaltyRoot],
    lambdas: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """``d log|S_lambda|_+ / d rho_j = lambda_j tr(S_g,lambda^+ S_j)``.

    Returned per ROOT (aligned with ``roots``/``lambdas``). For a singleton
    group this equals ``rank_j`` exactly — the block-orthogonal shortcut.
    """
    grad = np.zeros(len(roots), dtype=np.float64)
    for idxs in _group_order(roots):
        s_list = [_reconstruct(roots[i]) for i in idxs]
        lam_list = [float(lambdas[i]) for i in idxs]
        _, _, s_pinv = _joint(s_list, lam_list)
        for i, s_j in zip(idxs, s_list):
            grad[i] = float(lambdas[i]) * float(np.einsum("ab,ba->", s_pinv, s_j))
    return grad
