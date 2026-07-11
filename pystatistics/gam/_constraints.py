"""Identifiability constraint absorption for GAM smooth terms.

One job: absorb the sum-to-zero constraint ``sum_i f_j(x_i) = 0`` into each
smooth's basis, exactly as mgcv's ``smoothCon(absorb.cons=TRUE)`` does.
Without this, every smooth's span contains the constant function and the
augmented design ``[intercept | B_1 | ... | B_m]`` is exactly rank-deficient
— the root cause of the 4.5.x GAM defect.

The reparameterisation: with ``C = colsums(B)`` (1 x k), find the orthonormal
``Z`` (k x (k-1)) spanning ``null(C)`` via a full QR of ``C'``; the
constrained basis is ``B Z`` and penalty ``Z' S Z``. Any orthonormal basis of
``null(C)`` yields the identical fit (the choice is a coordinate convention);
mgcv's Householder convention may differ by signs/rotations, so raw smooth
coefficients are engine-specific while all fit invariants match.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr


def absorb_sum_to_zero(
    B: NDArray[np.floating[Any]],
    S: NDArray[np.floating[Any]],
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
]:
    """Absorb the sum-to-zero constraint into one smooth's basis/penalty.

    Args:
        B: Unconstrained basis matrix ``(n, k)``.
        S: Unconstrained penalty ``(k, k)``.

    Returns:
        ``(B_c, S_c, Z)``: constrained basis ``(n, k-1)``, constrained
        penalty ``(k-1, k-1)``, and the ``(k, k-1)`` reparameterisation
        matrix (needed to map constrained coefficients back to the
        unconstrained knot/eigen coordinates, e.g. for prediction).
    """
    C = B.sum(axis=0)[:, None]  # (k, 1)
    Q_full, _ = qr(C, mode="full")  # (k, k)
    Z = Q_full[:, 1:]  # (k, k-1), orthonormal basis of null(C')
    B_c = B @ Z
    S_c = Z.T @ S @ Z
    S_c = 0.5 * (S_c + S_c.T)
    return B_c, S_c, Z


def absorb_sum_to_zero_multi(
    B: NDArray[np.floating[Any]],
    S_list: list[NDArray[np.floating[Any]]],
) -> tuple[
    NDArray[np.floating[Any]],
    list[NDArray[np.floating[Any]]],
    NDArray[np.floating[Any]],
]:
    """Absorb ONE sum-to-zero constraint into a basis carrying SEVERAL penalties.

    The tensor-product ``te()`` smooth applies a single identifiability
    constraint to the whole tensor basis but owns one penalty per margin;
    every penalty is reparameterised by the SAME ``Z`` so they stay aligned
    in the constrained coordinate system (mgcv ``smoothCon(te, absorb.cons=
    TRUE)``).

    Args:
        B: Unconstrained tensor basis ``(n, K)``.
        S_list: Per-margin penalties, each ``(K, K)``.

    Returns:
        ``(B_c, S_list_c, Z)``: constrained basis ``(n, K-1)``, the same
        penalties reparameterised to ``(K-1, K-1)``, and the ``(K, K-1)``
        reparameterisation matrix.
    """
    C = B.sum(axis=0)[:, None]  # (K, 1)
    Q_full, _ = qr(C, mode="full")
    Z = Q_full[:, 1:]  # (K, K-1)
    B_c = B @ Z
    S_list_c = []
    for S in S_list:
        S_c = Z.T @ S @ Z
        S_list_c.append(0.5 * (S_c + S_c.T))
    return B_c, S_list_c, Z
