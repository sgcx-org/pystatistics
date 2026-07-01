"""Batched per-group factor for a single grouping factor (structured PLS).

For ONE grouping factor with J groups of q random-effect terms each,
observations in different groups share no random-effect columns, so the
penalized system matrix M = Λ'Z'ZΛ + I is **block-diagonal** with J blocks
of size q×q. The dense q_total×q_total Cholesky the original solver formed is,
mathematically, a *batched* [J, q, q] solve. This backend forms only the J
tiny q×q blocks — never the dense Z (n × J·q) or the dense Gram — so it runs
group counts the dense path runs out of memory on.

Ragged groups (unequal group sizes) are handled by segment-summing the
per-observation outer products into per-group accumulators via ``np.add.at``;
no equal-size assumption is made.

All operations are numpy (no torch): the CPU LMM path must not depend on the
optional torch extra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pystatistics.mixed._random_effects import RandomEffectSpec


def _theta_to_T(theta: NDArray, q: int) -> NDArray:
    """Lower-triangular q×q Cholesky factor T from a θ block."""
    T = np.zeros((q, q), dtype=np.float64)
    idx = 0
    for row in range(q):
        for col in range(row + 1):
            T[row, col] = theta[idx]
            idx += 1
    return T


@dataclass
class BatchedFactor:
    """Block-diagonal factor M = Λ'Z'ZΛ + I for a single grouping factor.

    Exposes the flat (term-major) cross-products ``a`` = Λ'Z'y and ``B`` =
    Λ'Z'X, the log-determinant ``logdet_M`` = log|M|, and the operators
    ``apply_Minv`` (M⁻¹·), ``lambda_apply`` (b = Λu), and ``z_apply`` (Z·b).
    The flat ordering is term-major — index ``t*J + j`` for term ``t``, group
    ``j`` — matching ``_extract_blups``.
    """
    M: NDArray         # (J, q, q) per-group system blocks
    T: NDArray         # (q, q) Cholesky factor of the relative covariance
    V: NDArray         # (n, q) per-observation term values
    gids: NDArray      # (n,) group index per observation
    J: int
    q: int
    p: int
    n: int
    a: NDArray         # (J*q,) Λ'Z'y, term-major
    B: NDArray         # (J*q, p) Λ'Z'X, term-major
    logdet_M: float

    def apply_Minv(self, R: NDArray) -> NDArray:
        """Return M⁻¹ R for R shaped (q_total,) or (q_total, k)."""
        J, q = self.J, self.q
        if R.ndim == 1:
            Rg = R.reshape(q, J).transpose()[:, :, None]  # (J, q, 1)
            Sol = np.linalg.solve(self.M, Rg)             # (J, q, 1)
            return Sol[:, :, 0].transpose().reshape(-1)
        k = R.shape[1]
        Rg = R.reshape(q, J, k).transpose(1, 0, 2)      # (J, q, k)
        Sol = np.linalg.solve(self.M, Rg)               # (J, q, k)
        return Sol.transpose(1, 0, 2).reshape(J * q, k)

    def lambda_apply(self, u: NDArray) -> NDArray:
        """Conditional modes b = Λu for flat spherical u (q_total,)."""
        J, q = self.J, self.q
        u_g = u.reshape(q, J).transpose()               # (J, q)
        b_g = u_g @ self.T.transpose()                  # b_j = T u_j
        return b_g.transpose().reshape(-1)

    def z_apply(self, b: NDArray) -> NDArray:
        """Random-effects contribution Z·b on the response scale (n,)."""
        b_g = b.reshape(self.q, self.J).transpose()     # (J, q)
        return np.einsum('iq,iq->i', self.V, b_g[self.gids])


def build_batched_factor(
    theta: NDArray,
    spec: RandomEffectSpec,
    X: NDArray,
    y: NDArray,
    weights: NDArray | None = None,
) -> BatchedFactor:
    """Build the block-diagonal factor for a single grouping factor.

    Args:
        theta: θ vector for this (single) grouping factor.
        spec: The grouping factor's RandomEffectSpec (must carry value_cols).
        X: Fixed-effects design (n, p).
        y: Response (n,) — the working response z for the GLMM PIRLS path.
        weights: Optional per-observation IRLS weights W (n,). When given the
            system becomes M = Λ'Z'WZΛ + I and the cross-products a = Λ'Z'Wy,
            B = Λ'Z'WX — exactly what the GLMM inner loop needs. None = unit
            weights (the LMM path), byte-for-behaviour unchanged.
    """
    V = spec.value_cols
    if V is None:
        raise ValueError("BatchedFactor requires spec.value_cols (parse with "
                         "build_dense=False for the structured path).")
    gids = spec.group_ids
    J, q = spec.n_groups, spec.n_terms
    n, p = X.shape

    T = _theta_to_T(theta, q)
    VT = V @ T                                          # (n, q) rotated values
    wVT = VT if weights is None else VT * weights[:, None]  # W-scaled rows

    # Per-group accumulators via segment sums (ragged-safe). Scaling ONE factor
    # of each outer product by W gives the weighted Gram Σ w·VT VTᵀ (symmetric).
    A = np.zeros((J, q, q), dtype=np.float64)
    np.add.at(A, gids, wVT[:, :, None] * VT[:, None, :])
    M = A + np.eye(q)[None, :, :]

    a_g = np.zeros((J, q), dtype=np.float64)
    np.add.at(a_g, gids, wVT * y[:, None])

    B_g = np.zeros((J, q, p), dtype=np.float64)
    np.add.at(B_g, gids, wVT[:, :, None] * X[:, None, :])

    sign, logabsdet = np.linalg.slogdet(M)
    logdet_M = float(logabsdet.sum())

    a = a_g.transpose().reshape(-1)                     # term-major (q*J,)
    B = B_g.transpose(1, 0, 2).reshape(J * q, p)        # term-major (q*J, p)

    return BatchedFactor(
        M=M, T=T, V=V, gids=gids, J=J, q=q, p=p, n=n,
        a=a, B=B, logdet_M=logdet_M,
    )
