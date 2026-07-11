"""Sparse factor for crossed / nested (multi-factor) designs (structured PLS).

When more than one grouping factor is present (crossed or nested random
effects), observations share random-effect columns across factors, so the
penalized system matrix M = Λ'Z'ZΛ + I is **not** block-diagonal and the
batched per-group trick does not apply. The correct approach — the one lme4
and MixedModels.jl take — is a genuine *sparse* factorization of M with a
fill-reducing ordering. We build Z and M as scipy sparse matrices (never
dense) and factor M with SuperLU under the ``MMD_AT_PLUS_A`` ordering
(minimum-degree on A+Aᵀ, appropriate for a symmetric system), which keeps the
fill-in tractable at the scale where a dense Cholesky runs out of memory.

The flat column ordering of Z is term-major within each factor, factors
concatenated in spec order — identical to ``build_z_matrix`` and what
``_extract_blups`` expects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
from scipy.sparse.linalg import splu, SuperLU

from pystatistics.core.exceptions import NumericalError
from pystatistics.mixed._random_effects import RandomEffectSpec, theta_to_factor


def build_sparse_z(specs: list[RandomEffectSpec]) -> tuple[sp.csc_matrix, list]:
    """Build the sparse design Z (n, q_total) and per-factor block metadata.

    Returns:
        Z: csc sparse matrix, columns term-major per factor, factors in order.
        blocks: list of (offset, J, q) — column offset, n_groups, n_terms —
            one per spec, used to assemble Λ from θ.
    """
    n = specs[0].value_cols.shape[0]
    rows_all, cols_all, data_all = [], [], []
    blocks = []
    offset = 0
    obs = np.arange(n)
    for spec in specs:
        V = spec.value_cols
        if V is None:
            raise ValueError("Sparse path requires spec.value_cols (parse with "
                             "build_dense=False).")
        gids = spec.group_ids
        J, q = spec.n_groups, spec.n_terms
        for t in range(q):
            rows_all.append(obs)
            cols_all.append(offset + t * J + gids)
            data_all.append(V[:, t])
        blocks.append((offset, J, q))
        offset += J * q

    total_q = offset
    Z = sp.csc_matrix(
        (np.concatenate(data_all),
         (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=(n, total_q),
    )
    return Z, blocks


def build_sparse_lambda(
    theta: NDArray, specs: list[RandomEffectSpec], blocks: list
) -> sp.csc_matrix:
    """Assemble the block-diagonal Λ (total_q, total_q) from θ, sparse.

    Λ block for a factor with q terms, J groups is T ⊗ I_J in term-major
    layout: entry (offset+r*J+j, offset+c*J+j) = T[r, c] for r ≥ c, all j.
    """
    total_q = blocks[-1][0] + blocks[-1][1] * blocks[-1][2]
    rows, cols, data = [], [], []
    theta_start = 0
    for spec, (offset, J, q) in zip(specs, blocks):
        n_theta = spec.theta_size
        T = theta_to_factor(theta[theta_start:theta_start + n_theta], spec)
        theta_start += n_theta
        jvec = np.arange(J)
        for r in range(q):
            for c in range(r + 1):
                if T[r, c] != 0.0:
                    rows.append(offset + r * J + jvec)
                    cols.append(offset + c * J + jvec)
                    data.append(np.full(J, T[r, c]))
    Lambda = sp.csc_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(total_q, total_q),
    )
    return Lambda


@dataclass
class SparseFactor:
    """Sparse factor M = Λ'Z'ZΛ + I for crossed / nested designs.

    Mirrors :class:`BatchedFactor`'s interface: flat term-major ``a`` = Λ'Z'y,
    ``B`` = Λ'Z'X, ``logdet_M``, and operators ``apply_Minv``, ``lambda_apply``,
    ``z_apply``.
    """
    lu: SuperLU
    Lambda: sp.csc_matrix
    Z: sp.csc_matrix
    a: NDArray
    B: NDArray
    logdet_M: float
    p: int
    n: int

    def apply_Minv(self, R: NDArray) -> NDArray:
        if R.ndim == 1:
            return self.lu.solve(R)
        # solve column-by-column (SuperLU.solve takes a 2-D RHS directly).
        return self.lu.solve(np.asarray(R, dtype=np.float64))

    def lambda_apply(self, u: NDArray) -> NDArray:
        return self.Lambda @ u

    def z_apply(self, b: NDArray) -> NDArray:
        return self.Z @ b


def build_sparse_factor(
    theta: NDArray,
    specs: list[RandomEffectSpec],
    Z: sp.csc_matrix,
    blocks: list,
    X: NDArray,
    y: NDArray,
    weights: NDArray | None = None,
) -> SparseFactor:
    """Build and factor the sparse penalized system for a crossed design.

    ``weights`` (optional per-observation IRLS weights W) turns the system into
    M = Λ'Z'WZΛ + I with cross-products a = Λ'Z'Wy, B = Λ'Z'WX — the GLMM inner
    loop. None = unit weights (the LMM path), byte-for-behaviour unchanged.
    """
    n, p = X.shape
    Lambda = build_sparse_lambda(theta, specs, blocks)
    ZL = (Z @ Lambda).tocsc()
    total_q = ZL.shape[1]
    if weights is None:
        M = (ZL.T @ ZL + sp.identity(total_q, format='csc')).tocsc()
        a = ZL.T @ y
        B = ZL.T @ X
    else:
        WZL = (sp.diags(weights) @ ZL).tocsc()          # diag(W) ZΛ
        M = (ZL.T @ WZL + sp.identity(total_q, format='csc')).tocsc()
        a = ZL.T @ (weights * y)
        B = ZL.T @ (weights[:, None] * X)

    try:
        lu = splu(M, permc_spec='MMD_AT_PLUS_A')
    except RuntimeError as exc:
        raise NumericalError(
            "Sparse penalized system (Λ'Z'ZΛ + I) could not be factored — "
            "the random effects structure may be degenerate for this data. "
            f"Underlying error: {exc}"
        )

    # log|M| = Σ log|diag(U)| (M is SPD, so det > 0; L is unit-diagonal and
    # the row/column permutations contribute ±1 that cancels in |det|).
    diagU = lu.U.diagonal()
    logdet_M = float(np.sum(np.log(np.abs(diagU))))

    if sp.issparse(B):
        B = B.toarray()
    a = np.asarray(a).ravel()

    return SparseFactor(
        lu=lu, Lambda=Lambda, Z=Z, a=a, B=B,
        logdet_M=logdet_M, p=p, n=n,
    )
