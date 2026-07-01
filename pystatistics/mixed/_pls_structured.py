"""Structure-exploiting penalized least squares for LMM.

This is the LMM analogue of ``_pls.solve_pls``, but it never materializes the
dense Z (n × Σ J_k·q_k) or the dense Gram Z'Z. It dispatches on the design:

  * a single grouping factor → block-diagonal system → batched per-group
    dense solve (:mod:`._struct_batched`);
  * crossed / nested (multiple grouping factors) → sparse factorization with a
    fill-reducing ordering (:mod:`._struct_sparse`).

Both backends expose the same small interface (flat term-major cross-products
``a`` = Λ'Z'y and ``B`` = Λ'Z'X, the operator ``apply_Minv`` = M⁻¹·, the
log-determinant ``logdet_M`` = log|M|, and ``lambda_apply`` / ``z_apply``), so
the Schur-complement assembly below is backend-agnostic.

The estimates (β, b, σ², the REML/ML deviance) are identical to the dense
``solve_pls`` at the same θ — only the linear-algebra path differs.

References:
    Bates, D., Maechler, M., Bolker, B., & Walker, S. (2015). Section 5.4
    (the sparse Cholesky factorization that this mirrors with batched / SuperLU
    structure depending on the design).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import scipy.linalg as sla

from pystatistics.core.exceptions import NumericalError
from pystatistics.mixed._random_effects import RandomEffectSpec
from pystatistics.mixed._struct_batched import build_batched_factor
from pystatistics.mixed._struct_sparse import (
    build_sparse_z, build_sparse_factor,
)


@dataclass(frozen=True)
class StructuredPLSResult:
    """Result of a structured PLS solve.

    Carries the same quantities the dense ``PLSResult`` exposes to downstream
    code (β, u, b, σ², pwrss, RX, fitted, residuals), plus ``logdet_M`` =
    log|Λ'Z'ZΛ + I| in place of the dense L factor (the structured path never
    forms a dense q×q L).
    """
    beta: NDArray
    u: NDArray
    b: NDArray
    sigma_sq: float
    pwrss: float
    logdet_M: float
    RX: NDArray
    fitted: NDArray
    residuals: NDArray


@dataclass
class StructuredContext:
    """Reusable, θ-independent setup for repeated structured solves.

    Built once per fit; the optimizer then calls ``deviance_structured`` /
    ``solve_structured`` many times with different θ against the same context.
    For the sparse path it caches the sparse Z and per-factor block metadata so
    they are not rebuilt every evaluation. For the single-factor (batched) path
    it also caches the θ-independent per-group cross-products (``bat_S`` = Vⱼ'Vⱼ,
    ``bat_P`` = Vⱼ'Xⱼ, ``bat_Vty`` = Vⱼ'yⱼ) that the analytic θ-gradient reuses.
    """
    specs: list
    X: NDArray
    y: NDArray
    reml: bool
    single_factor: bool
    Z_sparse: object = None
    sp_blocks: list | None = None
    # Single-factor analytic-gradient cache (None for the sparse path).
    bat_S: NDArray | None = None       # (J, q, q) per-group Vⱼ'Vⱼ
    bat_P: NDArray | None = None       # (J, q, p) per-group Vⱼ'Xⱼ
    bat_Vty: NDArray | None = None     # (J, q)    per-group Vⱼ'yⱼ
    bat_V: NDArray | None = None       # (n, q)    RE term values
    bat_gids: NDArray | None = None    # (n,)      group index per obs


def build_structured_context(
    X: NDArray, y: NDArray, specs: list[RandomEffectSpec], reml: bool,
) -> StructuredContext:
    """Prepare the θ-independent context for structured solves."""
    single = len(specs) == 1
    Z_sparse, blocks = (None, None)
    bat_S = bat_P = bat_Vty = bat_V = bat_gids = None
    if not single:
        Z_sparse, blocks = build_sparse_z(specs)
    else:
        spec = specs[0]
        V = spec.value_cols
        if V is not None:                          # structured parse
            gids = spec.group_ids
            J, q = spec.n_groups, spec.n_terms
            p = X.shape[1]
            bat_S = np.zeros((J, q, q))
            np.add.at(bat_S, gids, V[:, :, None] * V[:, None, :])
            bat_P = np.zeros((J, q, p))
            np.add.at(bat_P, gids, V[:, :, None] * X[:, None, :])
            bat_Vty = np.zeros((J, q))
            np.add.at(bat_Vty, gids, V * y[:, None])
            bat_V, bat_gids = V, gids
    return StructuredContext(
        specs=specs, X=X, y=y, reml=reml,
        single_factor=single, Z_sparse=Z_sparse, sp_blocks=blocks,
        bat_S=bat_S, bat_P=bat_P, bat_Vty=bat_Vty, bat_V=bat_V, bat_gids=bat_gids,
    )


def _build_factor(theta: NDArray, ctx: StructuredContext):
    if ctx.single_factor:
        return build_batched_factor(theta, ctx.specs[0], ctx.X, ctx.y)
    return build_sparse_factor(
        theta, ctx.specs, ctx.Z_sparse, ctx.sp_blocks, ctx.X, ctx.y,
    )


def build_weighted_factor(ctx: StructuredContext, theta: NDArray,
                          response: NDArray, weights: NDArray):
    """Structured factor for the GLMM inner loop: M = Λ'Z'WZΛ + I with the
    working ``response`` and IRLS ``weights`` (dispatches batched vs sparse on
    the same context the LMM path uses). ``ctx.y`` is ignored — the GLMM PIRLS
    passes the current working response explicitly."""
    if ctx.single_factor:
        return build_batched_factor(theta, ctx.specs[0], ctx.X, response,
                                    weights=weights)
    return build_sparse_factor(
        theta, ctx.specs, ctx.Z_sparse, ctx.sp_blocks, ctx.X, response,
        weights=weights,
    )


def solve_structured(theta: NDArray, ctx: StructuredContext) -> StructuredPLSResult:
    """Solve the penalized least squares problem at θ via the structured path.

    The Schur-complement assembly is identical for both backends, using *full*
    M⁻¹ solves (the batched and sparse backends each implement ``apply_Minv``):

        W   = M⁻¹ B,   m_y = M⁻¹ a
        RtR = X'X − Bᵀ W           (= X'V*⁻¹X, the Schur complement)
        rhs = X'y − Bᵀ m_y
        β   solves RtR β = rhs       via the Cholesky factor RX
        u   = m_y − W β
        b   = Λ u,   fitted = Xβ + Zb
    """
    X, y, reml = ctx.X, ctx.y, ctx.reml
    n, p = X.shape

    factor = _build_factor(theta, ctx)
    a, B = factor.a, factor.B

    W = factor.apply_Minv(B)                # (q, p)
    m_y = factor.apply_Minv(a)              # (q,)

    RtR = X.T @ X - B.T @ W                 # (p, p) Schur complement
    rhs = X.T @ y - B.T @ m_y               # (p,)

    try:
        RX = np.linalg.cholesky(RtR)        # lower triangular
        tmp = sla.solve_triangular(RX, rhs, lower=True)
        beta = sla.solve_triangular(RX.T, tmp, lower=False)
    except np.linalg.LinAlgError:
        raise NumericalError(
            "Fixed effects system (R'R) is singular — Cholesky solve failed. "
            "This typically means the design matrix has collinear predictors. "
            "Suggestions:\n"
            "  - Check for collinearity in the fixed effects design matrix\n"
            "  - Remove redundant predictors"
        )

    u = m_y - W @ beta                      # (q,)
    b = factor.lambda_apply(u)              # (q,)
    Zb = factor.z_apply(b)                  # (n,)

    fitted = X @ beta + Zb
    residuals = y - fitted
    pwrss = float(residuals @ residuals + u @ u)
    sigma_sq = pwrss / (n - p) if reml else pwrss / n

    return StructuredPLSResult(
        beta=beta, u=u, b=b, sigma_sq=sigma_sq, pwrss=pwrss,
        logdet_M=factor.logdet_M, RX=RX, fitted=fitted, residuals=residuals,
    )


def deviance_from_result(res: StructuredPLSResult, n: int, p: int, reml: bool) -> float:
    """Profiled REML/ML deviance from a structured PLS result."""
    # NUMERICAL GUARD: prevents log(0) in log-likelihood computation
    log_det_RX = 2.0 * np.sum(np.log(np.maximum(np.abs(np.diag(res.RX)), 1e-20)))
    if reml:
        df = n - p
        return float(res.logdet_M + log_det_RX
                     + df * (1.0 + np.log(2.0 * np.pi * res.pwrss / df)))
    return float(res.logdet_M
                 + n * (1.0 + np.log(2.0 * np.pi * res.pwrss / n)))


def deviance_structured(theta: NDArray, ctx: StructuredContext) -> float:
    """Profiled deviance at θ — the objective the outer optimizer minimizes."""
    res = solve_structured(theta, ctx)
    n, p = ctx.X.shape
    return deviance_from_result(res, n, p, ctx.reml)


def has_analytic_gradient(ctx: StructuredContext) -> bool:
    """True when :func:`deviance_and_grad_structured` returns a real gradient
    (the single grouping-factor / batched path). The crossed / nested (sparse)
    path has no analytic θ-gradient yet — callers fall back to finite differences.
    """
    return ctx.single_factor and ctx.bat_S is not None


def deviance_and_grad_structured(
    theta: NDArray, ctx: StructuredContext,
) -> tuple[float, NDArray]:
    """Profiled REML/ML deviance AND its analytic θ-gradient (single factor).

    Same objective value as :func:`deviance_structured` (verified equal to
    round-off), plus the exact gradient — so L-BFGS-B costs one evaluation per
    step instead of the ``2·dim(θ)+1`` a finite-difference gradient needs. The
    gradient is the sum of three terms, each assembled from the per-group
    quantities of the batched solve:

        ∂/∂θ_k [ log|M| ]      = Σ_j tr(Mⱼ⁻¹ ∂Mⱼ/∂θ_k)
        ∂/∂θ_k [ log|RtR| ]    = tr(RtR⁻¹ ∂RtR/∂θ_k)                 (REML only)
        ∂/∂θ_k [ pwrss ]       = -2 · residᵀ (∂(ZΛu)/∂θ_k)          (envelope thm)

    with Mⱼ = Tᵀ Sⱼ T + I, Bⱼ = Tᵀ Pⱼ, aⱼ = Tᵀ (Vⱼ'yⱼ), and E_k = ∂T/∂θ_k a
    single-entry lower-triangular basis matrix. Validated against a finite-
    difference gradient of the same deviance to ~1e-7.
    """
    from pystatistics.mixed._struct_batched import _theta_to_T

    X, y, reml = ctx.X, ctx.y, ctx.reml
    n, p = X.shape
    S, P, Vty, V, gids = ctx.bat_S, ctx.bat_P, ctx.bat_Vty, ctx.bat_V, ctx.bat_gids
    J, q = S.shape[0], S.shape[1]

    T = _theta_to_T(theta, q)
    M = np.einsum('ki,jkl,lm->jim', T, S, T) + np.eye(q)
    Minv = np.linalg.inv(M)
    a = np.einsum('ki,jk->ji', T, Vty)                     # Tᵀ (Vⱼ'yⱼ)
    B = np.einsum('ki,jkl->jil', T, P)                     # Tᵀ Pⱼ  (J,q,p)
    Minv_a = np.einsum('jik,jk->ji', Minv, a)
    Minv_B = np.einsum('jik,jkl->jil', Minv, B)
    RtR = X.T @ X - np.einsum('jqp,jqr->pr', B, Minv_B)
    rhs = X.T @ y - np.einsum('jqp,jq->p', B, Minv_a)
    RtRinv = np.linalg.inv(RtR)
    beta = RtRinv @ rhs
    u = Minv_a - np.einsum('jqp,p->jq', Minv_B, beta)
    Zb = np.einsum('nq,nq->n', V @ T, u[gids])
    resid = y - X @ beta - Zb
    pwrss = float(resid @ resid + (u * u).sum())

    # NUMERICAL GUARD: prevents log(0) in the determinant terms.
    logdet_M = float(np.sum(np.log(np.maximum(np.linalg.det(M), 1e-300))))
    _sign, logdet_RtR = np.linalg.slogdet(RtR)
    df = (n - p) if reml else n
    dev = logdet_M + (logdet_RtR if reml else 0.0) \
        + df * (1.0 + np.log(2.0 * np.pi * pwrss / df))

    # --- analytic gradient over the lower-triangular θ components ---
    tri = [(r, c) for r in range(q) for c in range(r + 1)]
    grad = np.zeros(len(tri))
    for k, (r, c) in enumerate(tri):
        E = np.zeros((q, q)); E[r, c] = 1.0
        dM = (np.einsum('ki,jkl,lm->jim', E, S, T)
              + np.einsum('ki,jkl,lm->jim', T, S, E))
        t1 = np.einsum('jik,jki->', Minv, dM)              # Σ tr(Mⱼ⁻¹ ∂Mⱼ)
        if reml:
            dB = np.einsum('ki,jkl->jil', E, P)            # ∂Bⱼ = Eᵀ Pⱼ
            ta = np.einsum('jqp,jqr->pr', dB, Minv_B)
            Minv_dB = np.einsum('jik,jkr->jir', Minv, dB)
            tb = np.einsum('jqp,jqr->pr', B, Minv_dB)
            tc = np.einsum('jqp,jqr,jrs->ps', Minv_B, dM, Minv_B)
            dRtR = -(ta + tb - tc)
            t2 = np.einsum('pr,rp->', RtRinv, dRtR)
        else:
            t2 = 0.0
        Eu = np.einsum('rc,jc->jr', E, u)                  # ∂(Λu)/∂θ_k per group
        dZb = np.einsum('nq,nq->n', V, Eu[gids])
        t3 = -2.0 * float(resid @ dZb)                     # envelope: ∂pwrss
        grad[k] = t1 + t2 + df * t3 / pwrss
    return float(dev), grad
