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
    they are not rebuilt every evaluation.
    """
    specs: list
    X: NDArray
    y: NDArray
    reml: bool
    single_factor: bool
    Z_sparse: object = None
    sp_blocks: list | None = None


def build_structured_context(
    X: NDArray, y: NDArray, specs: list[RandomEffectSpec], reml: bool,
) -> StructuredContext:
    """Prepare the θ-independent context for structured solves."""
    single = len(specs) == 1
    Z_sparse, blocks = (None, None)
    if not single:
        Z_sparse, blocks = build_sparse_z(specs)
    return StructuredContext(
        specs=specs, X=X, y=y, reml=reml,
        single_factor=single, Z_sparse=Z_sparse, sp_blocks=blocks,
    )


def _build_factor(theta: NDArray, ctx: StructuredContext):
    if ctx.single_factor:
        return build_batched_factor(theta, ctx.specs[0], ctx.X, ctx.y)
    return build_sparse_factor(
        theta, ctx.specs, ctx.Z_sparse, ctx.sp_blocks, ctx.X, ctx.y,
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
