"""
Penalized Least Squares (PLS) solver for Linear Mixed Models.

For fixed θ (and hence fixed Λ_θ), this solves the penalized least squares
problem to obtain conditional modes of the random effects and profiled
fixed effects:

    minimize ‖y - Xβ - ZΛu‖² + ‖u‖²

where u = Λ⁻¹b are the "spherical" random effects.

The solution proceeds via the augmented system approach. σ² is profiled
out (computed in closed form from the penalized RSS).

References:
    Bates, D., Maechler, M., Bolker, B., & Walker, S. (2015).
    Fitting Linear Mixed-Effects Models Using lme4.
    Journal of Statistical Software, 67(1), 1-48. Section 2.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import scipy.linalg as sla


@dataclass(frozen=True)
class PLSResult:
    """Result from penalized least squares solve.

    Attributes:
        beta: Fixed effects estimates (p,).
        u: Spherical random effects (q,).
        b: Conditional modes b = Λu (q,).
        sigma_sq: Profiled residual variance.
        pwrss: Penalized weighted residual sum of squares
               = ‖y - Xβ - Zb‖² + ‖u‖².
        L: Cholesky factor of (Λ'Z'ZΛ + I), shape (q, q).
        RX: R factor from QR of the projected X, shape (p, p).
        fitted: Xβ + Zb (n,).
        residuals: y - fitted (n,).
    """
    beta: NDArray
    u: NDArray
    b: NDArray
    sigma_sq: float
    pwrss: float
    L: NDArray
    RX: NDArray
    fitted: NDArray
    residuals: NDArray


def solve_pls(
    X: NDArray,
    Z: NDArray,
    y: NDArray,
    Lambda: NDArray,
    weights: NDArray | None = None,
    reml: bool = True,
) -> PLSResult:
    """Solve the penalized least squares problem.

    For the LMM: y = Xβ + Zb + ε, where b ~ N(0, σ²ΛΛ'), ε ~ N(0, σ²I).

    Setting u = Λ⁻¹b (spherical random effects), we minimize:
        ‖y - Xβ - ZΛu‖² + ‖u‖²

    The approach (following lme4):
    1. Form ZΛ (the weighted random effects design matrix)
    2. Compute L = cholesky(Λ'Z'ZΛ + I)  — the "L factor"
    3. Solve for β and u using the augmented system via QR

    In practice we use the two-step approach:
    a. Compute the cross-products and solve the system block-wise
    b. Profile σ² out from the penalized RSS

    Args:
        X: Fixed effects design matrix (n, p).
        Z: Random effects design matrix (n, q).
        y: Response vector (n,).
        Lambda: Relative covariance factor (q, q), block-diagonal.
        weights: Observation weights (n,). If None, unit weights.
        reml: If True, divide pwrss by (n-p) for σ²; if False, divide by n.

    Returns:
        PLSResult with all estimates.
    """
    n, p = X.shape
    q = Z.shape[1]

    # Apply observation weights if provided
    if weights is not None:
        sqrt_w = np.sqrt(weights)
        Xw = X * sqrt_w[:, np.newaxis]
        Zw = Z * sqrt_w[:, np.newaxis]
        yw = y * sqrt_w
    else:
        Xw = X
        Zw = Z
        yw = y

    # ZΛ: the weighted random effects design matrix
    ZLam = Zw @ Lambda  # (n, q)

    # L = cholesky(Λ'Z'ZΛ + I)
    ZtZ_lam = ZLam.T @ ZLam  # (q, q)
    LtL = ZtZ_lam + np.eye(q)

    try:
        L = np.linalg.cholesky(LtL)  # lower triangular
    except np.linalg.LinAlgError:
        # Add small ridge if not PD (shouldn't happen with + I, but be safe)
        LtL += 1e-10 * np.eye(q)
        L = np.linalg.cholesky(LtL)

    # Solve the system using the augmented approach:
    # We need to find β and u that minimize ‖yw - Xw β - ZΛ u‖² + ‖u‖²
    #
    # The normal equations for this penalized system are:
    #   [Λ'Z'ZΛ + I   Λ'Z'X ] [u]   [Λ'Z'y]
    #   [X'ZΛ         X'X   ] [β] = [X'y  ]
    #
    # We solve by first eliminating u via the L factor.

    # Cross-products
    ZLam_t_y = ZLam.T @ yw        # (q,)
    ZLam_t_X = ZLam.T @ Xw        # (q, p)
    Xt_y = Xw.T @ yw              # (p,)
    Xt_X = Xw.T @ Xw              # (p, p)

    # Solve L cu = Λ'Z'y  →  cu = L⁻¹ Λ'Z'y
    cu = sla.solve_triangular(L, ZLam_t_y, lower=True)  # (q,)

    # Solve L CX = Λ'Z'X  →  CX = L⁻¹ Λ'Z'X
    CX = sla.solve_triangular(L, ZLam_t_X, lower=True)  # (q, p)

    # RX'RX = X'X - CX'CX  (the Schur complement)
    RtR = Xt_X - CX.T @ CX  # (p, p)

    # Solve for β: RX'RX β = X'y - CX'cu
    rhs_beta = Xt_y - CX.T @ cu  # (p,)

    try:
        RX = np.linalg.cholesky(RtR)  # lower triangular of Schur complement
        # RX RX' β = rhs → solve two triangular systems
        tmp = sla.solve_triangular(RX, rhs_beta, lower=True)
        beta = sla.solve_triangular(RX.T, tmp, lower=False)
    except np.linalg.LinAlgError:
        # Fallback to lstsq if Cholesky fails
        beta, _, _, _ = np.linalg.lstsq(RtR, rhs_beta, rcond=None)
        # Compute RX via eigendecomposition for the deviance calculation
        eigvals = np.linalg.eigvalsh(RtR)
        eigvals = np.maximum(eigvals, 1e-20)
        RX = np.diag(np.sqrt(eigvals))

    # Solve for u: L L' u = Λ'Z'(y - Xβ)
    ZLam_t_resid = ZLam_t_y - ZLam_t_X @ beta  # (q,)
    cu_final = sla.solve_triangular(L, ZLam_t_resid, lower=True)
    u = sla.solve_triangular(L.T, cu_final, lower=False)  # (q,)

    # Conditional modes: b = Λu
    b = Lambda @ u  # (q,)

    # Fitted values and residuals (use original unweighted X, Z)
    fitted = X @ beta + Z @ b
    residuals = y - fitted

    # Penalized weighted RSS
    if weights is not None:
        wrss = float(np.sum(weights * residuals**2))
    else:
        wrss = float(np.sum(residuals**2))
    pwrss = wrss + float(u @ u)

    # Profiled σ²
    if reml:
        sigma_sq = pwrss / (n - p)
    else:
        sigma_sq = pwrss / n

    return PLSResult(
        beta=beta,
        u=u,
        b=b,
        sigma_sq=sigma_sq,
        pwrss=pwrss,
        L=L,
        RX=RX,
        fitted=fitted,
        residuals=residuals,
    )
