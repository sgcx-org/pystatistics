"""Satterthwaite degrees of freedom for fixed effects in LMM.

Computes approximate denominator degrees of freedom for t-tests on
fixed effects, following the algorithm in lmerTest (Kuznetsova et al., 2017).

The key idea: for each fixed effect β_k, the Satterthwaite df is:

    df_k = 2 × Var(β̂_k)² / [g' × A × g]

where:
    g_j = ∂Var(β̂_k)/∂φ_j   (gradient of the variance w.r.t. variance params)
    A = Var(φ̂)               (asymptotic variance of φ̂, from the Hessian)
    φ = (θ, σ)               (ALL variance parameters: RE theta + residual sd)

Both g and A are computed via numerical differentiation.

The critical distinction from the naive approach: the parameter vector φ must
include σ (residual standard deviation), not just θ. When σ is profiled out,
differentiating only w.r.t. θ misses the contribution of σ to Var(β̂), leading
to incorrect (often far too large) Satterthwaite df for fixed effects that are
not strongly affected by the random effects structure.

All linear algebra goes through the structure-exploiting PLS solver: the
per-coefficient variance C(θ) = (X'V*⁻¹X)⁻¹ is obtained as (RX·RXᵀ)⁻¹ from the
p×p Schur factor, never by forming the dense n×n V*.

References:
    Kuznetsova, A., Brockhoff, P. B., & Christensen, R. H. B. (2017).
    lmerTest Package: Tests in Linear Mixed Effects Models.
    Journal of Statistical Software, 82(13), 1-26.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatistics.mixed._pls_structured import (
    StructuredContext, solve_structured,
)


def _C_from_RX(RX: NDArray) -> NDArray:
    """C(θ) = (X'V*⁻¹X)⁻¹ = (RX·RXᵀ)⁻¹ from the p×p Schur factor RX."""
    try:
        RX_inv = np.linalg.inv(RX)
        return RX_inv.T @ RX_inv
    except np.linalg.LinAlgError:
        return np.linalg.pinv(RX @ RX.T)


def satterthwaite_df(
    theta: NDArray,
    ctx: StructuredContext,
    eps: float = 1e-4,
) -> NDArray:
    """Compute Satterthwaite denominator df for each fixed effect.

    Uses the full variance parameter vector φ = (θ, σ) where σ is the
    residual standard deviation. This matches lmerTest's approach.

    Args:
        theta: Converged θ̂ parameter vector.
        ctx: The structured-solve context (X, y, specs, reml).
        eps: Step size for numerical differentiation.

    Returns:
        Array of Satterthwaite df, one per fixed effect (p,).
    """
    n, p = ctx.X.shape
    n_theta = len(theta)

    res = solve_structured(theta, ctx)
    sigma_sq = res.sigma_sq
    sigma = np.sqrt(sigma_sq)

    # Step 1: C(θ) = (X' V*⁻¹ X)⁻¹ at the converged θ.
    C = _C_from_RX(res.RX)

    # Step 2: Gradients of Var(β̂_k) = σ² × C_kk w.r.t. (θ, σ).
    dC_dtheta = np.zeros((p, n_theta), dtype=np.float64)

    for j in range(n_theta):
        theta_plus = theta.copy()
        theta_minus = theta.copy()

        h = eps * max(abs(theta[j]), 1.0)
        theta_plus[j] += h
        theta_minus[j] -= h

        lb = _theta_lower_bound_at_index(j, ctx.specs)
        if theta_minus[j] < lb:
            # Forward difference (cannot cross the lower bound).
            C_plus = _compute_C(theta_plus, ctx)
            for k in range(p):
                dC_dtheta[k, j] = (C_plus[k, k] - C[k, k]) / h
        else:
            C_plus = _compute_C(theta_plus, ctx)
            C_minus = _compute_C(theta_minus, ctx)
            for k in range(p):
                dC_dtheta[k, j] = (C_plus[k, k] - C_minus[k, k]) / (2 * h)

    # Step 3: Hessian of the REML/ML deviance w.r.t. (θ, σ).
    hessian = _full_deviance_hessian(theta, sigma, ctx, eps)

    # A = 2 × H⁻¹
    try:
        H_inv = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(hessian)
    A = 2.0 * H_inv

    # Step 4: df for each fixed effect.
    df = np.zeros(p, dtype=np.float64)
    n_params = n_theta + 1  # theta + sigma

    for k in range(p):
        var_k = sigma_sq * C[k, k]

        g = np.zeros(n_params, dtype=np.float64)
        g[:n_theta] = sigma_sq * dC_dtheta[k, :]
        g[n_theta] = 2.0 * sigma * C[k, k]

        denom = float(g @ A @ g)

        if denom > 0 and var_k > 0:
            df[k] = 2.0 * var_k**2 / denom
        else:
            df[k] = float(n - p)

        df[k] = max(df[k], 1.0)

    return df


def _compute_C(theta: NDArray, ctx: StructuredContext) -> NDArray:
    """C(θ) = (X' V*(θ)⁻¹ X)⁻¹ via the structured solve's RX factor."""
    res = solve_structured(theta, ctx)
    return _C_from_RX(res.RX)


def _theta_lower_bound_at_index(j: int, specs: list) -> float:
    """Get the lower bound for the j-th element of θ."""
    idx = 0
    for spec in specs:
        q = spec.n_terms
        for row in range(q):
            for col in range(row + 1):
                if idx == j:
                    return 0.0 if row == col else -np.inf
                idx += 1
    return -np.inf


def _full_deviance_hessian(
    theta: NDArray,
    sigma: float,
    ctx: StructuredContext,
    eps: float,
) -> NDArray:
    """Hessian of the REML/ML deviance w.r.t. (θ, σ).

    The deviance as a function of (θ, σ) is:
        REML: d = log|L_θ|² + log|RX|² + (n-p)·log(σ²) + pwrss(θ)/σ²
        ML:   d = log|L_θ|²            + n·log(σ²)      + pwrss(θ)/σ²

    where log|L_θ|² (= logdet_M), RX, and pwrss come from the structured solve.
    """
    n, p = ctx.X.shape
    n_theta = len(theta)
    n_params = n_theta + 1
    reml = ctx.reml

    def deviance(th, sig):
        res = solve_structured(th, ctx)
        sig_sq = sig ** 2
        log_det_L = res.logdet_M
        # NUMERICAL GUARD: prevents log(0) in log-likelihood computation
        log_det_RX = 2.0 * np.sum(
            np.log(np.maximum(np.abs(np.diag(res.RX)), 1e-20))
        )
        if reml:
            df = n - p
            return (log_det_L + log_det_RX
                    + df * np.log(sig_sq) + res.pwrss / sig_sq)
        return (log_det_L + n * np.log(sig_sq) + res.pwrss / sig_sq)

    # Step sizes.
    h = np.zeros(n_params)
    for j in range(n_theta):
        h[j] = eps * max(abs(theta[j]), 1.0)
    h[n_theta] = eps * max(abs(sigma), 1.0)

    d0 = deviance(theta, sigma)

    d_plus = np.zeros(n_params)
    d_minus = np.zeros(n_params)

    for j in range(n_theta):
        tp = theta.copy()
        tp[j] += h[j]
        d_plus[j] = deviance(tp, sigma)

        tm = theta.copy()
        tm[j] -= h[j]
        lb = _theta_lower_bound_at_index(j, ctx.specs)
        if tm[j] < lb:
            tm[j] = lb
        d_minus[j] = deviance(tm, sigma)

    d_plus[n_theta] = deviance(theta, sigma + h[n_theta])
    d_minus[n_theta] = deviance(theta, sigma - h[n_theta])

    H = np.zeros((n_params, n_params), dtype=np.float64)

    for j in range(n_params):
        H[j, j] = (d_plus[j] - 2.0 * d0 + d_minus[j]) / (h[j] ** 2)

    for j in range(n_params):
        for l in range(j + 1, n_params):
            def perturbed(dj, dl):
                th_p = theta.copy()
                sig_p = sigma
                if j < n_theta:
                    th_p[j] += dj
                else:
                    sig_p += dj
                if l < n_theta:
                    th_p[l] += dl
                else:
                    sig_p += dl
                return deviance(th_p, sig_p)

            d_pp = perturbed(h[j], h[l])
            d_pm = perturbed(h[j], -h[l])
            d_mp = perturbed(-h[j], h[l])
            d_mm = perturbed(-h[j], -h[l])

            H[j, l] = (d_pp - d_pm - d_mp + d_mm) / (4.0 * h[j] * h[l])
            H[l, j] = H[j, l]

    return H
