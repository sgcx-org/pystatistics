"""
Satterthwaite degrees of freedom for fixed effects in LMM.

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

References:
    Kuznetsova, A., Brockhoff, P. B., & Christensen, R. H. B. (2017).
    lmerTest Package: Tests in Linear Mixed Effects Models.
    Journal of Statistical Software, 82(13), 1-26.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatistics.mixed._random_effects import RandomEffectSpec, build_lambda
from pystatistics.mixed._pls import solve_pls


def satterthwaite_df(
    theta: NDArray,
    X: NDArray,
    Z: NDArray,
    y: NDArray,
    specs: list[RandomEffectSpec],
    reml: bool = True,
    eps: float = 1e-4,
) -> NDArray:
    """Compute Satterthwaite denominator df for each fixed effect.

    Uses the full variance parameter vector φ = (θ, σ) where σ is the
    residual standard deviation. This matches lmerTest's approach.

    Algorithm:
    1. At converged (θ̂, σ̂), decompose Var(β̂) = σ² × C(θ)
       where C(θ) = (X'V*(θ)⁻¹X)⁻¹ and V*(θ) = ZΛΛ'Z' + I
    2. Compute gradients of Var(β̂_k) w.r.t. both θ and σ
    3. Compute the Hessian of the REML deviance w.r.t. (θ, σ)
    4. For each β_k: df_k = 2 Var(β̂_k)² / [g' A g]

    Args:
        theta: Converged θ̂ parameter vector.
        X, Z, y: Model matrices.
        specs: Random effect specifications.
        reml: REML or ML.
        eps: Step size for numerical differentiation.

    Returns:
        Array of Satterthwaite df, one per fixed effect (p,).
    """
    n, p = X.shape
    n_theta = len(theta)

    # Get sigma from a PLS solve at the converged theta
    Lambda = build_lambda(theta, specs)
    pls = solve_pls(X, Z, y, Lambda, reml=reml)
    sigma = np.sqrt(pls.sigma_sq)
    sigma_sq = pls.sigma_sq

    # Step 1: Compute C(θ) = (X' V*(θ)⁻¹ X)⁻¹ where V* = ZΛΛ'Z' + I
    V_star = Z @ Lambda @ Lambda.T @ Z.T + np.eye(n)
    C = np.linalg.inv(X.T @ np.linalg.solve(V_star, X))

    # Step 2: Gradients of Var(β̂_k) = σ² × C_kk w.r.t. (θ, σ)
    # dVar/dθ_j = σ² × dC_kk/dθ_j
    # dVar/dσ = 2σ × C_kk
    dC_dtheta = np.zeros((p, n_theta), dtype=np.float64)

    for j in range(n_theta):
        theta_plus = theta.copy()
        theta_minus = theta.copy()

        h = eps * max(abs(theta[j]), 1.0)
        theta_plus[j] += h
        theta_minus[j] -= h

        # Enforce lower bound on diagonal elements
        lb = _theta_lower_bound_at_index(j, specs)
        if theta_minus[j] < lb:
            # Forward difference
            C_plus = _compute_C(theta_plus, X, Z, specs, n)
            for k in range(p):
                dC_dtheta[k, j] = (C_plus[k, k] - C[k, k]) / h
        else:
            # Central difference
            C_plus = _compute_C(theta_plus, X, Z, specs, n)
            C_minus = _compute_C(theta_minus, X, Z, specs, n)
            for k in range(p):
                dC_dtheta[k, j] = (C_plus[k, k] - C_minus[k, k]) / (2 * h)

    # Full gradient for each beta_k: g = [sigma^2 * dC/dtheta, 2*sigma*C_kk]
    # (n_theta + 1 elements per beta_k)

    # Step 3: Hessian of the REML deviance w.r.t. (θ, σ)
    hessian = _full_deviance_hessian(theta, sigma, X, Z, y, specs, reml, eps)

    # A = 2 × H⁻¹
    try:
        H_inv = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(hessian)
    A = 2.0 * H_inv

    # Step 4: Compute df for each fixed effect
    df = np.zeros(p, dtype=np.float64)
    n_params = n_theta + 1  # theta + sigma

    for k in range(p):
        var_k = sigma_sq * C[k, k]

        # Gradient vector: [dVar/dtheta_1, ..., dVar/dtheta_m, dVar/dsigma]
        g = np.zeros(n_params, dtype=np.float64)
        g[:n_theta] = sigma_sq * dC_dtheta[k, :]
        g[n_theta] = 2.0 * sigma * C[k, k]

        # Denominator: g' A g
        denom = float(g @ A @ g)

        if denom > 0 and var_k > 0:
            df[k] = 2.0 * var_k**2 / denom
        else:
            df[k] = float(n - p)

        # Clamp to reasonable range
        df[k] = max(df[k], 1.0)

    return df


def _compute_C(
    theta: NDArray,
    X: NDArray,
    Z: NDArray,
    specs: list[RandomEffectSpec],
    n: int,
) -> NDArray:
    """Compute C(θ) = (X' V*(θ)⁻¹ X)⁻¹ where V* = ZΛΛ'Z' + I."""
    Lambda = build_lambda(theta, specs)
    V_star = Z @ Lambda @ Lambda.T @ Z.T + np.eye(n)
    return np.linalg.inv(X.T @ np.linalg.solve(V_star, X))


def _theta_lower_bound_at_index(j: int, specs: list[RandomEffectSpec]) -> float:
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
    X: NDArray,
    Z: NDArray,
    y: NDArray,
    specs: list[RandomEffectSpec],
    reml: bool,
    eps: float,
) -> NDArray:
    """Compute Hessian of the REML deviance w.r.t. (θ, σ).

    The REML deviance as a function of (θ, σ) is:
    d(θ, σ) = log|L_θ|² + log|RX|² + (n-p)log(σ²) + pwrss(θ)/σ²

    where L_θ and RX come from PLS at the given θ, and pwrss is the
    penalized weighted residual sum of squares.

    For ML: d(θ, σ) = log|L_θ|² + n·log(σ²) + pwrss(θ)/σ²

    Args:
        theta: θ parameter vector.
        sigma: Residual standard deviation.
        X, Z, y: Model matrices.
        specs: Random effect specifications.
        reml: REML or ML.
        eps: Step size for finite differences.

    Returns:
        Hessian matrix of shape (n_theta + 1, n_theta + 1).
    """
    n, p = X.shape
    n_theta = len(theta)
    n_params = n_theta + 1

    def deviance(th, sig):
        """Compute deviance as function of (theta, sigma)."""
        Lam = build_lambda(th, specs)
        pls_local = solve_pls(X, Z, y, Lam, reml=reml)

        sig_sq = sig ** 2
        log_det_L = 2.0 * np.sum(
            np.log(np.maximum(np.diag(pls_local.L), 1e-20))
        )
        log_det_RX = 2.0 * np.sum(
            np.log(np.maximum(np.abs(np.diag(pls_local.RX)), 1e-20))
        )

        if reml:
            df = n - p
            return (log_det_L + log_det_RX
                    + df * np.log(sig_sq) + pls_local.pwrss / sig_sq)
        else:
            return (log_det_L
                    + n * np.log(sig_sq) + pls_local.pwrss / sig_sq)

    # Compute step sizes
    h = np.zeros(n_params)
    for j in range(n_theta):
        h[j] = eps * max(abs(theta[j]), 1.0)
    h[n_theta] = eps * max(abs(sigma), 1.0)

    # Evaluate deviance at center
    d0 = deviance(theta, sigma)

    # Evaluate at single perturbations
    d_plus = np.zeros(n_params)
    d_minus = np.zeros(n_params)

    for j in range(n_theta):
        tp = theta.copy()
        tp[j] += h[j]
        d_plus[j] = deviance(tp, sigma)

        tm = theta.copy()
        tm[j] -= h[j]
        lb = _theta_lower_bound_at_index(j, specs)
        if tm[j] < lb:
            tm[j] = lb
        d_minus[j] = deviance(tm, sigma)

    d_plus[n_theta] = deviance(theta, sigma + h[n_theta])
    d_minus[n_theta] = deviance(theta, sigma - h[n_theta])

    # Build Hessian
    H = np.zeros((n_params, n_params), dtype=np.float64)

    # Diagonal
    for j in range(n_params):
        H[j, j] = (d_plus[j] - 2.0 * d0 + d_minus[j]) / (h[j] ** 2)

    # Off-diagonal (symmetric)
    for j in range(n_params):
        for l in range(j + 1, n_params):
            # Build perturbed theta and sigma for each of the 4 corners
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
