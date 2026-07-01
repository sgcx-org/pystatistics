"""CPU (float64) reference solver for the low-rank / GRM mixed model.

Model:  y = Xβ + g + ε,   g ~ N(0, σ²_g K),  ε ~ N(0, σ²_e I),  K = WW'/M.

Reparametrise g = Zu with Z = W/√M and u ~ N(0, σ²_g I_M); then
V = σ²_e (I + γ ZZ') with γ = σ²_g/σ²_e. Writing c = θ/√M (θ = √γ), the
profiled REML/ML deviance is evaluated in **M-space** via the Woodbury identity,
so the dominant linear algebra is the dense M×M Gram G = c²W'W + I and its
Cholesky — never the n×n matrix V. This is the reference path: pure numpy
float64, matching R/GCTA-style REML to machine precision and serving as the
correctness baseline the GPU backend is validated against.

References:
    Yang et al. (2011), GCTA (the GREML variance-component model);
    Bates et al. (2015) §5.4 (the profiled-deviance / Woodbury structure).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import scipy.linalg as sla
from scipy.optimize import minimize_scalar

from pystatistics.core.exceptions import NumericalError


@dataclass(frozen=True)
class GRMFit:
    """Raw primitives from a GRM solve at the optimal θ (backend-agnostic).

    Both the CPU and GPU backends return this; the public ``grm_lmm`` assembles
    the user-facing GRMParams (SEs, heritability, fit statistics) from it.
    """
    theta: float
    beta: NDArray
    u: NDArray
    sigma_e2: float
    pwrss: float
    logdet_G: float       # log|c²W'W + I| = log|V| in σ²_e units
    RX: NDArray           # p×p Cholesky of X'V⁻¹X
    genetic_values: NDArray
    fitted: NDArray
    residuals: NDArray
    converged: bool
    n_iter: int


def grm_solve_cpu(theta: float, W: NDArray, X: NDArray, y: NDArray,
                  reml: bool) -> dict:
    """Solve the profiled GRM system at a fixed θ (CPU float64).

    Returns a dict with beta, u, sigma_e2, pwrss, logdet_G, RX, genetic_values,
    fitted, residuals. Raises NumericalError if a Cholesky fails.
    """
    n, M = W.shape
    p = X.shape[1]
    c = theta / np.sqrt(M)
    ZL = c * W                                     # (n, M)

    G = ZL.T @ ZL + np.eye(M)                       # (M, M) dense Gram
    try:
        L = np.linalg.cholesky(G)
    except np.linalg.LinAlgError:
        raise NumericalError(
            "GRM Gram (c²W'W + I) is not positive definite — Cholesky failed. "
            "Check the low-rank factor W for degeneracy."
        )

    a = ZL.T @ y                                    # (M,)
    B = ZL.T @ X                                    # (M, p)
    # Full M⁻¹ solves via the Cholesky factor.
    m_y = sla.cho_solve((L, True), a)               # (M,)
    Wmat = sla.cho_solve((L, True), B)              # (M, p)  = G⁻¹ B

    RtR = X.T @ X - B.T @ Wmat                      # Schur complement = X'V⁻¹X
    rhs = X.T @ y - B.T @ m_y
    try:
        RX = np.linalg.cholesky(RtR)
        tmp = sla.solve_triangular(RX, rhs, lower=True)
        beta = sla.solve_triangular(RX.T, tmp, lower=False)
    except np.linalg.LinAlgError:
        raise NumericalError(
            "Fixed-effects system (X'V⁻¹X) is singular — collinear covariates "
            "in X. Remove redundant columns."
        )

    u = m_y - Wmat @ beta                           # (M,)
    genetic = ZL @ u                                # g = Zu (n,)
    fitted = X @ beta + genetic
    residuals = y - fitted
    pwrss = float(residuals @ residuals + u @ u)
    sigma_e2 = pwrss / (n - p) if reml else pwrss / n

    logdet_G = 2.0 * float(np.sum(np.log(np.maximum(np.diag(L), 1e-300))))

    return {
        "beta": beta, "u": u, "sigma_e2": sigma_e2, "pwrss": pwrss,
        "logdet_G": logdet_G, "RX": RX, "genetic_values": genetic,
        "fitted": fitted, "residuals": residuals,
    }


def grm_deviance_cpu(theta: float, W: NDArray, X: NDArray, y: NDArray,
                     reml: bool) -> float:
    """Profiled REML/ML deviance at θ (the 1-D objective to minimize)."""
    n, M = W.shape
    p = X.shape[1]
    s = grm_solve_cpu(theta, W, X, y, reml)
    log_det_RX = 2.0 * float(np.sum(np.log(np.maximum(np.abs(np.diag(s["RX"])), 1e-300))))
    if reml:
        df = n - p
        return float(s["logdet_G"] + log_det_RX
                     + df * (1.0 + np.log(2.0 * np.pi * s["pwrss"] / df)))
    return float(s["logdet_G"] + n * (1.0 + np.log(2.0 * np.pi * s["pwrss"] / n)))


def grm_fit_cpu(W: NDArray, X: NDArray, y: NDArray, *, reml: bool,
                tol: float, max_iter: int, theta_max: float = 1.0e3) -> GRMFit:
    """Fit the GRM model on the CPU by minimizing the profiled deviance over θ.

    θ ≥ 0 is the scalar √(σ²_g/σ²_e); a bounded 1-D search (Brent) over
    [0, theta_max] locates the REML/ML optimum.
    """
    res = minimize_scalar(
        grm_deviance_cpu, args=(W, X, y, reml),
        bounds=(0.0, theta_max), method="bounded",
        options={"xatol": tol, "maxiter": max_iter},
    )
    theta_hat = float(res.x)
    converged = bool(res.success)
    n_iter = int(getattr(res, "nfev", 0))

    s = grm_solve_cpu(theta_hat, W, X, y, reml)
    return GRMFit(
        theta=theta_hat, beta=s["beta"], u=s["u"], sigma_e2=s["sigma_e2"],
        pwrss=s["pwrss"], logdet_G=s["logdet_G"], RX=s["RX"],
        genetic_values=s["genetic_values"], fitted=s["fitted"],
        residuals=s["residuals"], converged=converged, n_iter=n_iter,
    )
