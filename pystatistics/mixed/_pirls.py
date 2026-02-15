"""
Penalized Iteratively Reweighted Least Squares (PIRLS) for GLMM.

For a GLMM with given θ (and hence Λ_θ), PIRLS iteratively finds the
conditional modes of the random effects by solving a sequence of penalized
weighted least squares problems.

This is the inner loop of GLMM estimation. The outer loop optimizes θ
to minimize the Laplace-approximated deviance.

References:
    Bates, D., Maechler, M., Bolker, B., & Walker, S. (2015).
    Fitting Linear Mixed-Effects Models Using lme4.
    Journal of Statistical Software, 67(1), Section 3.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from pystatistics.mixed._pls import solve_pls, PLSResult


@dataclass(frozen=True)
class PIRLSResult:
    """Result from PIRLS convergence.

    Attributes:
        pls: The final PLS result (contains beta, u, b, etc.).
        mu: Fitted values on the response scale (n,).
        eta: Linear predictor Xβ + Zb (n,).
        deviance: Family deviance at convergence.
        converged: Whether PIRLS converged.
        n_iter: Number of PIRLS iterations.
    """
    pls: PLSResult
    mu: NDArray
    eta: NDArray
    deviance: float
    converged: bool
    n_iter: int


def solve_pirls(
    X: NDArray,
    Z: NDArray,
    y: NDArray,
    Lambda: NDArray,
    family,  # regression.families.Family
    tol: float = 1e-8,
    max_iter: int = 25,
) -> PIRLSResult:
    """Penalized IRLS for GLMM (inner loop).

    For given θ (hence Λ), finds conditional modes of random effects b
    and fixed effects β by iterating:

    1. Compute working response: z = η + (y - μ) / (dμ/dη)
    2. Compute working weights: w = (dμ/dη)² / V(μ)
    3. Solve penalized WLS: minimize ‖√W(z - Xβ - ZΛu)‖² + ‖u‖²
    4. Update: η = Xβ + Zb, μ = g⁻¹(η)
    5. Check convergence on deviance change

    Args:
        X: Fixed effects design matrix (n, p).
        Z: Random effects design matrix (n, q).
        y: Response vector (n,).
        Lambda: Relative covariance factor (q, q).
        family: GLM family object with link, linkinv, mu_eta, variance, deviance.
        tol: Convergence tolerance on relative deviance change.
        max_iter: Maximum PIRLS iterations.

    Returns:
        PIRLSResult with converged estimates.
    """
    link = family.link
    n = len(y)

    # Initialize
    mu = family.initialize(y)
    eta = link.link(mu)

    wt = np.ones(n, dtype=np.float64)
    dev_old = family.deviance(y, mu, wt)
    converged = False
    pls_result = None

    for iteration in range(1, max_iter + 1):
        # Working response and weights
        mu_eta_val = link.mu_eta(eta)
        z = eta + (y - mu) / mu_eta_val
        w = (mu_eta_val ** 2) / family.variance(mu)
        w = np.maximum(w, 1e-10)

        # Solve penalized WLS
        pls_result = solve_pls(X, Z, z, Lambda, weights=w, reml=False)

        # Update linear predictor and mean
        eta = X @ pls_result.beta + Z @ pls_result.b
        mu = link.linkinv(eta)

        # Check convergence
        dev_new = family.deviance(y, mu, wt)
        if abs(dev_new - dev_old) / (abs(dev_old) + 0.1) < tol:
            converged = True
            dev_old = dev_new
            break
        dev_old = dev_new

    if pls_result is None:
        raise RuntimeError("PIRLS failed to produce a result")

    return PIRLSResult(
        pls=pls_result,
        mu=mu,
        eta=eta,
        deviance=dev_old,
        converged=converged,
        n_iter=iteration if 'iteration' in dir() else 0,
    )
