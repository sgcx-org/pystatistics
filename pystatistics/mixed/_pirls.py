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
import scipy.linalg as sla

from pystatistics.core.exceptions import NumericalError
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


@dataclass(frozen=True)
class PIRLSModeResult:
    """Result from PIRLS that profiles ONLY the random-effects modes u, holding
    the fixed effects β fixed (the Laplace / nAGQ=1 inner loop).

    Unlike :class:`PIRLSResult` (which solves for β and u jointly — the nAGQ=0
    scheme), this finds û = argmax_u of the penalized conditional log-likelihood
    for a GIVEN β, which is what the Laplace approximation to the marginal
    likelihood requires. β is supplied by the OUTER optimizer.

    Attributes:
        u: Spherical random-effects modes û (q,).
        b: Conditional modes b = Λû (q,).
        mu: Fitted mean g⁻¹(Xβ + ZΛû) (n,).
        eta: Linear predictor Xβ + ZΛû (n,).
        weights: Final IRLS weights W = (dμ/dη)² / V(μ) at the mode (n,).
        working_response: Final IRLS working response z = η + (y-μ)/(dμ/dη) (n,).
        L: Cholesky factor of (Λ'Z'WZΛ + I) at the mode (q, q), lower-triangular.
        deviance: Family deviance at the mode.
        converged: Whether the inner PIRLS converged.
        n_iter: Number of inner PIRLS iterations.
    """
    u: NDArray
    b: NDArray
    mu: NDArray
    eta: NDArray
    weights: NDArray
    working_response: NDArray
    L: NDArray
    deviance: float
    converged: bool
    n_iter: int

    @property
    def laplace_deviance(self) -> float:
        """The Laplace-approximated deviance at this mode:

            d = deviance(y, μ̂) + ‖û‖² + log|L|²

        (the objective the OUTER optimizer minimizes over (θ, β))."""
        penalty = float(self.u @ self.u)
        # NUMERICAL GUARD: prevents log(0) if a diagonal underflows.
        log_det_L = 2.0 * float(
            np.sum(np.log(np.maximum(np.diag(self.L), 1e-20))))
        return float(self.deviance) + penalty + log_det_L


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
        raise NumericalError("PIRLS failed to produce a result")

    return PIRLSResult(
        pls=pls_result,
        mu=mu,
        eta=eta,
        deviance=dev_old,
        converged=converged,
        n_iter=iteration if 'iteration' in dir() else 0,
    )


def solve_pirls_u(
    X: NDArray,
    Z: NDArray,
    y: NDArray,
    Lambda: NDArray,
    beta: NDArray,
    family,  # regression.families.Family
    tol: float = 1e-8,
    max_iter: int = 25,
) -> PIRLSModeResult:
    """Penalized IRLS that profiles ONLY the random-effects modes u (β fixed).

    This is the inner loop of the *Laplace* (nAGQ=1) GLMM: for a GIVEN θ (hence
    Λ) and a GIVEN β, it finds the conditional modes û by iterating the penalized
    weighted least squares problem in u alone, with Xβ entering as a fixed offset:

        minimize_u  ‖√W (z - Xβ - ZΛu)‖² + ‖u‖²

    The normal equations are (Λ'Z'WZΛ + I) u = Λ'Z'W (z - Xβ), solved via the
    Cholesky factor L = chol(Λ'Z'WZΛ + I). β is supplied by the OUTER optimizer,
    which minimizes the resulting Laplace deviance over (θ, β) — in contrast to
    :func:`solve_pirls`, which solves for β and u JOINTLY (the cruder nAGQ=0
    scheme where β never enters the outer optimization).

    Args:
        X: Fixed effects design matrix (n, p).
        Z: Random effects design matrix (n, q).
        y: Response vector (n,).
        Lambda: Relative covariance factor Λ (q, q).
        beta: Fixed effects β (p,), held fixed (the outer offset Xβ).
        family: GLM family object with link, linkinv, mu_eta, variance, deviance.
        tol: Convergence tolerance on relative deviance change.
        max_iter: Maximum PIRLS iterations.

    Returns:
        PIRLSModeResult with the modes and the quantities the Laplace deviance
        and the fixed-effect covariance need.
    """
    link = family.link
    n = len(y)
    q = Z.shape[1]

    ZLam = Z @ Lambda                      # (n, q) = ZΛ
    offset = X @ np.asarray(beta, dtype=np.float64)  # Xβ, fixed

    # η is clamped to this range wherever the mean/weights are formed, so a
    # non-canonical link (e.g. Poisson log with exp) cannot overflow while the
    # OUTER optimizer probes a far-from-optimum β. Real fits sit well inside it
    # (|η| well under ~30), so the clamp never binds at the optimum — it only
    # keeps bad probes finite so step-halving can back out of them.
    ETA_CLAMP = 30.0

    def _eval(u_vec):
        """(penalized deviance, eta, mu) at spherical modes u_vec."""
        eta_ = offset + ZLam @ u_vec
        eta_c = np.clip(eta_, -ETA_CLAMP, ETA_CLAMP)
        mu_ = link.linkinv(eta_c)
        pdev = float(family.deviance(y, mu_, wt) + u_vec @ u_vec)
        return pdev, eta_, mu_

    wt = np.ones(n, dtype=np.float64)
    u = np.zeros(q, dtype=np.float64)
    pdev_old, eta, mu = _eval(u)
    converged = False
    L = np.eye(q)
    w = wt
    z = eta

    iteration = 0
    for iteration in range(1, max_iter + 1):
        eta_c = np.clip(eta, -ETA_CLAMP, ETA_CLAMP)
        mu_eta_val = link.mu_eta(eta_c)
        z = eta_c + (y - mu) / mu_eta_val
        w = (mu_eta_val ** 2) / family.variance(mu)
        w = np.maximum(w, 1e-10)

        # Solve for the Newton step in u only, with Xβ as a fixed offset.
        ZLam_w = ZLam * w[:, np.newaxis]           # W ZΛ  (rows scaled)
        LtL = ZLam_w.T @ ZLam + np.eye(q)          # Λ'Z'WZΛ + I
        try:
            L = np.linalg.cholesky(LtL)
        except np.linalg.LinAlgError:
            raise NumericalError(
                "GLMM random-effects system (Λ'Z'WZΛ + I) is not "
                "positive-definite — Cholesky failed. The random-effects "
                "structure is likely too complex for the data (try removing "
                "random slopes or check for groups with very few observations)."
            )
        rhs = ZLam.T @ (w * (z - offset))          # Λ'Z'W (z - Xβ)
        cu = sla.solve_triangular(L, rhs, lower=True)
        u_new = sla.solve_triangular(L.T, cu, lower=False)

        # Step-halving on the PENALIZED deviance: accept the full PIRLS step only
        # if it does not increase (dev + ‖u‖²) and stays finite; otherwise back
        # off toward the current mode. This is what makes the inner loop robust
        # for non-canonical links (matching lme4's PIRLS) — a full step that
        # would diverge (e.g. Poisson η → ∞) is halved instead of crashing.
        step = 1.0
        pdev_new, eta_try, mu_try = _eval(u_new)
        while (not np.isfinite(pdev_new) or pdev_new > pdev_old + 1e-10) \
                and step > 1e-8:
            step *= 0.5
            pdev_new, eta_try, mu_try = _eval(u + step * (u_new - u))
        u = u + step * (u_new - u)
        eta, mu = eta_try, mu_try

        if abs(pdev_new - pdev_old) / (abs(pdev_old) + 0.1) < tol:
            converged = True
            pdev_old = pdev_new
            break
        pdev_old = pdev_new

    # Recompute the weights, working response and L factor at the final mode so
    # the Laplace log|L|² and the fixed-effect covariance are evaluated exactly
    # at (θ, β)'s converged modes (not the penultimate iterate).
    eta_c = np.clip(eta, -ETA_CLAMP, ETA_CLAMP)
    mu_eta_val = link.mu_eta(eta_c)
    z = eta_c + (y - mu) / mu_eta_val
    w = np.maximum((mu_eta_val ** 2) / family.variance(mu), 1e-10)
    LtL = (ZLam * w[:, np.newaxis]).T @ ZLam + np.eye(q)
    L = np.linalg.cholesky(LtL)

    dev_final = float(family.deviance(y, mu, wt))
    b = Lambda @ u
    return PIRLSModeResult(
        u=u, b=b, mu=mu, eta=eta, weights=w, working_response=z, L=L,
        deviance=dev_final, converged=converged, n_iter=iteration,
    )
