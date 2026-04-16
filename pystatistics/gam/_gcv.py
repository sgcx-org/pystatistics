"""
Smoothing parameter selection for Generalized Additive Models.

Provides GCV (Generalized Cross-Validation) and REML (Restricted
Maximum Likelihood) criteria, plus an optimizer that searches over
log-lambda space using L-BFGS-B.

References:
    Wood, S. N. (2004). Stable and efficient multiple smoothing parameter
        estimation for GAMs. JASA 99(467), 673--686.
    Wood, S. N. (2011). Fast stable restricted maximum likelihood and
        marginal likelihood estimation of semiparametric GLMs. JRSS-B
        73(1), 3--36.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from pystatistics.gam._fit import (
    _compute_hat_matrix_trace,
    _fit_gam_fixed_lambda,
)

if TYPE_CHECKING:
    from pystatistics.regression.families import Family


# ------------------------------------------------------------------
# GCV score
# ------------------------------------------------------------------

def _gcv_score(
    log_lambdas: NDArray,
    y: NDArray,
    X_aug: NDArray,
    S_penalties: list[NDArray],
    family: Family,
    parametric_cols: int,
    tol: float,
    max_iter: int,
) -> float:
    """Compute the GCV score for given smoothing parameters.

    The Generalized Cross-Validation criterion is::

        GCV = n * deviance / (n - edf_total)^2

    This penalises model complexity through the effective degrees of
    freedom, favouring models that balance fit and smoothness.

    Args:
        log_lambdas: Log-scale smoothing parameters ``(n_smooths,)``.
        y: Response ``(n,)``.
        X_aug: Augmented design ``(n, p)``.
        S_penalties: Padded penalties (one per smooth).
        family: GLM family.
        parametric_cols: Leading parametric columns.
        tol: P-IRLS convergence tolerance.
        max_iter: P-IRLS maximum iterations.

    Returns:
        GCV score (lower is better).
    """
    lambdas = np.exp(np.asarray(log_lambdas, dtype=np.float64))
    n = y.shape[0]

    beta, mu, eta, W, deviance, n_iter, converged = _fit_gam_fixed_lambda(
        y, X_aug, S_penalties, lambdas, family, parametric_cols, tol, max_iter,
    )

    edf_total = _compute_hat_matrix_trace(X_aug, W, S_penalties, lambdas)
    denom = n - edf_total

    # Guard against degenerate denominator
    if denom <= 0.0:
        return 1e20

    gcv = float(n) * deviance / (denom * denom)
    return gcv


# ------------------------------------------------------------------
# REML score
# ------------------------------------------------------------------

def _reml_score(
    log_lambdas: NDArray,
    y: NDArray,
    X_aug: NDArray,
    S_penalties: list[NDArray],
    family: Family,
    parametric_cols: int,
    tol: float,
    max_iter: int,
) -> float:
    """Compute a simplified REML criterion for given smoothing parameters.

    Uses a Laplace approximation to the restricted log-likelihood::

        REML ~ deviance/scale
               + sum_j(edf_j * log(lambda_j))
               + log|X'WX + sum lambda_j S_j|
               - log|X'WX|

    This is a working approximation that is effective for smooth
    parameter selection; see Wood (2011) for the full derivation.

    Args:
        log_lambdas: Log-scale smoothing parameters ``(n_smooths,)``.
        y: Response ``(n,)``.
        X_aug: Augmented design ``(n, p)``.
        S_penalties: Padded penalties (one per smooth).
        family: GLM family.
        parametric_cols: Leading parametric columns.
        tol: P-IRLS convergence tolerance.
        max_iter: P-IRLS maximum iterations.

    Returns:
        REML criterion value (lower is better).
    """
    lambdas = np.exp(np.asarray(log_lambdas, dtype=np.float64))
    n = y.shape[0]

    beta, mu, eta, W, deviance, n_iter, converged = _fit_gam_fixed_lambda(
        y, X_aug, S_penalties, lambdas, family, parametric_cols, tol, max_iter,
    )

    # Estimate scale for Gaussian-like families
    edf_total = _compute_hat_matrix_trace(X_aug, W, S_penalties, lambdas)
    scale = max(deviance / max(n - edf_total, 1.0), 1e-20)

    # Weighted cross-product matrices
    XtW = X_aug.T * W[np.newaxis, :]
    XtWX = XtW @ X_aug

    penalty = np.zeros_like(XtWX)
    for lam, S in zip(lambdas, S_penalties):
        penalty += lam * S

    A = XtWX + penalty

    # Log-determinants via Cholesky (numerically stable)
    try:
        sign_a, logdet_a = np.linalg.slogdet(A)
        sign_b, logdet_b = np.linalg.slogdet(XtWX + 1e-10 * np.eye(XtWX.shape[0]))
    except np.linalg.LinAlgError:
        return 1e20

    if sign_a <= 0 or sign_b <= 0:
        return 1e20

    reml = deviance / scale + logdet_a - logdet_b

    return float(reml)


# ------------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------------

def select_smoothing_parameters(
    y: NDArray,
    X_aug: NDArray,
    S_penalties: list[NDArray],
    family: Family,
    parametric_cols: int,
    method: str,
    tol: float,
    max_iter: int,
    n_smooths: int,
) -> NDArray:
    """Select optimal smoothing parameters via GCV or REML.

    Minimises the chosen criterion over ``log(lambda)`` space using
    L-BFGS-B with bounded search in ``[-10, 15]`` (i.e. lambda in
    roughly ``[5e-5, 3e6]``).

    Starting values are ``lambda = 1`` for all smooth terms
    (``log_lambda = 0``).

    Args:
        y: Response ``(n,)``.
        X_aug: Augmented design ``(n, p)``.
        S_penalties: Padded penalties.
        family: GLM family.
        parametric_cols: Leading parametric columns.
        method: ``'GCV'`` or ``'REML'``.
        tol: P-IRLS tolerance.
        max_iter: P-IRLS max iterations.
        n_smooths: Number of smooth terms.

    Returns:
        Optimal lambda values ``(n_smooths,)``.
    """
    if n_smooths == 0:
        return np.array([], dtype=np.float64)

    if method.upper() == "GCV":
        objective = _gcv_score
    elif method.upper() == "REML":
        objective = _reml_score
    else:
        from pystatistics.core.exceptions import ValidationError
        raise ValidationError(
            f"method must be 'GCV' or 'REML', got {method!r}"
        )

    log_lam0 = np.zeros(n_smooths, dtype=np.float64)
    bounds = [(-10.0, 15.0)] * n_smooths

    result = minimize(
        objective,
        log_lam0,
        args=(y, X_aug, S_penalties, family, parametric_cols, tol, max_iter),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 50, "ftol": 1e-6},
    )

    return np.exp(result.x)
