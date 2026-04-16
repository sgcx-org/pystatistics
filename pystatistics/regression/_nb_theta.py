"""
Profile likelihood estimation of the negative binomial dispersion parameter θ.

Given fitted values μ from a NB GLM, θ is estimated by maximizing the
profile log-likelihood (equivalently, finding the root of its score).

Algorithm matches R's MASS::theta.ml():
    1. Evaluate the profile score function s(θ) = dl/dθ
    2. Find the root via Brent's method over a bracket [lo, hi]

The outer glm.nb() loop in solvers.py iterates:
    θ_new = theta_ml(y, μ_old)  →  refit GLM with θ_new  →  repeat

References:
    Venables & Ripley (2002). Modern Applied Statistics with S, §7.4
    R: MASS::theta.ml, MASS::theta.mm
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import digamma, polygamma


def _nb_profile_score(
    theta: float,
    y: NDArray,
    mu: NDArray,
    wt: NDArray,
) -> float:
    """Score function dl/dθ of the NB profile log-likelihood.

    s(θ) = Σ wt_i * [ψ(y_i + θ) - ψ(θ) + log(θ/(μ_i + θ))
                      + 1 - (y_i + θ)/(μ_i + θ)]

    Args:
        theta: Current dispersion parameter (> 0).
        y: Response values (non-negative integers).
        mu: Fitted values from the current GLM iteration.
        wt: Prior weights (typically all 1).

    Returns:
        Scalar score value. Zero at the MLE of θ.
    """
    return float(np.sum(wt * (
        digamma(y + theta) - digamma(theta)
        + np.log(theta / (mu + theta))
        + 1.0 - (y + theta) / (mu + theta)
    )))


def _nb_profile_info(
    theta: float,
    y: NDArray,
    mu: NDArray,
    wt: NDArray,
) -> float:
    """Observed information -d²l/dθ² for Newton step (used as fallback).

    I(θ) = Σ wt_i * [-ψ'(y_i + θ) + ψ'(θ) - 1/θ + 2/(μ_i + θ)
                      - (y_i + θ)/(μ_i + θ)²]
    """
    return float(np.sum(wt * (
        -polygamma(1, y + theta) + polygamma(1, theta)
        - 1.0 / theta + 2.0 / (mu + theta)
        - (y + theta) / (mu + theta) ** 2
    )))


def theta_ml(
    y: NDArray,
    mu: NDArray,
    wt: NDArray | None = None,
    limit: int = 50,
    tol: float = 1e-8,
) -> float:
    """Estimate θ by maximizing the NB profile log-likelihood.

    Matches R's MASS::theta.ml(). Uses Brent's method to find the root
    of the profile score function. Falls back to moment-based estimate
    if the bracket search fails.

    Args:
        y: Response values (non-negative counts).
        mu: Fitted values from the current GLM iteration.
        wt: Prior weights. Defaults to all ones.
        limit: Maximum Brent iterations.
        tol: Convergence tolerance for theta.

    Returns:
        Estimated theta (> 0).

    Raises:
        pystatistics.core.exceptions.ConvergenceError:
            If the optimization fails to converge.
    """
    from pystatistics.core.exceptions import ConvergenceError

    if wt is None:
        wt = np.ones_like(y)

    # Moment-based starting estimate: θ_mm = mean(μ)² / (var(y) - mean(μ))
    mu_mean = float(np.average(mu, weights=wt))
    y_var = float(np.average((y - mu_mean) ** 2, weights=wt))
    overdispersion = y_var - mu_mean
    if overdispersion <= 0:
        # Data is underdispersed relative to Poisson — theta is effectively ∞
        return 1e6

    theta_mm = mu_mean ** 2 / overdispersion
    # NUMERICAL GUARD: clamp moment estimate to a sensible range
    theta_mm = max(min(theta_mm, 1e6), 1e-4)

    # Try Brent's method on the score function
    # Bracket: search around the moment estimate
    lo = theta_mm / 100.0
    hi = theta_mm * 100.0

    score_lo = _nb_profile_score(lo, y, mu, wt)
    score_hi = _nb_profile_score(hi, y, mu, wt)

    if score_lo * score_hi < 0:
        # Sign change found — Brent will converge
        theta_hat = brentq(
            _nb_profile_score, lo, hi,
            args=(y, mu, wt),
            xtol=tol, maxiter=limit,
        )
        return float(theta_hat)

    # No sign change — widen bracket and retry
    lo = 1e-6
    hi = 1e8
    score_lo = _nb_profile_score(lo, y, mu, wt)
    score_hi = _nb_profile_score(hi, y, mu, wt)

    if score_lo * score_hi < 0:
        theta_hat = brentq(
            _nb_profile_score, lo, hi,
            args=(y, mu, wt),
            xtol=tol, maxiter=limit,
        )
        return float(theta_hat)

    # Score is monotone — data may be Poisson-like (theta → ∞)
    # or extremely overdispersed (theta → 0)
    if score_lo > 0 and score_hi > 0:
        # Score always positive → theta should be larger → near-Poisson
        return 1e6
    if score_lo < 0 and score_hi < 0:
        # Score always negative → theta should be smaller → extreme overdispersion
        raise ConvergenceError(
            "NB theta estimation failed: data appears extremely overdispersed. "
            "The profile score is negative across the entire search range "
            f"[{lo:.2e}, {hi:.2e}]. Consider a different model.",
            details={'score_lo': score_lo, 'score_hi': score_hi,
                     'theta_mm': theta_mm},
        )

    raise ConvergenceError(
        "NB theta estimation failed to find a valid bracket.",
        details={'score_lo': score_lo, 'score_hi': score_hi,
                 'theta_mm': theta_mm},
    )
