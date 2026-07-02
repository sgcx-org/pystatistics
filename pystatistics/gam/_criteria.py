"""Smoothing-parameter selection criteria and the outer optimizer.

Criteria (matching R mgcv's conventions, each verified numerically against
mgcv 1.9-3 at fixed smoothing parameters):

- ``GCV``  (scale unknown):  n * D / (n - edf)^2            [mgcv 'GCV.Cp']
- ``UBRE`` (scale known):    D/n - phi + 2*edf*phi/n        [mgcv 'GCV.Cp']
- ``REML`` (Laplace, Wood 2011):
      V = -l(beta) + pen/(2*phi)
          + [log|A/phi| - log|S_lambda/phi|_+]/2 - (M_p/2) log(2*pi)
  with phi = 1 for fixed-dispersion families and phi profiled analytically,
  phi = (D + pen)/(n - M_p), for the Gaussian-identity model. Verified to
  0 (gaussian) / 6e-11 (poisson) against mgcv's reported REML score.

``method='GCV'`` follows mgcv ``GCV.Cp`` semantics: GCV when the scale is
free, UBRE when it is fixed — selecting UBRE for a Poisson/binomial GAM is
what mgcv's default does, not a substitution.

The outer search minimises over log(lambda) with L-BFGS-B and finite
differences. The 4.5.x objective was garbage in exactly the small-lambda
regime the optimizer probes (unconstrained singular design + normal-equations
EDF); on the stable QR path the criterion surface is smooth, so a
quasi-Newton search with a good starting point converges reliably.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from pystatistics.core.exceptions import ValidationError
from pystatistics.gam._edf import influence_matrix, logdet_penalized, total_edf
from pystatistics.gam._pirls import (
    PenaltyRoot,
    PirlsFit,
    fit_fixed_lambda,
    reduce_wls,
)

if TYPE_CHECKING:
    from pystatistics.regression.families import Family

_LOG2PI = float(np.log(2.0 * np.pi))
_BOUND_HALF_WIDTH = 15.0


def _is_gauss_identity(family: Family) -> bool:
    return family.name == "gaussian" and family.link.name == "identity"


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def gcv_score(deviance: float, n: int, edf: float) -> float:
    """GCV = n * D / (n - edf)^2 (lower is better)."""
    denom = n - edf
    if denom <= 0.0:
        return np.inf
    return float(n) * deviance / (denom * denom)


def ubre_score(deviance: float, n: int, edf: float, scale: float) -> float:
    """UBRE / scaled AIC for known-scale families (mgcv GCV.Cp branch)."""
    return deviance / n - scale + 2.0 * edf * scale / n


def reml_score(
    fit: PirlsFit,
    y: NDArray[np.floating[Any]],
    family: Family,
    roots: list[PenaltyRoot],
    lambdas: NDArray[np.floating[Any]],
) -> float:
    """Laplace REML criterion (Wood 2011), mgcv-exact conventions.

    Raises:
        ValidationError: for free-dispersion families other than
            Gaussian-identity (mgcv estimates their scale inside REML;
            this implementation does not — use ``method='GCV'``).
    """
    n = y.shape[0]
    p = fit.R.shape[0]
    rank_s = sum(r.rank for r in roots)
    m_p = p - rank_s
    logdet_a = logdet_penalized(fit.R, fit.rank)
    logdet_s = float(sum(
        r.rank * np.log(lam) + r.logdet_pos
        for r, lam in zip(roots, lambdas)
    ))

    if family.dispersion_is_fixed:
        phi = 1.0
        wt = np.ones(n, dtype=np.float64)
        neg_ll = -family.log_likelihood(y, fit.mu, wt, phi)
        return float(
            neg_ll + fit.penalty / 2.0
            + (logdet_a - logdet_s) / 2.0
            - (m_p / 2.0) * _LOG2PI
        )

    if _is_gauss_identity(family):
        d_p = fit.deviance + fit.penalty
        phi = d_p / (n - m_p)
        return float(
            d_p / (2.0 * phi)
            + (n / 2.0) * np.log(2.0 * np.pi * phi)
            + (logdet_a - p * np.log(phi)
               - (logdet_s - rank_s * np.log(phi))) / 2.0
            - (m_p / 2.0) * _LOG2PI
        )

    raise ValidationError(
        f"method='REML' is not supported for family "
        f"'{family.name}' with link '{family.link.name}' (free dispersion "
        f"outside the Gaussian-identity model); use method='GCV'"
    )


def estimate_scale(
    fit: PirlsFit,
    y: NDArray[np.floating[Any]],
    family: Family,
    edf: float,
) -> float:
    """Dispersion estimate matching mgcv's ``sig2`` conventions."""
    n = y.shape[0]
    if family.dispersion_is_fixed:
        return 1.0
    if _is_gauss_identity(family):
        return fit.deviance / max(n - edf, 1.0)
    # Free dispersion, non-gaussian link: Pearson estimator (mgcv).
    var = np.maximum(family.variance(fit.mu), 1e-300)
    pearson = float(np.sum((y - fit.mu) ** 2 / var))
    return pearson / max(n - edf, 1.0)


# ---------------------------------------------------------------------------
# Outer optimizer
# ---------------------------------------------------------------------------

def initial_log_lambdas(
    X: NDArray[np.floating[Any]],
    roots: list[PenaltyRoot],
) -> NDArray[np.floating[Any]]:
    """mgcv-style starting values: lambda_j ~ tr(X'X block)/tr(S_j)."""
    out = np.zeros(len(roots), dtype=np.float64)
    for i, r in enumerate(roots):
        s, e = r.block
        tr_x = float(np.sum(X[:, s:e] ** 2))
        tr_s = float(np.sum(r.rows ** 2))  # = tr(S_j)
        out[i] = np.log(max(tr_x, 1e-300) / max(tr_s, 1e-300))
    return out


def select_lambdas(
    y: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
    roots: list[PenaltyRoot],
    family: Family,
    method: str,
    tol: float,
    max_iter: int,
    smooth_names: list[str] | None = None,
) -> tuple[NDArray[np.floating[Any]], bool]:
    """Minimise the requested criterion over log(lambda).

    Returns:
        ``(lambdas, outer_converged)`` — ``outer_converged`` is False when
        the optimizer reports failure or the solution sits on the search
        bound (reported honestly, never silently).

    Raises:
        ValidationError: unknown method (callers validate too, belt and
            braces), or REML with an unsupported family.
    """
    n_smooth = len(roots)
    if n_smooth == 0:
        return np.array([], dtype=np.float64), True

    method_u = method.upper()
    if method_u not in ("GCV", "REML"):
        raise ValidationError(
            f"method must be 'GCV' or 'REML', got {method!r}"
        )

    n = y.shape[0]
    gauss_cache = (
        reduce_wls(X, np.ones(n), y) if _is_gauss_identity(family) else None
    )
    # Warm start: nearby lambdas need ~2 PIRLS steps from the previous
    # evaluation's mu instead of ~6 from family.initialize.
    warm: dict[str, Any] = {"mu": None}

    def objective(log_lam: NDArray[np.floating[Any]]) -> float:
        lam = np.exp(np.asarray(log_lam, dtype=np.float64))
        fit = fit_fixed_lambda(
            y, X, roots, lam, family, tol, max_iter,
            smooth_names=smooth_names, gaussian_cache=gauss_cache,
            mu_start=warm["mu"],
        )
        warm["mu"] = fit.mu
        h = influence_matrix(fit.R, fit.R_x, fit.piv, fit.rank)
        edf = total_edf(h)
        if method_u == "REML":
            return reml_score(fit, y, family, roots, lam)
        if family.dispersion_is_fixed:
            return ubre_score(fit.deviance, n, edf, scale=1.0)
        return gcv_score(fit.deviance, n, edf)

    # REML support check up front (fail before burning optimizer time).
    if method_u == "REML" and not (
        family.dispersion_is_fixed or _is_gauss_identity(family)
    ):
        raise ValidationError(
            f"method='REML' is not supported for family "
            f"'{family.name}' with link '{family.link.name}'; "
            f"use method='GCV'"
        )

    log_lam0 = initial_log_lambdas(X, roots)
    bounds = [
        (lo - _BOUND_HALF_WIDTH, lo + _BOUND_HALF_WIDTH) for lo in log_lam0
    ]
    # eps: the FD step must sit well above the inner P-IRLS convergence
    # noise floor (~tol relative on the criterion) or the quasi-Newton
    # gradients are noise. 1e-6 in log-lambda space clears a 1e-8 inner tol
    # by ~two orders while staying far below the criterion's O(1) curvature
    # scale. (Gaussian-identity fits are exact solves — no noise floor.)
    result = minimize(
        objective,
        log_lam0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-7, "eps": 1e-6},
    )

    at_bound = any(
        abs(v - lo) < 1e-6 or abs(v - hi) < 1e-6
        for v, (lo, hi) in zip(result.x, bounds)
    )
    # L-BFGS-B can end with success=False (ABNORMAL_..._LNSRCH) when the
    # line search bottoms out on the finite-difference noise floor AFTER
    # reaching the minimum; a small projected gradient means converged.
    grad_small = (
        result.jac is not None
        and np.max(np.abs(np.asarray(result.jac))) < 1e-2
    )
    outer_converged = (bool(result.success) or grad_small) and not at_bound
    return np.exp(result.x), outer_converged
