"""
Profile-likelihood confidence intervals for GLM coefficients (R's
``confint.glm`` / ``MASS:::confint.glm``).

For coefficient ``beta_j`` the profile deviance ``D(b)`` is the residual deviance
of the model refit with ``beta_j`` held fixed at ``b`` (the other coefficients
re-optimized). The ``(1 - alpha)`` interval is the set of ``b`` where

    D(b) - D_hat  <=  q

with ``q = z_{1-alpha/2}^2`` for a fixed-dispersion family (binomial/poisson) and
``q = phi * F_{1, df}(1-alpha)`` for an estimated-dispersion family — matching
R's profiling, which scales the deviance drop by the estimated dispersion and
compares to the t/F quantile. The endpoints are found by expanding a bracket
outward from ``beta_hat_j`` (scaled by the Wald SE) and root-finding.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import brentq

from pystatistics.core.exceptions import ValidationError


def _profile_deviance(model, j: int, b: float) -> float:
    """Residual deviance of the fit with coefficient ``j`` fixed at ``b``.

    Achieved with the standard offset trick: move ``x_j * b`` into the offset and
    fit the remaining columns.
    """
    from pystatistics.regression.solvers import fit
    X = np.asarray(model._design.X, dtype=np.float64)
    y = np.asarray(model._design.y, dtype=np.float64)
    family = model._result.info.get('family')
    n, p = X.shape
    keep = [k for k in range(p) if k != j]
    offset = X[:, j] * b
    if not keep:
        # No free coefficients left: deviance at mu = g^{-1}(offset).
        link = family.link
        mu = link.linkinv(offset)
        return float(family.deviance(y, mu, np.ones(n)))
    sub = fit(X[:, keep], y, family=family, offset=offset)
    return float(sub.deviance if family is not None else sub.rss)


def profile_conf_int(model, conf_level: float = 0.95) -> NDArray:
    """Profile-likelihood confidence intervals for all coefficients.

    Returns a ``(p, 2)`` array of ``[lower, upper]`` on the coefficient scale,
    matching R's ``confint(glm)``. Falls back to the Wald interval for a
    coefficient whose profile cannot be bracketed (e.g. complete separation).
    """
    if not (0.0 < conf_level < 1.0):
        raise ValidationError(f"conf_level must be in (0, 1), got {conf_level}")

    beta = np.asarray(model.coefficients, dtype=np.float64)
    se = np.asarray(model.standard_errors, dtype=np.float64)
    family = model._result.info.get('family')
    p = len(beta)

    if family is not None and family.dispersion_is_fixed:
        d_hat = float(model.deviance)
        disp = 1.0
        q = stats.norm.ppf((1.0 + conf_level) / 2.0) ** 2
    else:
        # Estimated dispersion (gaussian/gamma/quasi/OLS): scale by phi_hat and
        # compare to the F(1, df_resid) quantile (R's profiling convention).
        d_hat = float(model.deviance if family is not None else model.rss)
        disp = (model.dispersion if family is not None
                else model.residual_std_error ** 2)
        df_resid = model._result.params.df_residual
        q = disp * float(stats.f.ppf(conf_level, 1, df_resid))

    out = np.empty((p, 2), dtype=np.float64)
    for j in range(p):
        out[j] = _one_coef_interval(model, j, beta[j], se[j], d_hat, q)
    return out


def _one_coef_interval(model, j, bhat, se_j, d_hat, q):
    """Bracket-and-solve the two profile endpoints for coefficient ``j``."""
    def g(b):
        return (_profile_deviance(model, j, b) - d_hat) - q

    if not np.isfinite(se_j) or se_j <= 0:
        return np.array([np.nan, np.nan])

    lower = _solve_side(g, bhat, -se_j)
    upper = _solve_side(g, bhat, +se_j)
    # Wald fallback per side if the profile could not be bracketed.
    z = np.sqrt(q) if q > 0 else 0.0
    if lower is None:
        lower = bhat - (z / max(np.sqrt(q), 1e-12)) * se_j if False else bhat - z * se_j
    if upper is None:
        upper = bhat + z * se_j
    return np.array([lower, upper])


def _solve_side(g, bhat, step):
    """Find the root of ``g`` on the side ``sign(step)`` of ``bhat``.

    ``g(bhat) = -q < 0``; expand outward until ``g`` turns positive, then bisect.
    """
    b0 = bhat
    b1 = bhat + step
    g0 = g(b0)
    for _ in range(60):
        g1 = g(b1)
        if not np.isfinite(g1):
            # Overstepped into an infeasible region; pull the far end back in.
            b1 = 0.5 * (b0 + b1)
            if abs(b1 - b0) < 1e-10:
                return None
            continue
        if g1 > 0:
            return float(brentq(g, b0, b1, xtol=1e-8, rtol=1e-10))
        b0, g0 = b1, g1
        b1 = bhat + (b1 - bhat) * 1.6
    return None
