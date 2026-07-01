"""Structure-exploiting Penalized IRLS for the GLMM Laplace fit.

The GLMM analogue of :mod:`._pls_structured`: it never materializes the dense
random-effects design or the dense q×q system. Each PIRLS iteration builds the
WEIGHTED structured factor M = Λ'Z'WZΛ + I (batched per-group for a single
grouping factor, sparse for crossed / nested — the same backends the LMM path
uses, extended with IRLS weights) and solves through it, so the cost scales with
the block/sparsity structure rather than O(#groups³).

Two modes, sharing one iterate:

  * ``mode_joint`` (nAGQ=0): solve β and u jointly each step — the fast warm
    start for the Laplace stage.
  * ``mode_given_beta`` (nAGQ=1): profile only u for a fixed β supplied by the
    outer optimizer — the true Laplace inner loop.

Both step-halve on the penalized deviance and clamp η where the mean/weights are
formed, so a far-from-optimum probe (e.g. Poisson η → ∞) is backed out of rather
than overflowing (matching lme4's PIRLS).

References:
    Bates, D., Maechler, M., Bolker, B., & Walker, S. (2015), Sections 3 & 5.4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import scipy.linalg as sla

from pystatistics.core.exceptions import NumericalError
from pystatistics.mixed._pls_structured import (
    StructuredContext, build_weighted_factor,
)

# η is clamped to this range wherever the mean/weights are formed (a real fit
# sits well inside; the clamp only keeps far-off probes finite so step-halving
# can back out of them).
_ETA_CLAMP = 30.0
_PIRLS_MAX_ITER = 50


@dataclass(frozen=True)
class GLMMModeResult:
    """Modes + the quantities the Laplace deviance and fixed-effect SE need."""
    u: NDArray                 # spherical random-effect modes (q,)
    b: NDArray                 # conditional modes Λu (q,)
    beta: NDArray             # fixed effects (p,)
    mu: NDArray               # fitted mean (n,)
    eta: NDArray              # linear predictor Xβ + Zb (n,)
    weights: NDArray          # final IRLS weights (n,)
    logdet_M: float           # log|Λ'Z'WZΛ + I|
    RX: NDArray               # lower Cholesky of the p×p Schur complement
    deviance: float           # family deviance at the mode
    penalty: float            # ‖u‖²
    converged: bool
    n_iter: int

    @property
    def laplace_deviance(self) -> float:
        return float(self.deviance + self.penalty + self.logdet_M)


def _working(family, y, eta):
    """Clamped mean, working response and IRLS weights at η."""
    link = family.link
    eta_c = np.clip(eta, -_ETA_CLAMP, _ETA_CLAMP)
    mu = link.linkinv(eta_c)
    mu_eta = link.mu_eta(eta_c)
    z = eta_c + (y - mu) / mu_eta
    w = np.maximum((mu_eta ** 2) / family.variance(mu), 1e-10)
    return mu, z, w


def _schur(ctx, factor, w):
    """(RX, Minv_a, Minv_B): the p×p Schur Cholesky and the M⁻¹ solves."""
    X = ctx.X
    Minv_a = factor.apply_Minv(factor.a)
    Minv_B = factor.apply_Minv(factor.B)
    RtR = X.T @ (w[:, None] * X) - factor.B.T @ Minv_B
    try:
        RX = np.linalg.cholesky(RtR)
    except np.linalg.LinAlgError:
        raise NumericalError(
            "GLMM fixed-effects system (Schur complement) is singular — "
            "collinear fixed effects. Remove redundant predictors."
        )
    return RX, Minv_a, Minv_B


def _iterate(ctx: StructuredContext, theta: NDArray, family, beta_fixed,
             tol: float, max_iter: int) -> GLMMModeResult:
    """Core PIRLS. ``beta_fixed=None`` → joint (nAGQ=0); else profile u only."""
    X, y = ctx.X, ctx.y
    n, p = X.shape
    joint = beta_fixed is None

    beta = np.zeros(p) if joint else np.asarray(beta_fixed, dtype=np.float64)
    # Start from an intercept-free mean init; the mode is unique so the start
    # affects only iteration count.
    mu = family.initialize(y)
    eta = family.link.link(mu)
    u = None
    factor = None
    w = np.ones(n)
    pdev_old = float(family.deviance(y, family.link.linkinv(
        np.clip(eta, -_ETA_CLAMP, _ETA_CLAMP)), np.ones(n)))
    converged = False

    it = 0
    for it in range(1, max_iter + 1):
        mu, z, w = _working(family, y, eta)
        factor = build_weighted_factor(ctx, theta, z, w)

        if joint:
            RX, Minv_a, Minv_B = _schur(ctx, factor, w)
            rhs = X.T @ (w * z) - factor.B.T @ Minv_a
            tmp = sla.solve_triangular(RX, rhs, lower=True)
            beta_new = sla.solve_triangular(RX.T, tmp, lower=False)
            u_new = Minv_a - Minv_B @ beta_new
        else:
            beta_new = beta
            u_new = factor.apply_Minv(factor.a - factor.B @ beta)

        def _pdev(bt, uu):
            eta_try = X @ bt + factor.z_apply(factor.lambda_apply(uu))
            mu_try = family.link.linkinv(np.clip(eta_try, -_ETA_CLAMP, _ETA_CLAMP))
            return (float(family.deviance(y, mu_try, np.ones(n)) + uu @ uu),
                    eta_try, mu_try)

        # Step-halving on the penalized deviance.
        prev_beta = beta
        prev_u = u if u is not None else np.zeros_like(u_new)
        step = 1.0
        pdev_new, eta_new, mu_new = _pdev(beta_new, u_new)
        while (not np.isfinite(pdev_new) or pdev_new > pdev_old + 1e-10) \
                and step > 1e-8:
            step *= 0.5
            bt = prev_beta + step * (beta_new - prev_beta)
            uu = prev_u + step * (u_new - prev_u)
            pdev_new, eta_new, mu_new = _pdev(bt, uu)
            beta_new, u_new = bt, uu

        beta, u, eta, mu = beta_new, u_new, eta_new, mu_new
        if abs(pdev_new - pdev_old) / (abs(pdev_old) + 0.1) < tol:
            converged = True
            pdev_old = pdev_new
            break
        pdev_old = pdev_new

    # Final quantities at the converged mode (recompute weights/factor so the
    # log-det and Schur are exactly at the mode).
    mu, z, w = _working(family, y, eta)
    factor = build_weighted_factor(ctx, theta, z, w)
    RX, _, _ = _schur(ctx, factor, w)
    b = factor.lambda_apply(u)
    dev = float(family.deviance(y, mu, np.ones(n)))
    return GLMMModeResult(
        u=u, b=b, beta=beta, mu=mu, eta=eta, weights=w,
        logdet_M=float(factor.logdet_M), RX=RX, deviance=dev,
        penalty=float(u @ u), converged=converged, n_iter=it,
    )


def mode_joint(ctx, theta, family, *, tol=1e-8, max_iter=_PIRLS_MAX_ITER):
    """nAGQ=0 joint (β, u) PIRLS — the warm start for the Laplace stage."""
    return _iterate(ctx, theta, family, None, tol, max_iter)


def mode_given_beta(ctx, theta, beta, family, *, tol=1e-8,
                    max_iter=_PIRLS_MAX_ITER):
    """nAGQ=1 inner loop — profile the random-effect modes u for a fixed β."""
    return _iterate(ctx, theta, family, beta, tol, max_iter)


def profiled_deviance(ctx, theta, family) -> float:
    """nAGQ=0 profiled Laplace deviance over θ (β solved jointly)."""
    return mode_joint(ctx, theta, family).laplace_deviance


def laplace_deviance(ctx, params, family, n_theta) -> float:
    """nAGQ=1 Laplace deviance over the joint [θ, β] vector."""
    theta = params[:n_theta]
    beta = params[n_theta:]
    return mode_given_beta(ctx, theta, beta, family).laplace_deviance
