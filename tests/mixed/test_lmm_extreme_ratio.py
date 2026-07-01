"""Regression tests for the extreme variance-ratio tail (F3).

When the intraclass correlation approaches 1 — the residual variance orders of
magnitude below the between-group variance — the optimal relative RE scale θ is
very large (O(1e2)–O(1e3)) and the profiled deviance is flat and ill-scaled.
There the gradient-based L-BFGS-B primary optimizer can either fail its line
search (loud non-convergence) or, because its forward-difference gradient uses
an absolute step that is a negligible *relative* step at large θ, stop and
report success at a non-stationary point (a silent premature stop). lme4's
derivative-free bobyqa converges to the true optimum in this regime.

These tests pin the fix: the solver must converge to the true global optimum of
the profiled deviance across the extreme-ratio tail — matching lme4 — rather than
returning ``converged=False`` (the 4.5.0 regression) or a silently wrong fit.
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from pystatistics.mixed import lmm
from pystatistics.mixed._random_effects import parse_random_effects
from pystatistics.mixed._pls_structured import (
    build_structured_context, deviance_structured, solve_structured,
)


def _extreme_ratio_data(seed_off: int, resid_sd: float):
    """Random-intercept design with a near-perfect ICC (mirrors the validation
    hard case ``hc3_extreme_variance_ratio``: G=15, ni=6, between sd 10)."""
    rng = np.random.default_rng(20260630 + seed_off)
    G, ni = 15, 6
    n = G * ni
    g = np.repeat(np.arange(G), ni)
    b0 = rng.normal(0.0, 10.0, size=G)
    y = 1.0 + b0[g] + rng.normal(0.0, resid_sd, size=n)
    X = np.ones((n, 1))
    return y, X, g


def _true_global_varcomp(y, X, g):
    """Between-group variance at the true global minimum of the profiled REML
    deviance, found by an exhaustive 1-D scalar search (θ is scalar for a random
    intercept). This is the optimizer-independent reference lme4 also targets."""
    specs = parse_random_effects({'g': g}, None, None, len(y), build_dense=False)
    ctx = build_structured_context(X, y, specs, True)
    res = minimize_scalar(
        lambda t: deviance_structured(np.array([max(t, 1e-9)]), ctx),
        bounds=(1e-6, 1e5), method='bounded', options={'xatol': 1e-9},
    )
    pls = solve_structured(np.array([res.x]), ctx)
    return res.x ** 2 * pls.sigma_sq


def test_hc3_extreme_variance_ratio_converges():
    """The exact validation hard case (seed_off=2, residual sd 0.03, ICC≈0.99999)
    must converge — the 4.5.0 structured solver returned converged=False here
    (L-BFGS-B ABNORMAL line-search termination), a regression vs the 4.4.1 dense
    path which lme4 handles fine."""
    y, X, g = _extreme_ratio_data(seed_off=2, resid_sd=0.03)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # no boundary warning is expected here
        result = lmm(y, X, groups={'g': g})

    assert result.converged, "extreme-ratio fit must converge (F3 regression)"
    # Matches lme4's between-group variance (~130.3); the 4.5.0 regression
    # returned ~107.9 (~17% low).
    truth = _true_global_varcomp(y, X, g)
    np.testing.assert_allclose(truth, 130.305, rtol=1e-3)
    np.testing.assert_allclose(
        result.var_components[0].variance, truth, rtol=1e-3,
    )


@pytest.mark.parametrize("resid_sd", [0.01, 0.03, 0.1, 0.3, 1.0])
@pytest.mark.parametrize("seed_off", [0, 1, 2, 3, 4])
def test_extreme_ratio_sweep_matches_global_optimum(resid_sd, seed_off):
    """Across the residual-sd × seed sweep the solver must always converge AND
    reach the true global optimum of the profiled deviance — no non-convergence
    and no silent premature stop anywhere in the tail."""
    y, X, g = _extreme_ratio_data(seed_off=100 * seed_off + 7, resid_sd=resid_sd)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = lmm(y, X, groups={'g': g})

    assert result.converged, (
        f"non-convergence at resid_sd={resid_sd}, seed_off={seed_off}"
    )
    truth = _true_global_varcomp(y, X, g)
    # Optimizer-tier agreement with the exhaustive-search global optimum.
    np.testing.assert_allclose(
        result.var_components[0].variance, truth, rtol=5e-3,
    )
