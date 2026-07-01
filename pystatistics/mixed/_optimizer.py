"""Robust θ optimization for the profiled LMM deviance.

One job: minimize the profiled REML/ML deviance over the relative-covariance
parameter θ, robustly across the full range of designs — including the extreme
variance-ratio tail (ICC → 1) where the gradient-based primary optimizer is
unreliable.

Primary optimizer: gradient-based L-BFGS-B from each candidate start (the fast
path for the overwhelming majority of designs). Fallback: a bounded
derivative-free Nelder-Mead simplex (a bobyqa-analogue), engaged only when
L-BFGS-B is flagged as suspect, and adopted only if it strictly lowers the
deviance — so well-converged fits are left byte-for-behaviour identical.

Kept separate from :mod:`.solvers` (which owns solver dispatch / result
assembly) so each module does one thing (Rule 3).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, Bounds

from pystatistics.mixed._pls_structured import (
    deviance_structured, deviance_and_grad_structured, has_analytic_gradient,
)


# Deviance improvement (in REML/ML deviance units) required before the
# derivative-free fallback's result is adopted over L-BFGS-B's. Chosen well
# below the real improvements seen in the extreme-ratio tail (~1e-4 .. tens of
# deviance units) yet far above optimizer-tier round-off (~1e-8), so a
# well-converged L-BFGS-B fit is left byte-for-behaviour identical.
_FALLBACK_MIN_IMPROVEMENT = 1e-6

# Relative + absolute perturbation used to probe θ for premature L-BFGS-B
# termination. L-BFGS-B's forward-difference gradient uses an ABSOLUTE step
# (~1.5e-8); at the large θ of the extreme variance-ratio tail (θ can be O(1e3))
# that is a ~1e-11 RELATIVE step — swamped by round-off — so the gradient looks
# zero and L-BFGS-B "succeeds" at a non-stationary point. A relative probe sees
# the true slope the absolute-step gradient misses.
_PROBE_REL = 1e-3
_PROBE_ABS = 1e-4


def _is_stationary(theta, f_theta, ctx, lb):
    """Cheap check that θ is a genuine local minimum of the profiled deviance.

    Probes each coordinate with a scale-aware (relative + absolute) step in both
    directions, projecting onto the bounds. Returns False if any neighbour has a
    strictly lower deviance — the signature of a premature L-BFGS-B stop in the
    ill-scaled large-θ tail, which its absolute-step finite-difference gradient
    cannot see. Costs 2·dim deviance evaluations; only runs once per fit.
    """
    for i in range(len(theta)):
        step = _PROBE_REL * abs(theta[i]) + _PROBE_ABS
        for direction in (step, -step):
            cand = theta.copy()
            cand[i] = max(cand[i] + direction, lb[i])
            if cand[i] == theta[i]:
                continue
            if deviance_structured(cand, ctx) < f_theta - _FALLBACK_MIN_IMPROVEMENT:
                return False
    return True


def optimize_theta(ctx, starts, bounds, lb, max_iter, tol):
    """Minimize the profiled deviance over θ, with a derivative-free fallback.

    Primary optimizer: gradient-based L-BFGS-B from each candidate start (the
    fast path for the overwhelming majority of designs), keeping the best.

    Fallback (bobyqa-analogue): in the extreme variance-ratio tail (ICC → 1,
    residual variance orders of magnitude below the RE variance) the relative
    RE scale θ is very large and the profiled deviance is flat and ill-scaled.
    There L-BFGS-B's line search either terminates ABNORMALly (non-convergence)
    OR — because its forward-difference gradient uses an absolute step that is a
    negligible *relative* step at large θ — stops and reports success at a
    non-stationary point (a silent premature stop). lme4's derivative-free
    bobyqa handles both. We therefore run the fallback when L-BFGS-B did not
    converge OR a cheap scale-aware stationarity probe (:func:`_is_stationary`)
    flags a premature stop, restarting a bounded Nelder-Mead simplex from the
    best θ found. We adopt its result only if it strictly lowers the deviance —
    so the fallback can only move the fit toward the global optimum, never
    regress a fit L-BFGS-B already solved (well-converged fits are untouched).

    Args:
        ctx: Structured context for ``deviance_structured``.
        starts: Candidate starting θ vectors for L-BFGS-B.
        bounds: L-BFGS-B bounds (list of (lb, None) tuples).
        lb: Lower-bound array for θ (upper bounds are +inf).
        max_iter: Maximum iterations for L-BFGS-B.
        tol: Convergence tolerance.

    Returns:
        A scipy OptimizeResult (from whichever optimizer is adopted).
    """
    # Use the exact analytic θ-gradient where available (the single grouping-
    # factor / batched path) so L-BFGS-B costs one deviance evaluation per step
    # instead of the 2·dim(θ)+1 a finite-difference gradient needs (~2.3× fewer
    # evals; the A.3 optimization). The crossed / nested (sparse) path has no
    # analytic gradient yet and falls back to L-BFGS-B's finite differences.
    use_grad = has_analytic_gradient(ctx)
    if use_grad:
        objective, jac = (lambda th, c: deviance_and_grad_structured(th, c)), True
    else:
        objective, jac = deviance_structured, None

    best = None
    for start in starts:
        res = minimize(
            objective,
            start,
            args=(ctx,),
            method='L-BFGS-B',
            jac=jac,
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol * 10},
        )
        if best is None or res.fun < best.fun:
            best = res

    # Fast path: L-BFGS-B converged to a genuine stationary point.
    if best.success and _is_stationary(best.x, best.fun, ctx, lb):
        return best

    # Suspect fit (loud non-convergence or a silent premature stop in the
    # extreme-ratio tail): retry with a bounded derivative-free simplex from the
    # best θ found so far.
    nm_bounds = Bounds(np.asarray(lb, dtype=np.float64),
                       np.full(len(lb), np.inf))
    fallback = minimize(
        deviance_structured,
        best.x,
        args=(ctx,),
        method='Nelder-Mead',
        bounds=nm_bounds,
        options={'maxiter': max_iter * 100, 'xatol': tol, 'fatol': tol},
    )
    # Adopt only if it strictly improved the deviance beyond round-off.
    if fallback.success and fallback.fun < best.fun - _FALLBACK_MIN_IMPROVEMENT:
        return fallback
    return best
