"""Robust outer optimization for the GLMM Laplace deviance.

One job: minimize a GLMM deviance objective over its parameter vector robustly,
across designs where the gradient-based primary optimizer is unreliable — in
particular the case where L-BFGS-B overshoots an interior variance optimum to the
θ=0 boundary and "succeeds" at a clearly-suboptimal point (a silent
variance-collapse). This is the GLMM analogue of :mod:`._optimizer` (which does
the same for the LMM profiled deviance) and applies the same proven recipe.

Primary optimizer: gradient-based L-BFGS-B (the fast path for the overwhelming
majority of designs). Fallback: a bounded-by-projection derivative-free
Nelder-Mead simplex, engaged only when L-BFGS-B is flagged suspect (non-success,
or a scale-aware stationarity probe finds a strictly-lower neighbour — the
signature of the boundary overshoot), and adopted only if it STRICTLY lowers the
deviance. So a well-converged L-BFGS-B fit is left byte-for-behaviour identical;
only fragile fits are rescued.

Kept separate from :mod:`.solvers` (solver dispatch / result assembly) so each
module does one thing (Rule 3).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

# Deviance improvement required before the derivative-free fallback is adopted
# over L-BFGS-B — well below real rescues (~1e-4 .. tens of deviance units) yet
# far above optimizer-tier round-off, so a converged fit stays identical.
_FALLBACK_MIN_IMPROVEMENT = 1e-6

# Scale-aware probe step for detecting a premature / boundary L-BFGS-B stop.
_PROBE_REL = 1e-3
_PROBE_ABS = 1e-4


def _is_stationary(fun, x, fx, lb) -> bool:
    """Cheap check that ``x`` is a genuine local min: probe each coordinate with
    a scale-aware step in both directions (projected onto the lower bounds).
    Returns False if any neighbour is strictly lower — the signature of a
    boundary overshoot the absolute-step L-BFGS-B gradient can miss. Costs
    2·dim deviance evaluations; runs once per fit."""
    for i in range(len(x)):
        step = _PROBE_REL * abs(x[i]) + _PROBE_ABS
        for direction in (step, -step):
            cand = x.copy()
            cand[i] = max(cand[i] + direction, lb[i])
            if cand[i] == x[i]:
                continue
            if fun(cand) < fx - _FALLBACK_MIN_IMPROVEMENT:
                return False
    return True


def robust_minimize(
    fun: Callable[[NDArray], float],
    x0: NDArray,
    bounds: list[tuple[float | None, float | None]],
    *,
    max_iter: int,
    tol: float,
) -> tuple[NDArray, float, bool]:
    """Minimize ``fun`` over ``x`` with L-BFGS-B, rescued by a Nelder-Mead
    fallback when the primary optimizer stops at a suspect (non-stationary /
    boundary-collapsed) point.

    Returns ``(x_hat, f_hat, converged)``.
    """
    lb = np.array([(-np.inf if b[0] is None else b[0]) for b in bounds])

    res = minimize(fun, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": max_iter, "ftol": tol, "gtol": tol * 10})
    best_x = np.asarray(res.x, dtype=np.float64)
    best_f = float(res.fun)
    converged = bool(res.success)

    if not converged or not _is_stationary(fun, best_x, best_f, lb):
        # Bounded-by-projection Nelder-Mead from the L-BFGS-B point. Diagonal θ
        # entries are variance factors (variance = θ²), so the objective is even
        # in their sign; projecting the simplex result onto the lower bounds
        # keeps it feasible without distorting a genuine optimum.
        nm = minimize(fun, best_x, method="Nelder-Mead",
                      options={"xatol": 1e-7, "fatol": tol,
                               "maxiter": max_iter * 20})
        nm_x = np.maximum(np.asarray(nm.x, dtype=np.float64), lb)
        nm_f = float(fun(nm_x))
        if nm_f < best_f - _FALLBACK_MIN_IMPROVEMENT:
            best_x, best_f, converged = nm_x, nm_f, bool(nm.success)

    return best_x, best_f, converged
