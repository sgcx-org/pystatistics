"""
Shared Newton-Raphson driver for the Cox partial likelihood.

Both the single-stratum (``_cox.cox_fit``) and stratified
(``_cox_strata.cox_fit_stratified``) solvers maximise a partial log-likelihood
with the same Newton-Raphson iteration, step-halving, and convergence rule; the
only difference is how the log-likelihood, score, and information are evaluated
at a given ``beta`` (one risk-set sweep vs. a sum over per-stratum sweeps). This
module owns that iteration exactly once so the two paths cannot drift apart.

The caller supplies two closures:

    eval_full(beta)   -> (loglik, score, information)
    eval_loglik(beta) -> loglik

matching the ``(loglik, score, info)`` / ``loglik`` contracts of
``_cox._score_and_information`` / ``_cox._partial_loglik``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

# Largest per-step change in any coefficient. R's coxph applies the same guard
# (see coxph.fit / Ccoxfit6 step-halving) so exp(x @ beta) cannot overflow.
_MAX_STEP = 5.0

# Maximum step-halvings per iteration in the backtracking line search. R's
# coxfit halves the Newton step whenever it decreases the partial log-likelihood;
# this bounds how far it backtracks before giving up on that direction.
_MAX_HALVINGS = 20


@dataclass(frozen=True)
class NewtonResult:
    """Outcome of the Cox Newton-Raphson iteration.

    Attributes
    ----------
    beta : (p,)
        Coefficient estimates at the stopping point.
    null_loglik : float
        Partial log-likelihood at ``beta = 0``.
    model_loglik : float
        Partial log-likelihood at ``beta``.
    information : (p, p)
        Observed information (negative Hessian) at ``beta``.
    score : (p,)
        Score (gradient) at ``beta``. Near a clean optimum this is ~0; a
        component that stays large relative to its coefficient signals a
        coefficient running to +/- infinity (monotone likelihood / separation).
    converged : bool
        Whether the convergence criterion was met before ``max_iter``.
    n_iter : int
        Number of iterations performed.
    """

    beta: NDArray
    null_loglik: float
    model_loglik: float
    information: NDArray
    score: NDArray
    converged: bool
    n_iter: int


def solve_cox_newton(
    p: int,
    eval_full: Callable[[NDArray], tuple[float, NDArray, NDArray]],
    eval_loglik: Callable[[NDArray], float],
    tol: float,
    max_iter: int,
) -> NewtonResult:
    """Maximise a Cox partial log-likelihood by Newton-Raphson.

    Parameters
    ----------
    p : int
        Number of coefficients.
    eval_full : callable
        ``beta -> (loglik, score, information)``.
    eval_loglik : callable
        ``beta -> loglik`` (cheaper than ``eval_full`` when only the value is
        needed for the convergence test).
    tol : float
        Convergence tolerance: ``max|beta_new - beta| < tol``, or a relative
        log-likelihood change below ``tol``.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    NewtonResult
    """
    beta = np.zeros(p, dtype=np.float64)
    null_loglik = eval_loglik(beta)

    converged = False
    n_iter = 0
    loglik_old = null_loglik

    for iteration in range(1, max_iter + 1):
        loglik_cur, score, info_matrix = eval_full(beta)

        # Newton step: beta_new = beta + I^{-1} @ U.
        try:
            step = np.linalg.solve(info_matrix, score)
        except np.linalg.LinAlgError:
            # Singular information — stop; SEs downstream report inf.
            break

        # Cap the raw magnitude so exp(x @ beta) cannot overflow (matches R).
        max_step = np.max(np.abs(step))
        if max_step > _MAX_STEP:
            step = step * (_MAX_STEP / max_step)

        beta_new = beta + step
        loglik_new = eval_loglik(beta_new)

        # Backtracking line search: if the Newton step decreased the partial
        # log-likelihood (overshoot on ill-conditioned data), halve it until it
        # improves. This is R's coxfit step-halving and is what makes the
        # iteration converge where a bare Newton step diverges.
        halvings = 0
        while loglik_new < loglik_cur and halvings < _MAX_HALVINGS:
            step = step * 0.5
            beta_new = beta + step
            loglik_new = eval_loglik(beta_new)
            halvings += 1

        # R-style convergence: max|beta_new - beta| < tol.
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            n_iter = iteration
            break

        # Also accept convergence on relative log-likelihood change.
        if (
            iteration > 1
            and abs(loglik_new - loglik_old) / (abs(loglik_old) + 0.1) < tol
        ):
            beta = beta_new
            converged = True
            n_iter = iteration
            break

        beta = beta_new
        loglik_old = loglik_new
        n_iter = iteration

    model_loglik = eval_loglik(beta)
    _loglik, score_final, info_final = eval_full(beta)

    return NewtonResult(
        beta=beta,
        null_loglik=null_loglik,
        model_loglik=model_loglik,
        information=info_final,
        score=score_final,
        converged=converged,
        n_iter=n_iter,
    )


def flag_infinite_coefs(
    beta: NDArray,
    score: NDArray,
    variance: NDArray,
    tol: float,
) -> tuple[int, ...]:
    """Indices of coefficients that appear to be running to +/- infinity.

    Replicates R ``coxph.fit``'s post-convergence check: the remaining Newton
    step ``variance @ score`` is compared against the coefficient. A component
    that is non-finite, or larger than both ``tol`` and ``sqrt(tol) * |coef|``,
    signals a coefficient the loglik would keep pushing outward (monotone
    likelihood / separation) — R warns "coefficient may be infinite" for it.

    Parameters
    ----------
    beta : (p,)
        Coefficient estimates.
    score : (p,)
        Score at ``beta``.
    variance : (p, p)
        Inverse observed information (the coefficient covariance).
    tol : float
        Convergence tolerance (R's ``control$eps``); ``sqrt(tol)`` is R's
        ``toler.inf``.

    Returns
    -------
    tuple[int, ...]
        0-based indices of flagged coefficients (empty if none).
    """
    step = variance @ score
    toler_inf = np.sqrt(tol)
    flagged = []
    for j in range(len(beta)):
        if not np.isfinite(score[j]) or (
            abs(step[j]) > tol and abs(step[j]) > toler_inf * abs(beta[j])
        ):
            flagged.append(j)
    return tuple(flagged)
