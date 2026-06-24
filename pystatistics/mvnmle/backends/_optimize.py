"""
Scaled optimisation driver shared by the MVN MLE backends.

WHY THIS EXISTS
---------------
scipy's gradient-based methods (BFGS, L-BFGS-B) judge convergence on the
ABSOLUTE gradient infinity-norm against ``gtol``. The MVN MLE objective is the
summed ``-2 * log-likelihood`` over all observations, whose magnitude grows with
the dataset (~1e5+ for survey-scale data). At that magnitude FP64 cannot drive
the absolute gradient below the default ``gtol``, so the line search terminates
with "Desired error not necessarily achieved due to precision loss" and
``success=False`` even though the parameters are at the optimum — a false
non-convergence that surfaces spurious warnings and perturbs downstream logic
that branches on ``converged``.

THE FIX
-------
Run the optimiser on the objective scaled to a PER-OBSERVATION mean (divided by
the number of observed scalar values). Scaling by a positive constant does not
move the argmin, so the fitted estimates and log-likelihood are identical to
full precision; it only makes the gradient magnitude O(1) so ``gtol`` is a
meaningful, dataset-size-invariant convergence test.

The objective modules remain the R-exact reference (their ``compute_objective``
is unchanged and is used directly for the reported log-likelihood); scaling is
purely an optimiser-conditioning concern handled here.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from pystatistics.core.exceptions import NumericalError


@dataclass(frozen=True)
class ScaledMinimizeResult:
    """Outcome of a scaled MVN MLE optimisation.

    ``objective_value`` is reported on the original ``-2 * log-likelihood``
    scale (so it stays consistent with the reported log-likelihood), while
    ``gradient_norm`` is the PER-OBSERVATION (scaled) gradient — the quantity
    convergence is actually judged against, hence O(1) and directly comparable
    to ``gtol``. Reporting the raw unscaled gradient would be misleading: at a
    summed-objective magnitude of ~1e6 it is dominated by FP64 noise and looks
    "large" even at a genuine optimum.

    Attributes
    ----------
    x : np.ndarray
        Optimal parameter vector (unaffected by scaling).
    success : bool
        Whether the optimiser reported convergence.
    message : str
        Optimiser termination message.
    n_iter : int
        Number of optimiser iterations.
    n_function_evals : int
        Number of objective evaluations.
    n_gradient_evals : int
        Number of gradient evaluations.
    objective_value : float
        Unscaled objective (``-2 * log-likelihood``) at ``x``.
    gradient_norm : float or None
        Per-observation (scaled) max-abs gradient at ``x`` — the value compared
        against ``gtol``. ``None`` if the optimiser did not expose a final
        gradient (e.g. derivative-free methods).
    scale : float
        Divisor applied to the objective and gradient during optimisation.
    """

    x: np.ndarray
    success: bool
    message: str
    n_iter: int
    n_function_evals: int
    n_gradient_evals: int
    objective_value: float
    gradient_norm: Optional[float]
    scale: float


def run_scaled_minimize(
    objective,
    theta0: np.ndarray,
    *,
    method: str,
    tol: float,
    max_iter: int,
) -> ScaledMinimizeResult:
    """Optimise ``objective`` with a per-observation-scaled objective.

    Input contract (Rule 2 — validate inputs at the boundary)
    ---------------------------------------------------------
    ``objective`` MUST expose:
      - ``compute_objective(theta) -> float`` : the summed ``-2 * log-lik``.
      - ``compute_gradient(theta) -> np.ndarray`` : its gradient.
      - ``n_observed_scalars`` : a positive int, the number of observed
        (non-missing) scalar values in the data.

    Parameters
    ----------
    objective : object
        MVN MLE objective satisfying the contract above.
    theta0 : np.ndarray
        Initial parameter vector.
    method : str
        ``scipy.optimize.minimize`` method ('BFGS', 'L-BFGS-B', ...).
    tol : float
        Gradient tolerance (``gtol``) for gradient-based methods.
    max_iter : int
        Maximum optimiser iterations.

    Returns
    -------
    ScaledMinimizeResult
        Result with all magnitudes reported on the original (unscaled) scale.

    Raises
    ------
    NumericalError
        If ``objective.n_observed_scalars`` is missing or non-positive — we fail
        loud rather than divide by zero and silently produce a meaningless
        convergence test (Rule 1).
    """
    scale = float(getattr(objective, 'n_observed_scalars', 0))
    if not scale > 0.0:
        raise NumericalError(
            "Cannot scale the MVN MLE objective: objective reports "
            f"n_observed_scalars={scale!r}; expected a positive count of "
            "observed (non-missing) data values. The objective is misconfigured."
        )

    def scaled_objective(theta: np.ndarray) -> float:
        return objective.compute_objective(theta) / scale

    def scaled_gradient(theta: np.ndarray) -> np.ndarray:
        return objective.compute_gradient(theta) / scale

    options = {'maxiter': max_iter}
    # `gtol` only applies to the gradient-based methods; passing it to
    # derivative-free methods (Nelder-Mead, Powell) is meaningless and emits a
    # scipy warning, so set it only where it is honoured.
    if method in ('BFGS', 'L-BFGS-B'):
        options['gtol'] = tol

    opt_result = minimize(
        scaled_objective,
        theta0,
        jac=scaled_gradient,
        method=method,
        options=options,
    )

    # Report the objective on the ORIGINAL scale (multiply back) so it stays
    # consistent with the log-likelihood. Report the gradient norm on the
    # SCALED (per-observation) scale — that is the quantity compared against
    # `gtol`, so a small value here corresponds to the `converged` flag.
    grad_norm: Optional[float] = None
    jac = getattr(opt_result, 'jac', None)
    if jac is not None:
        grad_norm = float(np.max(np.abs(jac)))

    return ScaledMinimizeResult(
        x=opt_result.x,
        success=bool(opt_result.success),
        message=str(getattr(opt_result, 'message', '')),
        n_iter=int(getattr(opt_result, 'nit', 0)),
        n_function_evals=int(getattr(opt_result, 'nfev', 0)),
        n_gradient_evals=int(getattr(opt_result, 'njev', 0)),
        objective_value=float(opt_result.fun) * scale,
        gradient_norm=grad_norm,
        scale=scale,
    )
