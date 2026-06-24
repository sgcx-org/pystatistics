"""
Shared direct-MLE solve skeleton for the MVN MLE backends.

WHY THIS EXISTS
---------------
Every direct (gradient-based) MVN MLE backend runs the identical sequence:
build an objective, get initial parameters, optimise the per-observation-scaled
objective (``run_scaled_minimize``), extract the estimates, and assemble a
``Result[MVNParams]``. The only differences between backends are *which*
objective they build and a few labels (parameterization, device, precision).

Previously that skeleton was copy-pasted across the CPU and GPU backends, so a
fix to the timing sections, the convergence-warning text, or the ``info`` dict
had to be made in two places. This module owns the skeleton once; each backend
supplies an objective factory and its metadata.

INPUT CONTRACT (Rule 2 — validate at the boundary)
--------------------------------------------------
``objective_factory()`` MUST return an object exposing:
  - ``get_initial_parameters() -> np.ndarray``
  - ``compute_objective(theta) -> float`` and ``compute_gradient(theta)``
    (consumed by ``run_scaled_minimize``)
  - ``extract_parameters(theta) -> (mu, sigma, loglik)``
  - ``n_observed_scalars`` : positive int (consumed by ``run_scaled_minimize``)
An optional ``clear_cache()`` method is called if present (GPU objectives free
device memory; the numpy objective has nothing to release).
"""

from typing import Callable, Optional

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.mvnmle.solution import MVNParams
from pystatistics.mvnmle.backends._optimize import run_scaled_minimize


def run_direct_solve(
    objective_factory: Callable[[], object],
    *,
    method: str,
    tol: float,
    max_iter: int,
    backend_name: str,
    parameterization: str,
    device: Optional[str] = None,
    precision: Optional[str] = None,
    sync_cuda: bool = False,
) -> Result[MVNParams]:
    """Run the shared direct-MLE optimisation skeleton.

    Parameters
    ----------
    objective_factory : callable
        Zero-argument factory returning an objective satisfying the module
        contract above. Called inside the ``objective_setup`` timing section so
        construction cost (e.g. host->device transfer) is measured.
    method, tol, max_iter : str, float, int
        Passed through to ``run_scaled_minimize``.
    backend_name : str
        Identifier recorded on the returned ``Result``.
    parameterization : str
        Label for ``info['parameterization']`` ('cholesky' or
        'inverse_cholesky').
    device : str or None
        When given, recorded as ``info['device']`` and surfaced to callers.
    precision : str or None
        When given, recorded as ``info['precision']`` ('fp32'/'fp64').
    sync_cuda : bool
        Whether the timer should synchronise CUDA before reading the clock.

    Returns
    -------
    Result[MVNParams]
        Estimates plus optimisation metadata. ``warnings`` carries a
        non-convergence message when the optimiser did not report success.
    """
    timer = Timer(sync_cuda=sync_cuda)
    timer.start()
    warnings_list = []

    with timer.section('objective_setup'):
        objective = objective_factory()

    with timer.section('initial_parameters'):
        theta0 = objective.get_initial_parameters()

    # Optimise the per-observation-scaled objective so that `gtol` is a
    # meaningful, dataset-size-invariant convergence test (see _optimize.py).
    # Scaling does not move the optimum, so the estimates below are unchanged.
    with timer.section('optimization'):
        opt = run_scaled_minimize(
            objective, theta0, method=method, tol=tol, max_iter=max_iter
        )

    with timer.section('parameter_extraction'):
        mu, sigma, loglik = objective.extract_parameters(opt.x)

    if not opt.success:
        msg = opt.message or 'Unknown convergence failure'
        warnings_list.append(f"Optimization did not converge: {msg}")

    # Release device memory for backends that hold it (GPU); a no-op otherwise.
    clear_cache = getattr(objective, 'clear_cache', None)
    if callable(clear_cache):
        clear_cache()

    timer.stop()

    params = MVNParams(
        muhat=mu,
        sigmahat=sigma,
        loglik=loglik,
        n_iter=opt.n_iter,
        converged=opt.success,
        gradient_norm=opt.gradient_norm,
    )

    info = {
        'method': method,
        'objective_value': opt.objective_value,
        'n_function_evals': opt.n_function_evals,
        'n_gradient_evals': opt.n_gradient_evals,
        'message': opt.message,
        'parameterization': parameterization,
    }
    if device is not None:
        info['device'] = device
    if precision is not None:
        info['precision'] = precision

    return Result(
        params=params,
        info=info,
        timing=timer.result(),
        backend_name=backend_name,
        warnings=tuple(warnings_list),
    )
