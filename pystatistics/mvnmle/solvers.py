"""
Solver dispatch for MVN MLE.

Public API: mlest(data, ...) -> MVNSolution
"""

import warnings
from pystatistics.core.exceptions import ValidationError
from typing import Literal
import numpy as np

from pystatistics.core.compute.device import select_device
from pystatistics.core.compute.backend import resolve_backend, unknown_backend_message
from pystatistics.mvnmle.design import MVNDesign
from pystatistics.mvnmle.solution import MVNSolution


BackendChoice = Literal['auto', 'cpu', 'gpu', 'gpu_fp64']
AlgorithmChoice = Literal['direct', 'em', 'monotone']


def mlest(
    data_or_design,
    *,
    method: AlgorithmChoice = 'direct',
    backend: BackendChoice | None = None,
    solver: str | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
    regularize: bool = True,
    force: bool = False,
    collinearity_tol: float | None = None,
    verbose: bool = False,
) -> MVNSolution:
    """
    Maximum likelihood estimation for multivariate normal with missing data.

    Accepts EITHER:

        1. An MVNDesign object
        2. Raw data array or DataFrame (convenience)

    Parameters
    ----------
    data_or_design : array-like or MVNDesign
        Data matrix with NaN for missing values, or MVNDesign object.
    method : str
        Estimation method:

        - 'direct' (default): gradient-based optimization on the
          log-likelihood, parameterized by the Cholesky factor of the
          covariance (``Sigma = L L^T``) with a closed-form gradient.
        - 'em': Expectation-Maximization algorithm. Typically slower to
          converge but guaranteed monotone likelihood increase.
        - 'monotone': Closed-form MLE for monotone missingness patterns
          (Anderson 1957). Raises ValidationError if the data are not
          monotone — users should check with
          :func:`pystatistics.mvnmle.is_monotone` first, or use EM/direct
          for general patterns. When applicable, this is orders of
          magnitude faster than iterative algorithms.
    backend : str or None
        Backend selection. Default None → 'cpu'.

        - 'cpu' (and the default): the forward-Cholesky FP64 estimator. It
          runs on PyTorch when PyTorch is installed; otherwise the same
          estimator runs in NumPy (with a warning naming the implementation
          in use). The two agree to optimizer tolerance and match R.
        - 'gpu': require a GPU (CUDA or MPS), float32; raises if none available.
        - 'gpu_fp64': require CUDA, float64 (raises on MPS, which has no
          float64) — CPU-matching precision on the GPU.
        - 'auto': prefer CUDA (float32) when present, else the fast CPU path.
    solver : str or None
        scipy optimizer method for the direct method (e.g. ``'BFGS'``,
        ``'L-BFGS-B'``). If None, auto-selected by backend. Ignored for EM.
        ``'reference'`` (the inverse-Cholesky reference optimizer) was
        removed in 7.0.0 and raises ``ValidationError``.
    tol : float or None
        Convergence tolerance. If None, uses an algorithm- and
        precision-appropriate default: direct = per-observation gradient
        tolerance auto-selected by the backend (1e-5 on the FP64 paths,
        1e-3 on the FP32 GPU path, whose gradient noise floor is far above
        1e-5); em = 1e-4 (parameter change). For the direct method the
        tolerance is a scipy ``gtol`` and applies only to gradient-based
        solvers (BFGS, L-BFGS-B); derivative-free solvers ignore it.
    max_iter : int or None
        Maximum iterations. If None, uses algorithm-appropriate default:
        direct = 100, em = 1000.
    regularize : bool
        EM only (ignored by 'direct' and 'monotone'). When True (default),
        a numerically near-indefinite EM covariance iterate is restored to
        positive definiteness with a small diagonal ridge and a visible
        warning; when False, it raises ``NumericalError`` instead.
    force : bool
        When False (default), a rank-deficient fit — caused by
        (near-)collinear variables, for which no interior maximum-likelihood
        estimate exists — raises ``SingularMatrixError`` instead of
        returning a meaningless result. When True, the degenerate result is
        returned anyway with ``converged=False`` and a warning attached.
    collinearity_tol : float or None
        Full-rank detection threshold on the fitted correlation matrix's
        minimum eigenvalue. If None, uses the calibrated default (1e-5).
        Smaller values make the collinearity check more permissive.
    verbose : bool
        Print progress information.

    Returns
    -------
    MVNSolution

    Examples
    --------
    >>> from pystatistics.mvnmle import mlest, datasets
    >>> result = mlest(datasets.apple)
    >>> result_em = mlest(datasets.apple, method='em')
    >>> print(result.muhat)
    >>> print(result.loglik)
    """
    # Conflicting-request guards (Rule 1: fail loud, never silently override
    # an explicit choice). These must run BEFORE the backend normalization
    # below so an explicit request is distinguishable from the default.
    #
    # Removed option (hard removal, no shim): fail loud with the migration
    # path rather than passing 'reference' to scipy as an optimizer name.
    if solver == 'reference':
        raise ValidationError(
            "solver='reference' is no longer available: the inverse-Cholesky "
            "reference optimizer has been removed (pystatistics 7.0.0). Omit the "
            "solver argument to use the default CPU path, the double-precision "
            "forward-Cholesky estimator (PyTorch when it is installed, NumPy "
            "otherwise), which is the supported float64 estimator."
        )

    # method='monotone' is a CPU closed form (Anderson 1957) and cannot honor
    # an explicit GPU request.
    if method == 'monotone' and backend not in (None, 'cpu'):
        raise ValidationError(
            f"method='monotone' is a closed-form CPU solver and cannot "
            f"honor backend={backend!r}. Drop the backend argument, or use "
            f"method='direct'/'em' for GPU execution."
        )

    # Unspecified backend → CPU. GPU is never the default; callers must opt in
    # explicitly or request 'auto'.
    if backend is None:
        backend = 'cpu'

    # Get or build Design
    if isinstance(data_or_design, MVNDesign):
        design = data_or_design
    else:
        design = MVNDesign.from_array(data_or_design)

    if verbose:
        print(f"MVN MLE: {design.n} observations, {design.p} variables, "
              f"{design.missing_rate:.1%} missing")

    # Input-boundary guards (Rule 2). Each raises unless force=True, in which
    # case its warning is applied to the fit below.
    # 1. A (near-)constant column has zero variance, so no interior MLE
    #    exists. Invisible to the scale-invariant fitted-covariance guard
    #    (which divides each variable by its own standard deviation).
    # 2. A variable pair never observed in the same row leaves its covariance
    #    entry out of every likelihood term: the likelihood is flat in it and
    #    the entry is unidentified. Also invisible to the fitted-covariance
    #    guard (the fitted matrix is typically well-conditioned).
    from pystatistics.mvnmle._degeneracy import (
        check_observed_variances,
        check_pairwise_observation,
    )
    input_warnings = [
        w for w in (
            check_observed_variances(design.data, force=force),
            check_pairwise_observation(design.data, force=force),
        ) if w is not None
    ]

    if method == 'em':
        result = _solve_em(design, backend, tol, max_iter, regularize, verbose)
    elif method == 'direct':
        result = _solve_direct(design, backend, solver, tol, max_iter, verbose)
    elif method == 'monotone':
        result = _solve_monotone(design, verbose)
    else:
        raise ValidationError(
            f"Unknown method: {method!r}. "
            f"Use 'direct', 'em', or 'monotone'."
        )

    # Rank-deficiency guard (Rule 1: fail loud rather than report a
    # meaningless fit). Centralised here so every algorithm and backend
    # is covered by a single check. On (near-)collinear input the fitted
    # covariance is singular and the optimizer's convergence flag is not
    # trustworthy, so the fitted covariance is inspected directly.
    result = _guard_degeneracy(result, force=force, tol=collinearity_tol)

    # Apply the input-boundary warnings (only reachable under force=True;
    # otherwise the corresponding check raised).
    if input_warnings:
        from dataclasses import replace
        result = replace(
            result,
            params=replace(result.params, converged=False),
            warnings=result.warnings + tuple(input_warnings),
        )

    if verbose:
        print(f"Converged: {result.params.converged} "
              f"(iterations: {result.params.n_iter}, "
              f"loglik: {result.params.loglik:.6f})")

    return MVNSolution(_result=result, _design=design)


def _guard_degeneracy(result, *, force, tol):
    """Reject (or flag) a rank-deficient fit.

    Inspects the fitted covariance in ``result``. Returns ``result``
    unchanged when full-rank. When degenerate and ``force`` is True, returns
    a copy with ``converged=False`` and a warning appended. When degenerate
    and ``force`` is False, raises ``SingularMatrixError`` (via
    ``check_fitted_covariance``).
    """
    from dataclasses import replace

    from pystatistics.mvnmle._degeneracy import (
        DEFAULT_COLLINEARITY_TOL,
        check_fitted_covariance,
    )

    effective_tol = tol if tol is not None else DEFAULT_COLLINEARITY_TOL
    warning_msg = check_fitted_covariance(
        result.params.sigmahat, tol=effective_tol, force=force
    )
    if warning_msg is None:
        return result

    # force=True: keep the numbers but report the truth about them.
    return replace(
        result,
        params=replace(result.params, converged=False),
        warnings=result.warnings + (warning_msg,),
    )


def _solve_monotone(design, verbose):
    """Closed-form MVN MLE for monotone missingness patterns.

    Raises ``ValidationError`` if the data are not monotone.
    """
    import numpy as np

    from pystatistics.core.compute.timing import Timer
    from pystatistics.core.result import Result
    from pystatistics.mvnmle._monotone import mlest_monotone_closed_form
    from pystatistics.mvnmle._objectives.base import MLEObjectiveBase
    from pystatistics.mvnmle.backends._em_batched import (
        build_pattern_index,
        compute_loglik_batched_np,
    )
    from pystatistics.mvnmle.solution import MVNParams

    timer = Timer()
    timer.start()

    with timer.section('closed_form'):
        mu, sigma, _ = mlest_monotone_closed_form(design.data)

    with timer.section('loglikelihood'):
        obj = MLEObjectiveBase(design.data, skip_validation=True)
        index = build_pattern_index(obj.patterns, design.p)
        loglik = compute_loglik_batched_np(mu, sigma, obj.patterns, index)

    timer.stop()

    if verbose:
        print("Closed-form monotone MLE (Anderson 1957)")
        print(f"Log-likelihood: {loglik:.6f}")

    params = MVNParams(
        muhat=mu,
        sigmahat=sigma,
        loglik=loglik,
        n_iter=0,
        converged=True,
        gradient_norm=None,
    )
    return Result(
        params=params,
        info={
            'algorithm': 'monotone',
            'convergence_criterion': 'closed_form',
            'device': 'cpu',
        },
        timing=timer.result(),
        backend_name='cpu_monotone',
        warnings=(),
    )


def _solve_direct(design, backend, solver, tol, max_iter, verbose):
    """Dispatch direct (BFGS) optimization.

    ``solver``, when given, is the scipy optimizer method for the resolved
    (device, precision) backend.
    """
    effective_max_iter = max_iter if max_iter is not None else 100

    backend_impl = _get_backend(backend, verbose=verbose)
    scipy_method = solver

    if verbose:
        print(f"Backend: {backend_impl.name}")

    # Forward tol only when the user specified it — like `method`, the
    # backend owns the default (DirectMLEBackend auto-selects by precision:
    # 1e-5 for FP64 paths, 1e-3 for FP32, whose per-observation gradient
    # floor sits near 2e-4 and cannot meet the FP64 tolerance). Forcing a
    # value here would make that precision-aware default unreachable.
    solve_kwargs = {'max_iter': effective_max_iter}
    if tol is not None:
        solve_kwargs['tol'] = tol
    if scipy_method is not None:
        solve_kwargs['method'] = scipy_method

    return backend_impl.solve(design, **solve_kwargs)


def _solve_em(design, backend, tol, max_iter, regularize, verbose):
    """Dispatch EM algorithm."""
    from pystatistics.mvnmle.backends.em import EMBackend

    effective_tol = tol if tol is not None else 1e-4
    effective_max_iter = max_iter if max_iter is not None else 1000

    # Select device for EM backend, with size-aware dispatch and
    # Rule-1-compliant visibility on any non-obvious choice.
    device = _get_em_device(backend, design.n, design.p, verbose)

    backend_impl = EMBackend(device=device)

    if verbose:
        print(f"Backend: {backend_impl.name}")

    return backend_impl.solve(
        design,
        tol=effective_tol,
        max_iter=effective_max_iter,
        regularize=regularize,
    )


# ---------------------------------------------------------------------------
# EM GPU-vs-CPU dispatch heuristic
# ---------------------------------------------------------------------------
#
# The GPU EM path is launch-overhead-bound on small data: for shapes
# like apple (18x2) or iris (150x4) the H2D transfer plus per-iteration
# kernel launches exceed the scalar numpy work on CPU. We measured the
# crossover empirically across (apple, missvals, iris, wine, breast)
# at 15 % random MCAR; n*v ≈ 1500 is where GPU starts winning.
#
# Below this threshold, GPU ends up slower. We still respect explicit
# ``backend='gpu'`` (user asked for it, they get it) but emit a
# UserWarning so the tradeoff is visible. For ``backend='auto'`` the
# heuristic picks the actually-faster device and we likewise warn
# when the choice might surprise the user (e.g. GPU available but
# skipped because the data are small).

_EM_GPU_WORTH_IT_THRESHOLD = 1500


def _em_gpu_worth_it(n_obs: int, n_vars: int) -> bool:
    """Return True iff GPU EM is expected to beat CPU EM on a shape of
    (n_obs, n_vars). Empirically calibrated at n*v ≈ 1500 on random
    MCAR data."""
    return n_obs * n_vars >= _EM_GPU_WORTH_IT_THRESHOLD


def _get_em_device(
    backend_choice: BackendChoice,
    n_obs: int,
    n_vars: int,
    verbose: bool = False,
) -> str:
    """Select device for EM backend, applying the size heuristic and
    emitting visible warnings when a non-obvious choice is made.

    Per Rule 1 (no silent fallbacks, no 'for your own good' auto
    behaviour): every dispatch decision the user didn't explicitly
    make is surfaced via ``UserWarning``. ``backend='cpu'`` stays
    silent because it's a direct, obvious choice.
    """
    import warnings

    worth_gpu = _em_gpu_worth_it(n_obs, n_vars)

    if backend_choice == 'auto':
        device = select_device('auto')
        gpu_actually_available = device.device_type == 'cuda'

        if gpu_actually_available:
            try:
                import torch  # noqa: F401
            except ImportError:
                warnings.warn(
                    "backend='auto': CUDA detected but PyTorch not "
                    "available; dispatching EM to CPU.",
                    UserWarning, stacklevel=3,
                )
                return 'cpu'

            if worth_gpu:
                # GPU wins on this shape; pick it silently because this
                # is the default-assumed auto behaviour when a GPU is
                # present. No surprise to report.
                return 'cuda'

            # GPU is available but the shape is too small for it to
            # win. Surface the dispatch decision so the user knows
            # why ``backend='auto'`` isn't picking the GPU.
            warnings.warn(
                f"backend='auto': dispatching EM to CPU on "
                f"{n_obs}x{n_vars} data (n*v={n_obs * n_vars} below "
                f"the empirical GPU-worth-it threshold of "
                f"{_EM_GPU_WORTH_IT_THRESHOLD}). GPU is available "
                f"but would likely be slower due to kernel-launch "
                f"overhead on small per-iteration work. Pass "
                f"backend='gpu' to force GPU anyway.",
                UserWarning, stacklevel=3,
            )
            return 'cpu'

        # No GPU available: CPU is the only option, nothing to report.
        return 'cpu'

    elif backend_choice == 'cpu':
        return 'cpu'

    elif backend_choice in ('gpu', 'gpu_fp64'):
        # EM computes in float64 on CUDA regardless (float32 only on MPS,
        # which is rejected below), so the documented 'gpu_fp64' backend is
        # honored by the same path as 'gpu'.
        device = select_device('gpu')  # raises RuntimeError if no GPU
        if device.device_type == 'mps':
            raise RuntimeError(
                "backend='gpu' for the EM algorithm is not supported on "
                "Apple Silicon (MPS). MPS is float32-only, and EM's "
                "fixed-point iteration is unreliable in float32: the "
                "iteration stalls at the fp32 noise floor (measured ~50x "
                "the iteration count of the fp64 CPU path before "
                "converging) and the per-pattern E-step Cholesky can lose "
                "positive-definiteness outright at moderate dimension "
                "(see docs/GPU_NOTES.md). Use backend='cpu' (or "
                "backend='auto', which routes to CPU on MPS). CUDA is "
                "supported (EM runs in float64 there)."
            )
        if not worth_gpu:
            warnings.warn(
                f"backend='gpu': proceeding on GPU as requested, but "
                f"{n_obs}x{n_vars} data (n*v={n_obs * n_vars}) is "
                f"below the empirical GPU-worth-it threshold of "
                f"{_EM_GPU_WORTH_IT_THRESHOLD}. CPU is expected to be "
                f"faster on this shape due to GPU kernel-launch "
                f"overhead. Pass backend='cpu' or 'auto' to skip GPU.",
                UserWarning, stacklevel=3,
            )
        return device.device_type

    else:
        raise ValidationError(
            unknown_backend_message(
                backend_choice, ('auto', 'cpu', 'gpu', 'gpu_fp64')
            )
        )


def _fast_cpu_backend():
    """Return the forward-Cholesky FP64 CPU backend: PyTorch when it is
    installed, otherwise the NumPy implementation of the same estimator.

    The forward-Cholesky estimator is the default direct CPU path (it matches
    R to ~1e-9). PyTorch is an optional dependency, so on a bare install the
    estimator runs through ``NumpyCholeskyMLEBackend`` instead: same
    parameterization, objective, analytic gradient, optimizer and defaults,
    in NumPy. The two implementations agree to optimizer tolerance.

    Per Rule 1 (no silent fallbacks), the switch of implementation — the user
    asked for the default/'cpu'/'auto' and PyTorch is missing — is surfaced
    via ``UserWarning``; the result's ``backend_name`` also records
    which implementation ran.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        warnings.warn(
            "PyTorch is not installed, so direct MVN MLE runs the NumPy "
            "implementation of the forward-Cholesky estimator. It is the "
            "same estimator as the PyTorch path and gives the same "
            "estimates to optimizer tolerance, but it is CPU-only and can "
            "be slower on wide data with many missingness patterns. "
            "Install 'pystatistics[gpu]' to use PyTorch.",
            # _fast_cpu_backend <- _get_backend <- _solve_direct <- mlest
            # <- caller: attribute the warning to the caller's line.
            UserWarning, stacklevel=5,
        )
        from pystatistics.mvnmle.backends.numpy_cholesky import (
            NumpyCholeskyMLEBackend,
        )
        return NumpyCholeskyMLEBackend()
    from pystatistics.mvnmle.backends.gpu import DirectMLEBackend
    return DirectMLEBackend(device='cpu', use_fp64=True)


def _get_backend(choice: BackendChoice, verbose: bool = False):
    """Select the direct-MLE backend from the resolved (device, precision) target.

    Maps the (device, precision) ``backend`` axis via :func:`resolve_backend`:
    'cpu'/None -> the fast forward-Cholesky CPU path, 'gpu' -> float32,
    'gpu_fp64' -> CUDA float64, 'auto' -> CUDA-float32 or CPU.
    """
    target = resolve_backend(choice, supports_fp64=True)
    if target.device_type == 'cpu':
        return _fast_cpu_backend()

    from pystatistics.mvnmle.backends.gpu import DirectMLEBackend
    return DirectMLEBackend(device=target.device_type, use_fp64=target.use_fp64)
