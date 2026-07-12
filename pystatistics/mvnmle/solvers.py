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
from pystatistics.mvnmle.backends.cpu import CPUMLEBackend


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
          log-likelihood. The parameterization depends on the backend (see
          ``backend``): the default CPU path uses a forward-Cholesky
          factorization; ``solver='reference'`` uses the R-exact inverse
          Cholesky parameterization.
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
        - 'cpu' (and the default): fast PyTorch forward-Cholesky FP64 path
          when PyTorch is installed; otherwise falls back (with a warning) to
          the numpy inverse-Cholesky reference. Both match R; the PyTorch path
          is substantially faster.
        - 'gpu': require a GPU (CUDA or MPS), float32; raises if none available.
        - 'gpu_fp64': require CUDA, float64 (raises on MPS, which has no
          float64) — CPU-matching precision on the GPU.
        - 'auto': prefer CUDA (float32) when present, else the fast CPU path.
    solver : str or None
        Numerical routine for the direct method. If None, auto-selected by
        backend. ``'reference'`` selects the R-exact numpy inverse-Cholesky
        reference (no PyTorch needed; valid only with ``method='direct'``); any
        other value is passed as the scipy optimizer method. Ignored for EM.
    tol : float or None
        Convergence tolerance. If None, uses algorithm-appropriate default:
        direct = 1e-5 (gradient tolerance), em = 1e-4 (parameter change).
    max_iter : int or None
        Maximum iterations. If None, uses algorithm-appropriate default:
        direct = 100, em = 1000.
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
    # Unspecified backend → CPU. GPU is never the default; callers must opt in
    # explicitly or request 'auto'.
    if backend is None:
        backend = 'cpu'

    # The R-exact numpy inverse-Cholesky reference is selected by
    # ``solver='reference'`` (a numerical-routine choice, per CONVENTIONS) — a
    # ``solver`` axis choice, never a ``backend`` (device+precision) one.
    #
    # The numpy inverse-Cholesky reference is a direct-method routine; it has no
    # meaning for EM or the closed-form monotone solver. Fail loud (Rule 1)
    # rather than silently ignoring the request.
    if solver == 'reference' and method != 'direct':
        raise ValidationError(
            "solver='reference' selects the numpy inverse-Cholesky reference "
            "and is only valid with method='direct'. "
            f"Got method={method!r}."
        )

    # Get or build Design
    if isinstance(data_or_design, MVNDesign):
        design = data_or_design
    else:
        design = MVNDesign.from_array(data_or_design)

    if verbose:
        print(f"MVN MLE: {design.n} observations, {design.p} variables, "
              f"{design.missing_rate:.1%} missing")

    # Input-boundary guard (Rule 2): a (near-)constant column has zero variance,
    # so no interior MLE exists. This degeneracy is invisible to the
    # scale-invariant fitted-covariance guard below (which divides each variable
    # by its own standard deviation), so it must be caught here on the raw input.
    # Raises unless force=True, in which case the warning is applied to the fit.
    from pystatistics.mvnmle._degeneracy import check_observed_variances
    constant_col_warning = check_observed_variances(design.data, force=force)

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

    # Apply the constant-column warning gathered at the input boundary (only
    # reachable under force=True; otherwise check_observed_variances raised).
    if constant_col_warning is not None:
        from dataclasses import replace
        result = replace(
            result,
            params=replace(result.params, converged=False),
            warnings=result.warnings + (constant_col_warning,),
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

    ``solver='reference'`` selects the R-exact numpy inverse-Cholesky backend
    (no PyTorch); any other ``solver`` value is the scipy optimizer method for
    the resolved (device, precision) backend.
    """
    effective_tol = tol if tol is not None else 1e-5
    effective_max_iter = max_iter if max_iter is not None else 100

    if solver == 'reference':
        backend_impl = CPUMLEBackend()
        scipy_method = None
    else:
        backend_impl = _get_backend(backend, verbose=verbose)
        scipy_method = solver

    if verbose:
        print(f"Backend: {backend_impl.name}")

    solve_kwargs = {'max_iter': effective_max_iter, 'tol': effective_tol}
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

    elif backend_choice == 'gpu':
        device = select_device('gpu')  # raises RuntimeError if no GPU
        if device.device_type == 'mps':
            raise RuntimeError(
                "backend='gpu' for the EM algorithm is not supported on "
                "Apple Silicon (MPS). EM is an iterative fixed-point "
                "method with small per-step work and per-pattern scatter "
                "fills — a workload shape where Metal's kernel-launch "
                "overhead makes it far slower than the CPU (see "
                "docs/GPU_BACKEND_NOTES.md). Use backend='cpu' (or "
                "backend='auto', which routes to CPU on MPS). CUDA is "
                "supported."
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
            unknown_backend_message(backend_choice, ('auto', 'cpu', 'gpu'))
        )


def _fast_cpu_backend(implicit: bool):
    """Return the fast forward-Cholesky FP64 CPU backend, or the numpy
    inverse-Cholesky reference when PyTorch is unavailable.

    The PyTorch forward-Cholesky estimator on a CPU torch device is the fast
    default CPU path (it beats the numpy reference substantially and matches R
    to ~1e-9). PyTorch is an optional dependency, so on a bare install the only
    direct path is the numpy reference; we fall back to it rather than failing.

    Per Rule 1 (no silent fallbacks), an *implicit* fallback — the user asked
    for the default/'cpu'/'auto' and got the reference because PyTorch is
    missing — is surfaced via ``UserWarning``. An explicit ``solver='reference'``
    request is silent (handled by the caller).
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        if implicit:
            warnings.warn(
                "PyTorch is not installed, so direct MVN MLE falls back to the "
                "numpy inverse-Cholesky reference. This path is correct and "
                "R-validated but substantially slower than the PyTorch "
                "forward-Cholesky path. Install 'pystatistics[gpu]' for the "
                "fast path, or pass solver='reference' to select the "
                "reference explicitly and silence this warning.",
                UserWarning, stacklevel=4,
            )
        return CPUMLEBackend()
    from pystatistics.mvnmle.backends.gpu import DirectMLEBackend
    return DirectMLEBackend(device='cpu', use_fp64=True)


def _get_backend(choice: BackendChoice, verbose: bool = False):
    """Select the direct-MLE backend from the resolved (device, precision) target.

    The reference numpy path is not selected here — it is a ``solver``
    ('reference') choice handled in :func:`_solve_direct`. This resolver only
    maps the (device, precision) ``backend`` axis via :func:`resolve_backend`:
    'cpu'/None -> the fast forward-Cholesky CPU path, 'gpu' -> float32,
    'gpu_fp64' -> CUDA float64, 'auto' -> CUDA-float32 or CPU.
    """
    target = resolve_backend(choice, supports_fp64=True)
    if target.device_type == 'cpu':
        return _fast_cpu_backend(implicit=True)

    from pystatistics.mvnmle.backends.gpu import DirectMLEBackend
    return DirectMLEBackend(device=target.device_type, use_fp64=target.use_fp64)
