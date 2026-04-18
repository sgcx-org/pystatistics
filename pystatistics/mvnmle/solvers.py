"""
Solver dispatch for MVN MLE.

Public API: mlest(data, ...) -> MVNSolution
"""

import warnings
from typing import Literal
import numpy as np

from pystatistics.core.compute.device import select_device
from pystatistics.mvnmle.design import MVNDesign
from pystatistics.mvnmle.solution import MVNSolution
from pystatistics.mvnmle.backends.cpu import CPUMLEBackend


BackendChoice = Literal['auto', 'cpu', 'gpu']
AlgorithmChoice = Literal['direct', 'em', 'monotone']


def mlest(
    data_or_design,
    *,
    algorithm: AlgorithmChoice = 'direct',
    backend: BackendChoice | None = None,
    method: str | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
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
    algorithm : str
        Estimation algorithm:
        - 'direct' (default): BFGS optimization on the log-likelihood,
          using R-exact inverse Cholesky parameterization.
        - 'em': Expectation-Maximization algorithm. Typically slower to
          converge but guaranteed monotone likelihood increase.
        - 'monotone': Closed-form MLE for monotone missingness patterns
          (Anderson 1957). Raises ValidationError if the data are not
          monotone — users should check with
          :func:`pystatistics.mvnmle.is_monotone` first, or use EM/direct
          for general patterns. When applicable, this is orders of
          magnitude faster than iterative algorithms.
    backend : str or None
        Backend selection. Default None → 'cpu' (R-reference path,
        validated for regulated-industry use). Explicit values:
        'cpu', 'gpu', or 'auto' to prefer GPU when available.
    method : str or None
        Optimization method for direct algorithm. If None, auto-selected
        by backend. Ignored for EM.
    tol : float or None
        Convergence tolerance. If None, uses algorithm-appropriate default:
        direct = 1e-5 (gradient tolerance), em = 1e-4 (parameter change).
    max_iter : int or None
        Maximum iterations. If None, uses algorithm-appropriate default:
        direct = 100, em = 1000.
    verbose : bool
        Print progress information.

    Returns
    -------
    MVNSolution

    Examples
    --------
    >>> from pystatistics.mvnmle import mlest, datasets
    >>> result = mlest(datasets.apple)
    >>> result_em = mlest(datasets.apple, algorithm='em')
    >>> print(result.muhat)
    >>> print(result.loglik)
    """
    # Unspecified backend → CPU (R-reference path). GPU is never the
    # default; callers must opt in explicitly or request 'auto'.
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

    if algorithm == 'em':
        result = _solve_em(design, backend, tol, max_iter, verbose)
    elif algorithm == 'direct':
        result = _solve_direct(design, backend, method, tol, max_iter, verbose)
    elif algorithm == 'monotone':
        result = _solve_monotone(design, verbose)
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm!r}. "
            f"Use 'direct', 'em', or 'monotone'."
        )

    if verbose:
        print(f"Converged: {result.params.converged} "
              f"(iterations: {result.params.n_iter}, "
              f"loglik: {result.params.loglik:.6f})")

    return MVNSolution(_result=result, _design=design)


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


def _solve_direct(design, backend, method, tol, max_iter, verbose):
    """Dispatch direct (BFGS) optimization."""
    effective_tol = tol if tol is not None else 1e-5
    effective_max_iter = max_iter if max_iter is not None else 100

    backend_impl = _get_backend(backend, verbose=verbose)

    if verbose:
        print(f"Backend: {backend_impl.name}")

    solve_kwargs = {'max_iter': effective_max_iter, 'tol': effective_tol}
    if method is not None:
        solve_kwargs['method'] = method

    return backend_impl.solve(design, **solve_kwargs)


def _solve_em(design, backend, tol, max_iter, verbose):
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

    return backend_impl.solve(design, tol=effective_tol, max_iter=effective_max_iter)


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
        raise ValueError(f"Unknown backend: {backend_choice!r}")


def _get_backend(choice: BackendChoice, verbose: bool = False):
    """Select backend based on user choice and hardware availability."""
    if choice == 'auto':
        device = select_device('auto')
        if device.device_type == 'cuda':
            try:
                from pystatistics.mvnmle.backends.gpu import GPUMLEBackend
                return GPUMLEBackend(device='cuda')
            except (ImportError, RuntimeError) as e:
                if verbose:
                    print(f"GPU backend unavailable: {e}. Using CPU.")
                return CPUMLEBackend()
        # auto + MPS -> CPU (same as regression: MPS not auto-selected)
        # auto + CPU -> CPU
        return CPUMLEBackend()

    elif choice == 'cpu':
        return CPUMLEBackend()

    elif choice == 'gpu':
        device = select_device('gpu')  # raises RuntimeError if no GPU
        from pystatistics.mvnmle.backends.gpu import GPUMLEBackend
        return GPUMLEBackend(device=device.device_type)

    else:
        raise ValueError(f"Unknown backend: {choice!r}")
