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
AlgorithmChoice = Literal['direct', 'em']


def mlest(
    data_or_design,
    *,
    algorithm: AlgorithmChoice = 'direct',
    backend: BackendChoice = 'auto',
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
    backend : str
        Backend selection: 'auto', 'cpu', 'gpu'.
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
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm!r}. Use 'direct' or 'em'."
        )

    if verbose:
        print(f"Converged: {result.params.converged} "
              f"(iterations: {result.params.n_iter}, "
              f"loglik: {result.params.loglik:.6f})")

    return MVNSolution(_result=result, _design=design)


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

    # Select device for EM backend
    device = _get_em_device(backend, verbose)

    backend_impl = EMBackend(device=device)

    if verbose:
        print(f"Backend: {backend_impl.name}")

    return backend_impl.solve(design, tol=effective_tol, max_iter=effective_max_iter)


def _get_em_device(backend_choice: BackendChoice, verbose: bool = False) -> str:
    """Select device for EM backend."""
    if backend_choice == 'auto':
        device = select_device('auto')
        if device.device_type == 'cuda':
            try:
                import torch
                return 'cuda'
            except ImportError:
                if verbose:
                    print("CUDA detected but PyTorch not available. Using CPU.")
                return 'cpu'
        # auto + MPS -> CPU (same as direct: MPS not auto-selected)
        # auto + CPU -> CPU
        return 'cpu'

    elif backend_choice == 'cpu':
        return 'cpu'

    elif backend_choice == 'gpu':
        device = select_device('gpu')  # raises RuntimeError if no GPU
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
