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


def mlest(
    data_or_design,
    *,
    backend: BackendChoice = 'auto',
    method: str | None = None,
    tol: float = 1e-5,
    max_iter: int = 100,
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
    backend : str
        Backend selection: 'auto', 'cpu', 'gpu'
    method : str or None
        Optimization method. If None, auto-selected by backend.
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
    verbose : bool
        Print progress information

    Returns
    -------
    MVNSolution

    Examples
    --------
    >>> from pystatistics.mvnmle import mlest, datasets
    >>> result = mlest(datasets.apple)
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

    # Select backend
    backend_impl = _get_backend(backend, verbose=verbose)

    if verbose:
        print(f"Backend: {backend_impl.name}")

    # Solve
    solve_kwargs = {'max_iter': max_iter}
    if method is not None:
        solve_kwargs['method'] = method
    if tol is not None:
        solve_kwargs['tol'] = tol

    result = backend_impl.solve(design, **solve_kwargs)

    if verbose:
        print(f"Converged: {result.params.converged} "
              f"(iterations: {result.params.n_iter}, "
              f"loglik: {result.params.loglik:.6f})")

    return MVNSolution(_result=result, _design=design)


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
