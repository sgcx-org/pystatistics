"""
Solver dispatch for regression.

Handles both Linear Models (LM) and Generalized Linear Models (GLM).

When family is None (default), the existing LM path is used → LinearSolution.
When family is provided, the IRLS path is used → GLMSolution.
"""

import warnings
from typing import Literal, Union
import numpy as np
from numpy.typing import ArrayLike

from pystatistics.core.compute.device import select_device
from pystatistics.regression.design import Design
from pystatistics.regression.solution import LinearSolution, GLMSolution
from pystatistics.regression.backends.cpu import CPUQRBackend


BackendChoice = Literal['auto', 'cpu', 'gpu', 'cpu_qr', 'cpu_svd', 'gpu_qr']


def fit(
    X_or_design: ArrayLike | Design,
    y: ArrayLike | None = None,
    *,
    family: 'str | Family | None' = None,
    backend: BackendChoice = 'auto',
    force: bool = False,
    tol: float = 1e-8,
    max_iter: int = 25,
) -> Union[LinearSolution, GLMSolution]:
    """
    Fit a linear or generalized linear model.

    When family is None (default), fits ordinary least squares (LM) via
    QR decomposition or GPU Cholesky. When family is specified, fits a
    GLM via IRLS (Iteratively Reweighted Least Squares).

    Accepts EITHER:
        1. A Design object (from DataSource or arrays)
        2. Raw X and y arrays (convenience)

    Args:
        X_or_design: Design object or X matrix
        y: Response vector (required if X_or_design is array)
        family: GLM family specification. None for OLS, or a string
            ('gaussian', 'binomial', 'poisson') or Family instance.
        backend: 'auto', 'cpu', 'gpu', 'cpu_qr', 'cpu_svd', 'gpu_qr'
        force: If True, proceed with GPU Cholesky even on ill-conditioned
            matrices. Has no effect on CPU backends.
        tol: Convergence tolerance for IRLS (GLM only). Default 1e-8
            matches R's glm.control().
        max_iter: Maximum IRLS iterations (GLM only). Default 25
            matches R's glm.control().

    Returns:
        LinearSolution when family is None.
        GLMSolution when family is specified.

    Examples:
        # OLS (unchanged from before)
        >>> result = fit(X, y)
        >>> result = fit(X, y, backend='gpu')

        # Logistic regression
        >>> result = fit(X, y, family='binomial')

        # Poisson regression
        >>> result = fit(X, y, family='poisson')

        # Gaussian GLM (equivalent to OLS)
        >>> result = fit(X, y, family='gaussian')
    """
    # Get or build Design
    if isinstance(X_or_design, Design):
        design = X_or_design
    else:
        if y is None:
            raise ValueError("y required when passing arrays")
        design = Design.from_arrays(np.asarray(X_or_design), np.asarray(y))

    # Dispatch: GLM path if family specified, otherwise LM path
    if family is not None:
        return _fit_glm(design, family, backend, tol, max_iter)
    else:
        return _fit_lm(design, backend, force)


def _fit_lm(
    design: Design,
    backend: BackendChoice,
    force: bool,
) -> LinearSolution:
    """Fit ordinary least squares (existing LM path, unchanged)."""
    backend_impl = _get_lm_backend(backend, design)

    # Solve — pass force to GPU backends, CPU ignores it
    if hasattr(backend_impl, 'solve') and 'force' in backend_impl.solve.__code__.co_varnames:
        result = backend_impl.solve(design, force=force)
    else:
        result = backend_impl.solve(design)

    return LinearSolution(_result=result, _design=design)


def _fit_glm(
    design: Design,
    family: 'str | Family',
    backend: BackendChoice,
    tol: float,
    max_iter: int,
) -> GLMSolution:
    """Fit GLM via IRLS."""
    from pystatistics.regression.families import Family, resolve_family

    family_obj = resolve_family(family) if not isinstance(family, Family) else family
    backend_impl = _get_glm_backend(backend)

    result = backend_impl.solve(design, family_obj, tol=tol, max_iter=max_iter)

    return GLMSolution(_result=result, _design=design)


# =====================================================================
# Backend selection
# =====================================================================

def _get_lm_backend(choice: BackendChoice, design: Design):
    """Select LM backend based on user choice and hardware availability."""
    if choice == 'auto':
        device = select_device('auto')
        if device.device_type == 'cuda':
            try:
                from pystatistics.regression.backends.gpu import GPUQRBackend
                return GPUQRBackend(device='cuda')
            except ImportError:
                warnings.warn(
                    "CUDA detected but PyTorch not available. "
                    "Falling back to CPU backend.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return CPUQRBackend()
        # auto + MPS -> CPU (MPS is FP32-only, not reliable for auto)
        # auto + CPU -> CPU
        return CPUQRBackend()

    elif choice in ('cpu', 'cpu_qr'):
        return CPUQRBackend()

    elif choice == 'cpu_svd':
        raise NotImplementedError("CPU SVD backend not yet implemented")

    elif choice in ('gpu', 'gpu_qr'):
        device = select_device('gpu')  # raises RuntimeError if no GPU
        from pystatistics.regression.backends.gpu import GPUQRBackend
        return GPUQRBackend(device=device.device_type)

    else:
        raise ValueError(f"Unknown backend: {choice!r}")


def _get_glm_backend(choice: BackendChoice):
    """Select GLM backend based on user choice and hardware availability."""
    from pystatistics.regression.backends.cpu_glm import CPUIRLSBackend

    if choice == 'auto':
        device = select_device('auto')
        if device.device_type == 'cuda':
            try:
                from pystatistics.regression.backends.gpu_glm import GPUIRLSBackend
                return GPUIRLSBackend(device='cuda')
            except ImportError:
                warnings.warn(
                    "CUDA detected but GPU GLM backend not available. "
                    "Falling back to CPU IRLS.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return CPUIRLSBackend()
        # auto + MPS -> CPU (MPS not auto-selected)
        # auto + CPU -> CPU
        return CPUIRLSBackend()

    elif choice in ('cpu', 'cpu_qr'):
        return CPUIRLSBackend()

    elif choice in ('gpu', 'gpu_qr'):
        device = select_device('gpu')  # raises RuntimeError if no GPU
        try:
            from pystatistics.regression.backends.gpu_glm import GPUIRLSBackend
            return GPUIRLSBackend(device=device.device_type)
        except ImportError:
            raise ImportError(
                "GPU GLM backend requires PyTorch. "
                "Install with: pip install torch"
            )

    else:
        raise ValueError(f"Unknown backend: {choice!r}")
