"""
Solver dispatch for regression.
"""

import warnings
from typing import Literal
import numpy as np
from numpy.typing import ArrayLike

from pystatistics.core.compute.device import select_device
from pystatistics.regression.design import Design
from pystatistics.regression.solution import LinearSolution
from pystatistics.regression.backends.cpu import CPUQRBackend


BackendChoice = Literal['auto', 'cpu', 'gpu', 'cpu_qr', 'cpu_svd', 'gpu_qr']


def fit(
    X_or_design: ArrayLike | Design,
    y: ArrayLike | None = None,
    *,
    backend: BackendChoice = 'auto',
    force: bool = False,
) -> LinearSolution:
    """
    Fit a linear regression model.

    Accepts EITHER:
        1. A Design object (from DataSource or arrays)
        2. Raw X and y arrays (convenience)

    Args:
        X_or_design: Design object or X matrix
        y: Response vector (required if X_or_design is array)
        backend: 'auto', 'cpu', 'gpu', 'cpu_qr', 'cpu_svd', 'gpu_qr'
        force: If True, proceed with GPU Cholesky even on ill-conditioned
            matrices. Has no effect on CPU QR backend.

    Returns:
        LinearSolution

    Examples:
        # From Design (recommended)
        >>> design = Design.from_datasource(ds, y='target')
        >>> result = fit(design)

        # From arrays (convenience)
        >>> result = fit(X, y)
    """
    # Get or build Design
    if isinstance(X_or_design, Design):
        design = X_or_design
    else:
        if y is None:
            raise ValueError("y required when passing arrays")
        design = Design.from_arrays(np.asarray(X_or_design), np.asarray(y))

    # Select backend
    backend_impl = _get_backend(backend, design)

    # Solve — pass force to GPU backends, CPU ignores it
    if hasattr(backend_impl, 'solve') and 'force' in backend_impl.solve.__code__.co_varnames:
        result = backend_impl.solve(design, force=force)
    else:
        result = backend_impl.solve(design)

    return LinearSolution(_result=result, _design=design)


def _get_backend(choice: BackendChoice, design: Design):
    """Select backend based on user choice and hardware availability."""
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
