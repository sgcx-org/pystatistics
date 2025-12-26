"""
Solver dispatch for regression.

This module provides the fit() function (public API) and backend selection.
"""

from typing import Literal
import numpy as np
from numpy.typing import ArrayLike

from pystatistics.core.validation import check_array
from pystatistics.core.backends.device import select_device
from pystatistics.regression.design import RegressionDesign
from pystatistics.regression.solution import LinearSolution
from pystatistics.regression.backends.cpu import CPUQRBackend


# Type alias for backend selection
BackendChoice = Literal['auto', 'cpu', 'gpu', 'cpu_qr', 'cpu_svd', 'gpu_qr']


def fit(
    X: ArrayLike,
    y: ArrayLike,
    *,
    backend: BackendChoice = 'auto',
) -> LinearSolution:
    """
    Fit a linear regression model.
    
    Solves the ordinary least squares problem:
        min_β ||y - Xβ||²
    
    This is the primary public API for linear regression. All input validation,
    backend selection, and result wrapping happens here.
    
    Args:
        X: Design matrix (n x p). Can be any array-like.
        y: Response vector (n,). Can be any array-like.
        backend: Computational backend to use:
            - 'auto': Select best available (GPU if available, else CPU)
            - 'cpu': Use CPU backend (QR decomposition)
            - 'gpu': Use GPU backend (requires PyTorch with CUDA/MPS)
            - 'cpu_qr': Explicitly use CPU QR decomposition
            - 'cpu_svd': Use CPU SVD (handles rank-deficient X) [not yet implemented]
            - 'gpu_qr': Use GPU QR decomposition [not yet implemented]
            
    Returns:
        LinearSolution with coefficients, diagnostics, and summary methods
        
    Raises:
        ValidationError: If inputs are invalid
        DimensionError: If X and y have inconsistent dimensions
        SingularMatrixError: If X is rank-deficient (for QR backend)
        
    Example:
        >>> import numpy as np
        >>> from pystatistics.regression import fit
        >>> 
        >>> X = np.column_stack([np.ones(100), np.random.randn(100, 2)])
        >>> y = X @ [1, 2, 3] + np.random.randn(100) * 0.1
        >>> 
        >>> result = fit(X, y)
        >>> print(result.coefficients)
        >>> print(result.summary())
    """
    # === Input Validation ===
    # This is the boundary - validate here, trust everywhere else
    X_arr = check_array(X, 'X')
    y_arr = check_array(y, 'y')
    
    # Ensure y is 1D (squeeze if needed)
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr.ravel()
    
    # === Construct Design ===
    design = RegressionDesign.build(X_arr, y_arr)
    
    # === Select Backend ===
    backend_impl = _get_backend(backend, design)
    
    # === Solve ===
    result = backend_impl.solve(design)
    
    # === Wrap and Return ===
    return LinearSolution(_result=result, _design=design)


def _get_backend(choice: BackendChoice, design: RegressionDesign):
    """
    Select and instantiate the appropriate backend.
    
    Args:
        choice: User's backend preference
        design: The regression design (used for GPU capability check)
        
    Returns:
        Backend instance ready to solve
        
    Raises:
        ValueError: If unknown backend specified
        RuntimeError: If GPU requested but unavailable
    """
    if choice == 'auto':
        # Try GPU first, fall back to CPU
        device = select_device('auto')
        if device.is_gpu:
            # Check if design supports GPU tensors
            if design.supports('gpu_tensors'):
                from pystatistics.regression.backends.gpu import GPUQRBackend
                return GPUQRBackend()
        return CPUQRBackend()
    
    elif choice in ('cpu', 'cpu_qr'):
        return CPUQRBackend()
    
    elif choice == 'cpu_svd':
        raise NotImplementedError("CPU SVD backend not yet implemented")
    
    elif choice in ('gpu', 'gpu_qr'):
        from pystatistics.regression.backends.gpu import GPUQRBackend
        return GPUQRBackend()
    
    else:
        raise ValueError(f"Unknown backend: {choice!r}")
