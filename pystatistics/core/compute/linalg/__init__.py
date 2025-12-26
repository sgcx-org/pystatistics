"""
Linear algebra kernels for PyStatistics.

This module provides CPU and GPU implementations of core linear algebra
operations used across all statistical domains.

All functions follow these conventions:
    - CPU functions use NumPy/SciPy (LAPACK under the hood)
    - GPU functions use PyTorch and return NumPy arrays (data moved to CPU)
    - Each operation returns a structured result dataclass
    - Errors are raised immediately with clear messages

Submodules:
    qr: QR decomposition
    cholesky: Cholesky decomposition (stub)
    svd: Singular value decomposition (stub)
    solve: Triangular and symmetric solvers (stub)
    determinant: Log-determinant computation (stub)
"""

from pystatistics.core.compute.linalg.qr import (
    QRResult,
    qr_cpu,
    qr_gpu,
    qr_solve_cpu,
    qr_solve_gpu,
)

__all__ = [
    # QR decomposition
    "QRResult",
    "qr_cpu",
    "qr_gpu",
    "qr_solve_cpu",
    "qr_solve_gpu",
]
