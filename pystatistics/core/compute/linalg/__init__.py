"""
Linear algebra kernels.
"""

from pystatistics.core.compute.linalg.qr import (
    QRResult,
    qr_decompose,
    qr_solve,
)
from pystatistics.core.compute.linalg.triangular import (
    batched_tri_inv_series,
    use_blocked_inverse,
)

__all__ = [
    "QRResult",
    "qr_decompose",
    "qr_solve",
    "batched_tri_inv_series",
    "use_blocked_inverse",
]