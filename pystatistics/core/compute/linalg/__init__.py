"""
Linear algebra kernels.
"""

from pystatistics.core.compute.linalg.qr import (
    QRResult,
    qr_decompose,
    qr_solve,
)

__all__ = [
    "QRResult",
    "qr_decompose",
    "qr_solve",
]