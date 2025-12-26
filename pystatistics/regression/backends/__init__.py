"""
Regression backends.

Available backends:
    CPUQRBackend: CPU reference implementation using QR decomposition
    GPUQRBackend: GPU implementation using PyTorch (stub)
"""

from pystatistics.regression.backends.cpu import CPUQRBackend

__all__ = [
    "CPUQRBackend",
]
