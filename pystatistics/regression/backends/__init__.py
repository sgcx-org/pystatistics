"""
Regression backends.

Available backends:
    CPUQRBackend: CPU reference implementation using QR decomposition
    GPUQRBackend: GPU implementation using PyTorch
"""

from pystatistics.regression.backends.cpu import CPUQRBackend

__all__ = [
    "CPUQRBackend",
]

# GPU backend imported on-demand to avoid requiring PyTorch
# Use: from pystatistics.regression.backends.gpu import GPUQRBackend