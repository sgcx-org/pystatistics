"""
Regression backends.

Available backends:
    CPUQRBackend: CPU reference implementation using QR decomposition (LM)
    GPUQRBackend: GPU implementation using PyTorch (LM)
    CPUIRLSBackend: CPU IRLS implementation (GLM)
    GPUIRLSBackend: GPU IRLS implementation (GLM)
"""

from pystatistics.regression.backends.cpu import CPUQRBackend
from pystatistics.regression.backends.cpu_glm import CPUIRLSBackend

__all__ = [
    "CPUQRBackend",
    "CPUIRLSBackend",
]

# GPU backends imported on-demand to avoid requiring PyTorch
# Use: from pystatistics.regression.backends.gpu import GPUQRBackend
# Use: from pystatistics.regression.backends.gpu_glm import GPUIRLSBackend