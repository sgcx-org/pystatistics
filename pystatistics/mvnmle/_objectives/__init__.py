"""
Objective functions for MVN MLE optimization.

The forward-Cholesky objectives (``numpy_fp64`` in NumPy; ``gpu_fp64`` and
``gpu_fp32`` in PyTorch) share the data preparation in ``base.py``.
"""

from .parameterizations import (
    CholeskyParameterization,
    BoundedCholeskyParameterization,
)

__all__ = [
    'CholeskyParameterization',
    'BoundedCholeskyParameterization',
]

# PyTorch objectives are imported lazily to avoid a torch dependency
def _get_gpu_fp32():
    from .gpu_fp32 import GPUObjectiveFP32
    return GPUObjectiveFP32

def _get_gpu_fp64():
    from .gpu_fp64 import GPUObjectiveFP64
    return GPUObjectiveFP64
