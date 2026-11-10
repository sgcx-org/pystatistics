"""
Objective functions for MVN MLE optimization.

Forward-Cholesky objectives (NumPy and PyTorch) share the data preparation in
``base.py``. The inverse-Cholesky reference objective (``cpu.py``) and its
data preparation (``_r_port_base.py``) are a port of the R package ``mvnmle``.
"""

from .cpu import CPUObjectiveFP64
from .parameterizations import (
    InverseCholeskyParameterization,
    CholeskyParameterization,
    BoundedCholeskyParameterization,
)

__all__ = [
    'CPUObjectiveFP64',
    'InverseCholeskyParameterization',
    'CholeskyParameterization',
    'BoundedCholeskyParameterization',
]

# GPU objectives are imported lazily to avoid torch dependency
def _get_gpu_fp32():
    from .gpu_fp32 import GPUObjectiveFP32
    return GPUObjectiveFP32

def _get_gpu_fp64():
    from .gpu_fp64 import GPUObjectiveFP64
    return GPUObjectiveFP64
