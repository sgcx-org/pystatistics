"""
Objective functions for MVN MLE optimization.

Provides CPU and GPU implementations of the negative log-likelihood
objective with R-compatible pattern ordering.
"""

from .cpu import CPUObjectiveFP64
from .parameterizations import (
    get_parameterization,
    InverseCholeskyParameterization,
    CholeskyParameterization,
    BoundedCholeskyParameterization,
)

__all__ = [
    'CPUObjectiveFP64',
    'get_parameterization',
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
