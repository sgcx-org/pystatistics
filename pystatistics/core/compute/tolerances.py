"""
Tolerance tiers for numerical validation.

Defines precision expectations for different compute paths:
- CPU FP64 (reference): machine precision match with R
- GPU FP64: same as CPU
- GPU FP32: relaxed for single-precision arithmetic
- MPS FP32: same as GPU FP32

Used by test suite, benchmarks, and the GPU backend's condition check.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToleranceTier:
    """Tolerance specification for numerical comparison."""
    rtol: float
    atol: float
    name: str
    description: str


# CPU reference: must match R to machine precision
CPU_FP64 = ToleranceTier(
    rtol=1e-10,
    atol=1e-12,
    name='cpu_fp64',
    description='CPU double precision — matches R exactly',
)

# CPU reference, ill-conditioned problems (cond > 1e4)
CPU_FP64_ILL_CONDITIONED = ToleranceTier(
    rtol=1e-4,
    atol=1e-6,
    name='cpu_fp64_ill_conditioned',
    description='CPU double precision, ill-conditioned (cond > 1e4)',
)

# GPU with FP64 (compute GPUs or explicit request)
GPU_FP64 = ToleranceTier(
    rtol=1e-10,
    atol=1e-12,
    name='gpu_fp64',
    description='GPU double precision — matches CPU reference',
)

# GPU with FP32 (consumer GPUs, default)
GPU_FP32 = ToleranceTier(
    rtol=1e-4,
    atol=1e-5,
    name='gpu_fp32',
    description='GPU single precision — statistically equivalent',
)

# GPU FP32, ill-conditioned problems
GPU_FP32_ILL_CONDITIONED = ToleranceTier(
    rtol=1e-2,
    atol=1e-3,
    name='gpu_fp32_ill_conditioned',
    description='GPU single precision, ill-conditioned',
)

# MPS (Apple Silicon GPU) — same as GPU FP32
MPS_FP32 = GPU_FP32

# Condition number threshold for GPU Cholesky safety check.
# At cond(X) = 1e6, cond(X'X) = 1e12 — near float64 epsilon
# and well past float32 usability.
GPU_CONDITION_THRESHOLD = 1e6


def select_tolerance(
    backend_name: str,
    is_ill_conditioned: bool = False,
) -> ToleranceTier:
    """Select appropriate tolerance tier for a given backend."""
    if 'gpu' in backend_name:
        if 'fp64' in backend_name:
            return GPU_FP64
        if is_ill_conditioned:
            return GPU_FP32_ILL_CONDITIONED
        return GPU_FP32
    else:
        if is_ill_conditioned:
            return CPU_FP64_ILL_CONDITIONED
        return CPU_FP64
