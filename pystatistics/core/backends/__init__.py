"""
Shared backend infrastructure for PyStatistics.

This module provides hardware detection, timing utilities, and linear algebra
primitives that are shared across all domain-specific backends.

Submodules:
    device: Hardware detection and device selection
    timing: Execution timing utilities
    precision: Numerical precision constants and utilities
    linalg: Linear algebra primitives (QR, Cholesky, SVD, etc.)
    optimization: Convergence checking and optimization utilities
"""

from pystatistics.core.backends.device import (
    DeviceInfo,
    detect_gpu,
    get_cpu_info,
    select_device,
)
from pystatistics.core.backends.timing import Timer, timed

__all__ = [
    # Device detection
    "DeviceInfo",
    "detect_gpu",
    "get_cpu_info",
    "select_device",
    # Timing
    "Timer",
    "timed",
]
