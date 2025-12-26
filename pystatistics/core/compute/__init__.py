"""
Shared compute infrastructure for PyStatistics.

This module provides hardware detection, timing utilities, and linear algebra
kernels that are shared across all domain-specific backends.

IMPORTANT: This is NOT where domain-specific backends live. Those go in
{domain}/backends/. This module contains shared NUMERIC infrastructure.

Submodules:
    device: Hardware detection and device selection
    timing: Execution timing utilities
    precision: Numerical precision constants and utilities
    linalg: Linear algebra kernels (QR, Cholesky, SVD, etc.)
"""

from pystatistics.core.compute.device import (
    DeviceInfo,
    detect_gpu,
    get_cpu_info,
    select_device,
)
from pystatistics.core.compute.timing import Timer, timed

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
