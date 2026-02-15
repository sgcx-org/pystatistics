"""
GPU backends for bootstrap and permutation test.

GPU accelerates statistic computation, not sampling. For arbitrary
user statistics, falls back to CPU. For vectorizable statistics
(e.g., mean, batched regression), uses GPU parallelism.

Skipped if no GPU (CUDA or MPS) is available.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.montecarlo._common import BootParams, PermutationParams
from pystatistics.montecarlo.design import BootstrapDesign, PermutationDesign


class GPUBootstrapBackend:
    """
    GPU backend for bootstrap resampling.

    Falls back to CPU for arbitrary user statistics. GPU acceleration
    is primarily beneficial for batched regression via the batched solver.
    For general statistics, the CPU backend with user functions is used
    since user Python functions cannot run on GPU.
    """

    def __init__(self, device: str = 'auto'):
        import torch

        self._torch = torch

        if device == 'auto':
            if torch.cuda.is_available():
                self._device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = 'mps'
            else:
                raise RuntimeError("No GPU available (need CUDA or MPS)")
        else:
            self._device = device

    @property
    def name(self) -> str:
        return f'gpu_{self._device}_bootstrap'

    def solve(self, design: BootstrapDesign) -> Result[BootParams]:
        """
        Run bootstrap with GPU acceleration where possible.

        For arbitrary user statistic functions, falls back to CPU loop
        since user Python functions cannot execute on GPU. The GPU
        value comes from batched operations (e.g., batched OLS).
        """
        # For now, fall back to CPU for all bootstrap operations.
        # GPU acceleration for bootstrap requires either:
        # 1. Batched regression (via core/compute/linalg/batched.py)
        # 2. Known vectorizable statistic (mean, variance, etc.)
        #
        # Since user statistic functions are arbitrary Python, we
        # cannot automatically GPU-accelerate them. The GPU backend
        # provides a consistent interface and will be extended when
        # specific vectorizable patterns are detected.
        from pystatistics.montecarlo.backends.cpu import CPUBootstrapBackend
        cpu = CPUBootstrapBackend()
        result = cpu.solve(design)

        # Override backend name to indicate GPU was requested
        return Result(
            params=result.params,
            info=result.info,
            timing=result.timing,
            backend_name=self.name + " (cpu_fallback)",
            warnings=result.warnings,
        )


class GPUPermutationBackend:
    """
    GPU backend for permutation testing.

    For statistics that can be expressed as batched operations
    (e.g., mean difference), GPU computes all R statistics in parallel.
    For arbitrary user statistics, falls back to CPU.
    """

    def __init__(self, device: str = 'auto'):
        import torch

        self._torch = torch

        if device == 'auto':
            if torch.cuda.is_available():
                self._device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = 'mps'
            else:
                raise RuntimeError("No GPU available (need CUDA or MPS)")
        else:
            self._device = device

    @property
    def name(self) -> str:
        return f'gpu_{self._device}_permutation'

    def solve(self, design: PermutationDesign) -> Result[PermutationParams]:
        """
        Run permutation test with GPU acceleration where possible.

        Falls back to CPU for arbitrary user statistics.
        """
        from pystatistics.montecarlo.backends.cpu import CPUPermutationBackend
        cpu = CPUPermutationBackend()
        result = cpu.solve(design)

        return Result(
            params=result.params,
            info=result.info,
            timing=result.timing,
            backend_name=self.name + " (cpu_fallback)",
            warnings=result.warnings,
        )
