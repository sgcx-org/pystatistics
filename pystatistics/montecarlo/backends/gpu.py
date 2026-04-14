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

        Raises NotImplementedError because GPU bootstrap is not yet
        implemented. The solver dispatches to CPU for backend='auto';
        this method is only reached when backend='gpu' was explicitly
        requested.
        """
        raise NotImplementedError(
            "GPU acceleration for bootstrap is not yet implemented. "
            "Use backend='cpu'."
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

        Raises NotImplementedError because GPU permutation testing is
        not yet implemented. The solver dispatches to CPU for
        backend='auto'; this method is only reached when backend='gpu'
        was explicitly requested.
        """
        raise NotImplementedError(
            "GPU acceleration for permutation testing is not yet implemented. "
            "Use backend='cpu'."
        )
