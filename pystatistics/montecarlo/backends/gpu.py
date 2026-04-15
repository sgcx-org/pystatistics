"""
GPU backends for bootstrap and permutation test.

GPU accelerates statistic computation, not sampling. For arbitrary
user statistics, falls back to CPU. For vectorizable statistics
(e.g., mean difference), computes all R replicates in parallel.

Skipped if no GPU (CUDA or MPS) is available.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.montecarlo._common import BootParams, PermutationParams
from pystatistics.montecarlo.design import BootstrapDesign, PermutationDesign


def _select_device():
    """Select GPU device. Raises RuntimeError if none available."""
    import torch

    if torch.cuda.is_available():
        return torch.device('cuda'), torch
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps'), torch
    raise RuntimeError("No GPU available (need CUDA or MPS)")


class GPUBootstrapBackend:
    """
    GPU backend for bootstrap resampling.

    GPU acceleration is primarily beneficial for batched regression
    via the batched solver. For general statistics, the CPU backend
    with user functions is used since arbitrary Python callables
    cannot run on GPU.
    """

    def __init__(self, device: str = 'auto'):
        self._device, self._torch = _select_device()

    @property
    def name(self) -> str:
        return f'gpu_{self._device.type}_bootstrap'

    def solve(self, design: BootstrapDesign) -> Result[BootParams]:
        """Run bootstrap with GPU acceleration for simple statistics.

        For 1-D data with stype='i' and a mean-like statistic, generates
        all R bootstrap samples on GPU and computes statistics vectorized.
        For arbitrary or complex statistics, falls back to CPU.
        """
        torch = self._torch

        # Constraints: only ordinary bootstrap, stype='i', no strata,
        # 1-D data, and statistic must be vectorizable (mean).
        can_gpu = (
            design.sim == "ordinary"
            and design.stype == "i"
            and design.strata is None
            and design.data.ndim == 1
        )

        if not can_gpu:
            from pystatistics.montecarlo.backends.cpu import CPUBootstrapBackend
            return CPUBootstrapBackend().solve(design)

        timer = Timer()
        timer.start()

        data = design.data
        statistic = design.statistic
        R = design.R
        seed = design.seed
        n = data.shape[0]

        # Compute t0 on CPU (user function)
        with timer.section('t0_computation'):
            original_indices = np.arange(n)
            t0 = np.atleast_1d(np.asarray(
                statistic(data, original_indices), dtype=np.float64,
            ))
            k = len(t0)

        # Verify: is statistic(data, indices) == mean(data[indices])?
        # Check on one bootstrap sample.
        with timer.section('statistic_check'):
            rng_check = np.random.default_rng(seed)
            test_idx = rng_check.choice(n, size=n, replace=True)
            user_val = np.atleast_1d(np.asarray(
                statistic(data, test_idx), dtype=np.float64,
            ))

            if k == 1:
                gpu_val = np.mean(data[test_idx])
                is_mean = abs(user_val[0] - gpu_val) < 1e-10 * (
                    abs(user_val[0]) + 1e-10
                )
            else:
                is_mean = False

        if not is_mean:
            # Statistic is not simple mean — fall back to CPU.
            timer.stop()
            from pystatistics.montecarlo.backends.cpu import CPUBootstrapBackend
            return CPUBootstrapBackend().solve(design)

        # GPU path: generate all R bootstrap index sets, compute means.
        with timer.section('gpu_compute'):
            device = self._device
            dtype = torch.float32 if device.type == 'mps' else torch.float64

            data_t = torch.from_numpy(data).to(device=device, dtype=dtype)

            # Memory: R * n * 8 bytes for indices. Chunk if needed.
            target_bytes = 1_000_000_000
            chunk_size = max(1, target_bytes // (n * 8))
            chunk_size = min(chunk_size, R)

            t = np.empty((R, k), dtype=np.float64)

            # NON-DETERMINISTIC: GPU RNG differs from CPU. Bootstrap
            # replicates are statistically equivalent but not identical.
            if seed is not None:
                torch.manual_seed(seed)

            offset = 0
            while offset < R:
                batch = min(chunk_size, R - offset)

                # Generate random indices via randint on GPU
                idx = torch.randint(0, n, (batch, n), device=device)
                sampled = data_t[idx]            # (batch, n)
                means = sampled.mean(dim=1)      # (batch,)

                t[offset:offset + batch, 0] = (
                    means.cpu().numpy().astype(np.float64)
                )
                offset += batch

        with timer.section('summary_statistics'):
            bias = np.mean(t, axis=0) - t0
            se = np.std(t, axis=0, ddof=1)

        timer.stop()

        return Result(
            params=BootParams(
                t0=t0, t=t, R=R, bias=bias, se=se,
                ci=None, ci_conf_level=None,
            ),
            info={
                'sim': design.sim, 'stype': design.stype,
                'n': n, 'k': k, 'gpu_vectorized': True,
            },
            timing=timer.result(),
            backend_name=self.name,
            warnings=(),
        )


class GPUPermutationBackend:
    """
    GPU backend for permutation testing.

    For 1-D data with a mean-difference-like statistic, generates all R
    permutations on GPU and computes all statistics in a single vectorized
    operation. For arbitrary or multi-dimensional statistics, falls back
    to the CPU backend.
    """

    def __init__(self, device: str = 'auto'):
        self._device, self._torch = _select_device()

    @property
    def name(self) -> str:
        return f'gpu_{self._device.type}_permutation'

    def solve(self, design: PermutationDesign) -> Result[PermutationParams]:
        """Run permutation test with GPU acceleration.

        For 1-D data with mean-difference statistic: generates permutations
        in chunks, computes all statistics vectorized on GPU. Falls back to
        CPU for multi-dimensional data or non-mean-difference statistics.
        """
        torch = self._torch
        timer = Timer()
        timer.start()

        x = design.x
        y = design.y
        statistic = design.statistic
        R = design.R
        alternative = design.alternative
        seed = design.seed

        # Multi-dimensional data: fall back to CPU.
        if x.ndim > 1 or y.ndim > 1:
            from pystatistics.montecarlo.backends.cpu import CPUPermutationBackend
            return CPUPermutationBackend().solve(design)

        n1 = len(x)
        n2 = len(y)
        n = n1 + n2
        combined = np.concatenate([x, y])

        with timer.section('observed_stat'):
            observed = float(statistic(x, y))

        # Verify statistic is mean-difference before going GPU path.
        # Check on one permutation.
        with timer.section('statistic_check'):
            rng_check = np.random.default_rng(seed)
            perm_check = rng_check.permutation(n)
            user_val = float(statistic(
                combined[perm_check[:n1]], combined[perm_check[n1:]],
            ))
            gpu_val = (
                combined[perm_check[:n1]].mean()
                - combined[perm_check[n1:]].mean()
            )
            if abs(user_val - gpu_val) > 1e-4 * (abs(user_val) + 1e-10):
                timer.stop()
                from pystatistics.montecarlo.backends.cpu import CPUPermutationBackend
                return CPUPermutationBackend().solve(design)

        # GPU path: generate random permutations directly on GPU using
        # random-key sorting (torch.rand + argsort). Chunked to fit VRAM.
        # Memory per chunk: chunk_size * n * 4 bytes (float32 keys)
        with timer.section('gpu_compute'):
            device = self._device
            dtype = torch.float32 if device.type == 'mps' else torch.float64

            # Chunk size: keep keys matrix under ~1 GB
            target_bytes = 1_000_000_000
            chunk_size = max(1, target_bytes // (n * 4))
            chunk_size = min(chunk_size, R)

            combined_t = torch.from_numpy(combined).to(
                device=device, dtype=dtype,
            )
            total_sum = combined_t.sum()

            perm_stats = np.empty(R, dtype=np.float64)

            # NON-DETERMINISTIC: GPU RNG differs from CPU RNG. P-values
            # are statistically equivalent but not identical to CPU path.
            if seed is not None:
                torch.manual_seed(seed)

            offset = 0
            while offset < R:
                batch = min(chunk_size, R - offset)

                # Random-key sort: generate uniform keys, argsort gives
                # a random permutation per row — fully parallel on GPU.
                keys = torch.rand(batch, n, device=device)
                perm_idx = keys.argsort(dim=1)  # (batch, n)

                # Only need group-1 indices for mean-difference
                idx_g1 = perm_idx[:, :n1]                  # (batch, n1)
                gathered = combined_t[idx_g1]              # (batch, n1)
                sum_g1 = gathered.sum(dim=1)               # (batch,)
                stats_t = sum_g1 / n1 - (total_sum - sum_g1) / n2

                perm_stats[offset:offset + batch] = (
                    stats_t.cpu().numpy().astype(np.float64)
                )
                offset += batch

        with timer.section('p_value'):
            if alternative == "two.sided":
                count = int(np.sum(np.abs(perm_stats) >= np.abs(observed)))
            elif alternative == "greater":
                count = int(np.sum(perm_stats >= observed))
            elif alternative == "less":
                count = int(np.sum(perm_stats <= observed))
            else:
                raise ValueError(f"Unknown alternative: {alternative!r}")

            p_value = float(count + 1) / float(R + 1)

        timer.stop()

        return Result(
            params=PermutationParams(
                observed_stat=observed,
                perm_stats=perm_stats,
                p_value=p_value,
                R=R,
                alternative=alternative,
            ),
            info={
                'n1': n1, 'n2': n2,
                'alternative': alternative,
                'gpu_vectorized': True,
                'chunk_size': chunk_size,
            },
            timing=timer.result(),
            backend_name=self.name,
            warnings=(),
        )
