"""
GPU backends for bootstrap and permutation test.

The GPU vectorizes ONE closed-form statistic each: the sample mean (bootstrap)
and the difference in means (permutation). It never *infers* whether the user's
statistic is that closed form — the caller declares it explicitly via
``gpu_statistic`` (see :mod:`pystatistics.montecarlo.solvers`). The dispatcher
only routes a design here once it is declared and vectorizable, so these
backends never fall back and never guess. As a final fail-loud guard, the
declaration is verified against the observed statistic on the original data
before any GPU work — a declared-but-wrong statistic raises, it is never
silently replaced by the mean.

Skipped if no GPU (CUDA or MPS) is available.
"""

from __future__ import annotations

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.compute.backend import NO_GPU_MSG

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.montecarlo._common import BootParams, PermutationParams
from pystatistics.montecarlo.design import BootstrapDesign, PermutationDesign


# Relative tolerance for verifying a declared GPU statistic against the observed
# value on the original data. The observed statistic is evaluated in fp64 on the
# CPU, so a genuine mean matches to well within this bound; a materially
# different statistic (a trimmed mean, a different estimator) does not.
_DECLARATION_RTOL = 1e-9


def _select_device():
    """Select GPU device. Raises RuntimeError if none available."""
    import torch

    if torch.cuda.is_available():
        return torch.device('cuda'), torch
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps'), torch
    raise RuntimeError(NO_GPU_MSG)


class GPUBootstrapBackend:
    """
    GPU backend for bootstrap resampling of a declared-mean statistic.

    Generates all R ordinary bootstrap index sets on the GPU and reduces them
    to sample means in parallel. Only reached when the caller has declared
    ``gpu_statistic='mean'`` on a vectorizable design; arbitrary Python
    statistics run on the CPU backend (they cannot execute on the GPU).
    """

    def __init__(self, device: str = 'auto'):
        self._device, self._torch = _select_device()

    @property
    def name(self) -> str:
        return f'gpu_{self._device.type}_bootstrap'

    def solve(self, design: BootstrapDesign) -> Result[BootParams]:
        """Run the vectorized GPU bootstrap for a declared-mean statistic.

        The dispatcher only routes a design here once ``gpu_statistic='mean'``
        is declared and the configuration is vectorizable (ordinary, stype='i',
        no strata, 1-D data), so no probing or CPU fallback happens here. The
        declaration is verified against the observed statistic on the original
        data (fail-loud) before any GPU work — a statistic that is not the mean
        raises rather than being silently computed as the mean.
        """
        torch = self._torch

        timer = Timer()
        timer.start()

        data = design.data
        statistic = design.statistic
        R = design.R
        seed = design.seed
        n = data.shape[0]

        # Compute t0 on CPU (user function) and verify the declaration.
        with timer.section('t0_computation'):
            original_indices = np.arange(n)
            t0 = np.atleast_1d(np.asarray(
                statistic(data, original_indices), dtype=np.float64,
            ))
            k = len(t0)

        with timer.section('declaration_check'):
            mean_all = float(np.mean(data))
            if k != 1 or abs(t0[0] - mean_all) > _DECLARATION_RTOL * (
                abs(mean_all) + _DECLARATION_RTOL
            ):
                raise ValidationError(
                    "gpu_statistic='mean' was declared, but statistic(data, "
                    "all-indices) does not equal mean(data) "
                    f"(got {t0.tolist()}, expected {mean_all}). The GPU path "
                    "computes the sample mean; refusing to silently compute a "
                    "different quantity. Use backend='cpu' for this statistic."
                )

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

            batch_start = 0
            while batch_start < R:
                batch = min(chunk_size, R - batch_start)

                # Generate random indices via randint on GPU
                idx = torch.randint(0, n, (batch, n), device=device)
                sampled = data_t[idx]            # (batch, n)
                means = sampled.mean(dim=1)      # (batch,)

                t[batch_start:batch_start + batch, 0] = (
                    means.cpu().numpy().astype(np.float64)
                )
                batch_start += batch

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
    GPU backend for permutation testing of a declared mean-difference statistic.

    Generates all R permutations on the GPU (random-key argsort) and computes
    the difference in means for each in a single vectorized operation. Only
    reached when the caller has declared ``gpu_statistic='mean_diff'`` on 1-D
    groups; arbitrary Python statistics run on the CPU backend.
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

        n1 = len(x)
        n2 = len(y)
        n = n1 + n2
        combined = np.concatenate([x, y])

        with timer.section('observed_stat'):
            observed = float(statistic(x, y))

        # Verify the gpu_statistic='mean_diff' declaration against the observed
        # statistic (fail-loud). The dispatcher already guaranteed 1-D groups
        # and the declaration; this refuses to silently compute a different
        # quantity than the user's statistic.
        with timer.section('declaration_check'):
            mean_diff_obs = float(x.mean() - y.mean())
            if abs(observed - mean_diff_obs) > _DECLARATION_RTOL * (
                abs(mean_diff_obs) + _DECLARATION_RTOL
            ):
                raise ValidationError(
                    "gpu_statistic='mean_diff' was declared, but statistic(x, y) "
                    f"does not equal mean(x)-mean(y) (got {observed}, expected "
                    f"{mean_diff_obs}). The GPU path computes the difference in "
                    "means; refusing to silently compute a different quantity. "
                    "Use backend='cpu' for this statistic."
                )

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

            batch_start = 0
            while batch_start < R:
                batch = min(chunk_size, R - batch_start)

                # Random-key sort: generate uniform keys, argsort gives
                # a random permutation per row — fully parallel on GPU.
                keys = torch.rand(batch, n, device=device)
                perm_idx = keys.argsort(dim=1)  # (batch, n)

                # Only need group-1 indices for mean-difference
                idx_g1 = perm_idx[:, :n1]                  # (batch, n1)
                gathered = combined_t[idx_g1]              # (batch, n1)
                sum_g1 = gathered.sum(dim=1)               # (batch,)
                stats_t = sum_g1 / n1 - (total_sum - sum_g1) / n2

                perm_stats[batch_start:batch_start + batch] = (
                    stats_t.cpu().numpy().astype(np.float64)
                )
                batch_start += batch

        with timer.section('p_value'):
            if alternative == "two-sided":
                count = int(np.sum(np.abs(perm_stats) >= np.abs(observed)))
            elif alternative == "greater":
                count = int(np.sum(perm_stats >= observed))
            elif alternative == "less":
                count = int(np.sum(perm_stats <= observed))
            else:
                raise ValidationError(f"Unknown alternative: {alternative!r}")

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
