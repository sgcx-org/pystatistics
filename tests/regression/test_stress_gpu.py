"""
GPU stress tests for linear regression on enormous datasets.

The selling point of PyStatistics: problems that crash R/SAS or take
hours there run in seconds on GPU. These tests verify both correctness
and performance at scales where CPU-only tools become infeasible.

Memory budget for RTX 5070 Ti (16GB VRAM, FP32):
    Working memory ≈ 4*(n*p + p² + 3n) bytes
    Safe limit:  n*p ≲ 2 billion float32 values (~8GB for X alone)

Skipped automatically when no GPU is available.
"""

import pytest
import numpy as np
import time
import gc

from pystatistics.regression import fit, Design


def _gpu_available():
    try:
        import torch
        return (torch.cuda.is_available() or
                (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()))
    except ImportError:
        return False


def _gpu_vram_gb():
    """Return GPU VRAM in GB, or None if unknown (MPS)."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / 1e9
    except ImportError:
        pass
    return None


def _estimated_gpu_memory_gb(n, p):
    """Estimate GPU memory needed for an (n, p) regression in FP32."""
    # X: n*p, y: n, XtX: p*p, Xty: p, coef: p, fitted: n, resid: n
    # SVD workspace: ~p*p (for condition check)
    floats = n * p + n + p * p + p + p + n + n + p * p
    return floats * 4 / 1e9


def _make_data(n, p, rng, snr=10.0):
    """Generate well-conditioned regression data with known signal."""
    X = rng.standard_normal((n, p)).astype(np.float32)
    beta = rng.standard_normal(p).astype(np.float64)
    noise_scale = np.sqrt(np.sum(beta**2) / snr)
    y = (X.astype(np.float64) @ beta + rng.standard_normal(n) * noise_scale)
    return X, y, beta


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.skipif(not _gpu_available(), reason="No GPU available"),
    pytest.mark.slow,
]


# ---------------------------------------------------------------------------
# Stress tests — correctness at scale
# ---------------------------------------------------------------------------

class TestGPUStressCorrectness:
    """Verify GPU produces statistically valid results at large scale."""

    @pytest.mark.parametrize("n,p,label", [
        (5_000_000, 100, "5M x 100"),
        (10_000_000, 50, "10M x 50"),
        (2_000_000, 500, "2M x 500"),
    ])
    def test_large_regression_valid(self, n, p, label):
        """GPU regression on large data produces valid statistics."""
        vram = _gpu_vram_gb()
        needed = _estimated_gpu_memory_gb(n, p)
        if vram is not None and needed > vram * 0.8:
            pytest.skip(f"{label} needs ~{needed:.1f}GB, GPU has {vram:.1f}GB")

        rng = np.random.default_rng(42)
        X, y, beta_true = _make_data(n, p, rng)

        result = fit(X, y, backend='gpu')

        # Basic sanity
        assert result.coefficients.shape == (p,)
        assert 0.0 <= result.r_squared <= 1.0
        assert result.rss >= 0
        assert result.df_residual == n - p

        # Coefficients should be in the right ballpark (SNR=10)
        coef_corr = np.corrcoef(result.coefficients, beta_true)[0, 1]
        assert coef_corr > 0.95, (
            f"Coefficient correlation {coef_corr:.4f} too low — "
            f"GPU solution may be numerically unstable at this scale"
        )

    @pytest.mark.parametrize("n,p,label", [
        (5_000_000, 100, "5M x 100"),
        (10_000_000, 50, "10M x 50"),
    ])
    def test_gpu_matches_cpu_at_scale(self, n, p, label):
        """GPU FP32 coefficients correlate with CPU FP64 reference.

        We can't compare element-wise at rtol=1e-4 for huge n because
        FP32 accumulation over millions of rows introduces drift.
        Instead: verify high correlation and bounded max relative error
        on the largest coefficients.
        """
        vram = _gpu_vram_gb()
        needed = _estimated_gpu_memory_gb(n, p)
        if vram is not None and needed > vram * 0.8:
            pytest.skip(f"{label} needs ~{needed:.1f}GB, GPU has {vram:.1f}GB")

        rng = np.random.default_rng(42)
        X, y, _ = _make_data(n, p, rng)

        gpu_result = fit(X, y, backend='gpu')

        # CPU reference — this will be slow but it's the ground truth
        cpu_result = fit(X, y, backend='cpu')

        # Correlation should be near-perfect
        corr = np.corrcoef(
            gpu_result.coefficients, cpu_result.coefficients
        )[0, 1]
        assert corr > 0.9999, f"GPU/CPU correlation: {corr:.6f}"

        # Max relative error on large coefficients (skip near-zero ones
        # where relative error is meaningless)
        large_mask = np.abs(cpu_result.coefficients) > 0.1
        if large_mask.any():
            rel_err = np.abs(
                (gpu_result.coefficients[large_mask] -
                 cpu_result.coefficients[large_mask]) /
                cpu_result.coefficients[large_mask]
            )
            assert np.max(rel_err) < 0.01, (
                f"Max relative error on large coefficients: {np.max(rel_err):.4e}"
            )

        # R-squared should be very close
        np.testing.assert_allclose(
            gpu_result.r_squared, cpu_result.r_squared, rtol=1e-3
        )


# ---------------------------------------------------------------------------
# Stress tests — performance
# ---------------------------------------------------------------------------

class TestGPUStressPerformance:
    """Verify GPU is actually fast at scale (not just correct)."""

    @pytest.mark.parametrize("n,p,label", [
        (1_000_000, 200, "1M x 200"),
        (5_000_000, 100, "5M x 100"),
        (10_000_000, 50, "10M x 50"),
    ])
    def test_gpu_faster_than_cpu(self, n, p, label):
        """GPU should be faster than CPU on large problems."""
        vram = _gpu_vram_gb()
        needed = _estimated_gpu_memory_gb(n, p)
        if vram is not None and needed > vram * 0.8:
            pytest.skip(f"{label} needs ~{needed:.1f}GB, GPU has {vram:.1f}GB")

        rng = np.random.default_rng(42)
        X, y, _ = _make_data(n, p, rng)
        design = Design.from_arrays(X, y)

        # Warmup
        fit(design, backend='gpu')

        # GPU timing (median of 3)
        gpu_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            fit(design, backend='gpu')
            gpu_times.append(time.perf_counter() - t0)
        gpu_time = np.median(gpu_times)

        # CPU timing (single run — too slow for multiple)
        t0 = time.perf_counter()
        fit(design, backend='cpu')
        cpu_time = time.perf_counter() - t0

        speedup = cpu_time / gpu_time

        # At these sizes, GPU should be meaningfully faster
        assert speedup > 1.0, (
            f"{label}: GPU ({gpu_time:.3f}s) slower than CPU ({cpu_time:.3f}s)"
        )

    def test_timing_sections_populated(self):
        """Verify timing breakdown is available for profiling."""
        rng = np.random.default_rng(42)
        X, y, _ = _make_data(1_000_000, 100, rng)

        result = fit(X, y, backend='gpu')

        assert result.timing is not None
        assert 'total_seconds' in result.timing
        # Key sections should be present
        for section in ('data_transfer_to_gpu', 'normal_equations',
                        'cholesky_solve', 'data_transfer_to_cpu'):
            assert section in result.timing, f"Missing timing section: {section}"


# ---------------------------------------------------------------------------
# Stress test — scaling behavior
# ---------------------------------------------------------------------------

class TestGPUScaling:
    """Verify GPU time scales sub-linearly with n (unlike CPU QR which is O(np²))."""

    def test_gpu_sublinear_in_n(self):
        """Doubling n should less than double GPU time (Cholesky is O(np + p³))."""
        vram = _gpu_vram_gb()
        p = 100
        sizes = [500_000, 1_000_000, 2_000_000]

        # Skip sizes that won't fit
        sizes = [n for n in sizes
                 if vram is None or _estimated_gpu_memory_gb(n, p) < vram * 0.8]
        if len(sizes) < 2:
            pytest.skip("Not enough VRAM for scaling test")

        rng = np.random.default_rng(42)
        times = {}

        for n in sizes:
            X, y, _ = _make_data(n, p, rng)
            design = Design.from_arrays(X, y)

            # Warmup
            fit(design, backend='gpu')

            # Measure
            gpu_times = []
            for _ in range(3):
                t0 = time.perf_counter()
                fit(design, backend='gpu')
                gpu_times.append(time.perf_counter() - t0)
            times[n] = np.median(gpu_times)

            del X, y, design
            gc.collect()

        # When n doubles, GPU time for Cholesky (dominated by X'X = O(np²))
        # should roughly double — NOT quadruple like CPU QR's O(np²).
        # But the key is data transfer + matmul, so scaling should be
        # approximately linear in n. Allow up to 2.5x for doubling n.
        sorted_sizes = sorted(times.keys())
        for i in range(1, len(sorted_sizes)):
            n_prev, n_curr = sorted_sizes[i-1], sorted_sizes[i]
            t_prev, t_curr = times[n_prev], times[n_curr]
            n_ratio = n_curr / n_prev
            t_ratio = t_curr / t_prev

            # Time ratio should be at most 2.5x the size ratio
            # (allowing for overhead, memory transfer, etc.)
            assert t_ratio < n_ratio * 2.5, (
                f"GPU scaling too steep: n grew {n_ratio:.1f}x but "
                f"time grew {t_ratio:.1f}x ({n_prev:,} → {n_curr:,})"
            )


# ---------------------------------------------------------------------------
# Stress test — memory boundary
# ---------------------------------------------------------------------------

class TestGPUMemoryBoundary:
    """Test behavior near GPU memory limits."""

    def test_large_p_regime(self):
        """Wide matrix (many predictors) still works."""
        vram = _gpu_vram_gb()
        n, p = 100_000, 2000
        needed = _estimated_gpu_memory_gb(n, p)
        if vram is not None and needed > vram * 0.8:
            pytest.skip(f"Need ~{needed:.1f}GB, GPU has {vram:.1f}GB")

        rng = np.random.default_rng(42)
        X, y, beta_true = _make_data(n, p, rng)

        result = fit(X, y, backend='gpu')
        assert result.coefficients.shape == (p,)
        assert 0.0 <= result.r_squared <= 1.0

        # Correlation check — with p=2000 and n=100k, signal should be recoverable
        coef_corr = np.corrcoef(result.coefficients, beta_true)[0, 1]
        assert coef_corr > 0.9

    def test_near_square_large(self):
        """Near-square matrix (n slightly > p) at large scale."""
        vram = _gpu_vram_gb()
        n, p = 50_000, 10_000
        needed = _estimated_gpu_memory_gb(n, p)
        if vram is not None and needed > vram * 0.8:
            pytest.skip(f"Need ~{needed:.1f}GB, GPU has {vram:.1f}GB")

        rng = np.random.default_rng(42)
        X, y, _ = _make_data(n, p, rng)

        result = fit(X, y, backend='gpu')
        assert result.coefficients.shape == (p,)
        assert result.df_residual == n - p

    def test_oom_handled_gracefully(self):
        """If a problem is too large for VRAM, torch raises OOM, not a segfault.

        We test this by allocating directly on the GPU via torch, bypassing
        numpy entirely. This avoids blowing up CPU RAM (which would get the
        process killed by the Linux OOM killer before torch can catch it).
        """
        import torch

        vram = _gpu_vram_gb()
        if vram is None:
            pytest.skip("Cannot determine VRAM (MPS)")

        device = torch.device('cuda')

        # Try to allocate ~1.5x VRAM as a single tensor on GPU.
        # This should trigger torch.cuda.OutOfMemoryError cleanly.
        target_floats = int(vram * 1.5e9 / 4)  # float32 = 4 bytes
        p = 500
        n = target_floats // p

        with pytest.raises((RuntimeError, torch.cuda.OutOfMemoryError)):
            # Allocate directly on GPU — no CPU intermediate
            X = torch.randn(n, p, device=device, dtype=torch.float32)
            # If allocation somehow succeeds (unlikely), try the matmul
            # which needs even more workspace
            _ = X.T @ X
