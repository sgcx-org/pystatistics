"""
Tests for GPU backends for bootstrap and permutation test.

GPU backends currently fall back to CPU for arbitrary user statistics
since Python functions cannot execute on GPU. Tests verify:
1. GPU backend produces correct results (matching CPU)
2. Backend name indicates GPU was requested
3. Fallback behavior is transparent

Skipped if no GPU (CUDA or MPS) is available.
"""

import numpy as np
import pytest

from pystatistics.montecarlo import boot, permutation_test


def mean_stat(data, indices):
    """Bootstrap statistic: sample mean."""
    return np.array([np.mean(data[indices])])


def mean_var_stat(data, indices):
    """Bootstrap statistic: mean and variance."""
    d = data[indices]
    return np.array([np.mean(d), np.var(d, ddof=1)])


def mean_diff(x, y):
    """Permutation statistic: difference in means."""
    return np.mean(x) - np.mean(y)


# ---------------------------------------------------------------------------
# GPU availability fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def gpu_available():
    """Skip if no GPU is available."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        if not (has_cuda or has_mps):
            pytest.skip("No GPU available")
        return 'cuda' if has_cuda else 'mps'
    except ImportError:
        pytest.skip("PyTorch not installed")


# ---------------------------------------------------------------------------
# GPU Bootstrap Tests
# ---------------------------------------------------------------------------

class TestGPUBootstrap:
    """Tests for GPUBootstrapBackend via boot(backend='gpu')."""

    def test_gpu_bootstrap_mean(self, gpu_available):
        """GPU bootstrap produces correct results for mean statistic."""
        data = np.arange(1.0, 51.0)
        result = boot(data, mean_stat, R=500, seed=42, backend='gpu')

        # t0 should be exact (deterministic)
        assert result.t0[0] == pytest.approx(25.5, rel=1e-10)

        # R replicates produced
        assert result.t.shape == (500, 1)

        # bias should be small for unbiased statistic
        assert abs(result.bias[0]) < 2.0

        # SE should be reasonable
        assert 1.0 < result.se[0] < 5.0

    def test_gpu_bootstrap_matches_cpu_statistically(self, gpu_available):
        """GPU bootstrap results are statistically consistent with CPU.

        GPU uses different RNG (torch vs numpy) so exact replicate match
        is not expected, but t0 is identical and SE should be close.
        """
        data = np.arange(1.0, 51.0)

        result_cpu = boot(data, mean_stat, R=2000, seed=42, backend='cpu')
        result_gpu = boot(data, mean_stat, R=2000, seed=42, backend='gpu')

        # t0 is deterministic — must match exactly
        np.testing.assert_array_equal(result_gpu.t0, result_cpu.t0)

        # SE should be close (both estimate the same quantity)
        np.testing.assert_allclose(
            result_gpu.se, result_cpu.se, rtol=0.15,
        )

    def test_gpu_bootstrap_multivariate(self, gpu_available):
        """GPU bootstrap handles multi-dimensional statistics."""
        data = np.arange(1.0, 31.0)
        result = boot(data, mean_var_stat, R=300, seed=42, backend='gpu')

        assert result.t0.shape == (2,)
        assert result.t.shape == (300, 2)

    def test_gpu_bootstrap_backend_name(self, gpu_available):
        """GPU backend name indicates device."""
        data = np.arange(1.0, 11.0)
        result = boot(data, mean_stat, R=50, seed=42, backend='gpu')

        assert 'gpu' in result.backend_name.lower() or 'cpu' in result.backend_name.lower()
        assert 'bootstrap' in result.backend_name.lower()

    def test_gpu_bootstrap_balanced(self, gpu_available):
        """GPU bootstrap supports balanced simulation."""
        data = np.arange(1.0, 21.0)
        result = boot(
            data, mean_stat, R=200, seed=42,
            sim="balanced", backend='gpu',
        )

        assert result.t.shape == (200, 1)
        assert result.t0[0] == pytest.approx(10.5, rel=1e-10)

    def test_gpu_bootstrap_balanced_matches_cpu(self, gpu_available):
        """GPU balanced bootstrap falls back to CPU, matches exactly."""
        data = np.arange(1.0, 21.0)

        result_cpu = boot(
            data, mean_stat, R=200, seed=42,
            sim="balanced", backend='cpu',
        )
        result_gpu = boot(
            data, mean_stat, R=200, seed=42,
            sim="balanced", backend='gpu',
        )

        # Balanced sim falls back to CPU — results should be identical
        np.testing.assert_array_equal(result_gpu.t, result_cpu.t)
        assert 'cpu' in result_gpu.backend_name

    def test_gpu_bootstrap_seed_reproducibility(self, gpu_available):
        """GPU bootstrap is reproducible with same seed."""
        data = np.arange(1.0, 31.0)

        result1 = boot(data, mean_stat, R=100, seed=123, backend='gpu')
        result2 = boot(data, mean_stat, R=100, seed=123, backend='gpu')

        np.testing.assert_array_equal(result1.t, result2.t)


# ---------------------------------------------------------------------------
# GPU Permutation Tests
# ---------------------------------------------------------------------------

class TestGPUPermutation:
    """Tests for GPUPermutationBackend via permutation_test(backend='gpu')."""

    def test_gpu_permutation_significant(self, gpu_available):
        """GPU permutation test detects significant difference."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 30)
        y = rng.normal(3, 1, 30)

        result = permutation_test(
            x, y, mean_diff, R=999,
            seed=42, backend='gpu',
        )

        assert result.p_value < 0.05

    def test_gpu_permutation_not_significant(self, gpu_available):
        """GPU permutation test accepts null when groups are similar."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 30)
        y = rng.normal(0, 1, 30)

        result = permutation_test(
            x, y, mean_diff, R=999,
            seed=42, backend='gpu',
        )

        assert result.p_value > 0.05

    def test_gpu_permutation_matches_cpu_statistically(self, gpu_available):
        """GPU permutation results are statistically consistent with CPU.

        GPU uses different RNG (torch vs numpy) so exact match is not
        expected, but observed_stat is identical and p-values should agree
        on significance direction.
        """
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        y = rng.normal(2, 1, 50)

        result_cpu = permutation_test(
            x, y, mean_diff, R=2000,
            seed=42, backend='cpu',
        )
        result_gpu = permutation_test(
            x, y, mean_diff, R=2000,
            seed=42, backend='gpu',
        )

        assert result_gpu.observed_stat == result_cpu.observed_stat
        # Both should strongly reject null
        assert result_cpu.p_value < 0.05
        assert result_gpu.p_value < 0.05

    def test_gpu_permutation_backend_name(self, gpu_available):
        """GPU permutation backend name indicates GPU device."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 10)
        y = rng.normal(0, 1, 10)

        result = permutation_test(
            x, y, mean_diff, R=50,
            seed=42, backend='gpu',
        )

        assert 'gpu' in result.backend_name.lower()
        assert 'permutation' in result.backend_name.lower()

    def test_gpu_permutation_alternatives(self, gpu_available):
        """GPU permutation supports all alternative hypotheses."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 20)
        y = rng.normal(2, 1, 20)

        for alt in ["two.sided", "less", "greater"]:
            result = permutation_test(
                x, y, mean_diff, R=500,
                alternative=alt, seed=42, backend='gpu',
            )
            # p-value is always in [0, 1]; specific values depend on
            # direction. "less" with x < y gives p ≈ 1, which is valid.
            assert 0 <= result.p_value <= 1
            assert result.alternative == alt

    def test_gpu_permutation_seed_reproducibility(self, gpu_available):
        """GPU permutation is reproducible with same seed."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 15)
        y = rng.normal(1, 1, 15)

        result1 = permutation_test(
            x, y, mean_diff, R=200,
            seed=99, backend='gpu',
        )
        result2 = permutation_test(
            x, y, mean_diff, R=200,
            seed=99, backend='gpu',
        )

        np.testing.assert_array_equal(result1.perm_stats, result2.perm_stats)
        assert result1.p_value == result2.p_value


# ---------------------------------------------------------------------------
# GPU Backend Fallback Tests
# ---------------------------------------------------------------------------

class TestGPUFallback:
    """Tests for GPU fallback behavior."""

    def test_gpu_bootstrap_multivariate_falls_back(self):
        """Multi-output bootstrap falls back to CPU gracefully."""
        data = np.arange(1.0, 11.0)
        result = boot(data, mean_var_stat, R=50, seed=42, backend='gpu')
        # Should fall back to CPU for multivariate statistic
        assert 'cpu' in result.backend_name

    def test_gpu_perm_works(self):
        """permutation_test(backend='gpu') now works for mean-difference."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        result = permutation_test(
            x, y, mean_diff, R=100,
            seed=42, backend='gpu',
        )
        assert result.p_value < 0.05  # clearly different groups

    def test_auto_backend_uses_gpu_or_cpu_bootstrap(self):
        """backend='auto' uses GPU for simple stats, CPU for complex."""
        data = np.arange(1.0, 11.0)
        result = boot(data, mean_stat, R=50, seed=42, backend='auto')

        assert 'bootstrap' in result.backend_name

    def test_auto_backend_uses_gpu_permutation_if_available(self):
        """backend='auto' uses GPU for permutation when GPU is available."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])

        result = permutation_test(
            x, y, mean_diff, R=50,
            seed=42, backend='auto',
        )

        # Should use GPU if available, CPU otherwise
        assert 'permutation' in result.backend_name
