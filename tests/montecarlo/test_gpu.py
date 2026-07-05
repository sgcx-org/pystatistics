"""
Tests for the GPU backends for bootstrap and permutation test.

The GPU vectorizes ONE closed-form statistic each — the sample mean (bootstrap)
and the difference in means (permutation). The statistic form is NEVER inferred:
the caller declares it via ``gpu_statistic`` (``'mean'`` / ``'mean_diff'``).

Fail-loud contract (the reason this module was rewritten at 4.6.7):
- ``backend='gpu'`` without the declaration raises (Guarantee 2 — an explicit
  GPU request that cannot be honoured raises, never silently runs a different
  backend or a different statistic).
- ``backend='gpu'`` on a non-vectorizable configuration raises.
- A declared statistic that does not match the observed value on the original
  data raises — it is never silently replaced by the mean.

GPU-executing tests are skipped if no GPU (CUDA or MPS) is available; the
fail-loud/dispatch tests are hardware-independent (the raise happens before any
device work) and always run.
"""

import numpy as np
import pytest

from pystatistics.montecarlo import boot, permutation_test
from pystatistics.core.exceptions import ValidationError


def mean_stat(data, indices):
    """Bootstrap statistic: sample mean."""
    return np.array([np.mean(data[indices])])


def mean_var_stat(data, indices):
    """Bootstrap statistic: mean and variance (not a GPU-supported form)."""
    d = data[indices]
    return np.array([np.mean(d), np.var(d, ddof=1)])


def median_stat(data, indices):
    """Bootstrap statistic: median (not the mean)."""
    return np.array([np.median(data[indices])])


def mean_diff(x, y):
    """Permutation statistic: difference in means."""
    return np.mean(x) - np.mean(y)


def median_diff(x, y):
    """Permutation statistic: difference in medians (not the mean-difference)."""
    return np.median(x) - np.median(y)


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
# Fail-loud dispatch tests (hardware-independent — raise before device work)
# ---------------------------------------------------------------------------

class TestBootstrapFailLoud:
    """backend='gpu' must be honoured on the GPU or raise — never fall back."""

    def test_gpu_without_declaration_raises(self):
        data = np.arange(1.0, 51.0)
        with pytest.raises(ValidationError, match="requires gpu_statistic='mean'"):
            boot(data, mean_stat, n_resamples=99, seed=1, backend='gpu')

    def test_gpu_balanced_raises(self):
        data = np.arange(1.0, 51.0)
        with pytest.raises(ValidationError, match="cannot run on the GPU"):
            boot(data, mean_stat, n_resamples=99, seed=1, backend='gpu',
                 gpu_statistic='mean', method='balanced')

    def test_gpu_parametric_raises(self):
        data = np.arange(1.0, 51.0)
        with pytest.raises(ValidationError, match="cannot run on the GPU"):
            boot(data, mean_stat, n_resamples=99, seed=1, backend='gpu',
                 gpu_statistic='mean', method='parametric',
                 ran_gen=lambda d, m, rng: d)

    def test_gpu_strata_raises(self):
        data = np.arange(1.0, 51.0)
        strata = np.repeat([0, 1], 25)
        with pytest.raises(ValidationError, match="cannot run on the GPU"):
            boot(data, mean_stat, n_resamples=99, seed=1, backend='gpu',
                 gpu_statistic='mean', strata=strata)

    def test_gpu_2d_data_raises(self):
        data = np.arange(1.0, 51.0).reshape(25, 2)
        with pytest.raises(ValidationError, match="cannot run on the GPU"):
            boot(data, mean_stat, n_resamples=99, seed=1, backend='gpu',
                 gpu_statistic='mean')

    def test_invalid_gpu_statistic_raises(self):
        data = np.arange(1.0, 51.0)
        with pytest.raises(ValidationError, match="gpu_statistic must be 'mean'"):
            boot(data, mean_stat, n_resamples=99, seed=1, gpu_statistic='median')

    def test_gpu_fp64_rejected(self):
        data = np.arange(1.0, 51.0)
        with pytest.raises(ValidationError):
            boot(data, mean_stat, n_resamples=99, seed=1, backend='gpu_fp64',
                 gpu_statistic='mean')


class TestPermutationFailLoud:
    def test_gpu_without_declaration_raises(self):
        x, y = np.arange(1.0, 11.0), np.arange(11.0, 21.0)
        with pytest.raises(ValidationError,
                           match="requires gpu_statistic='mean_diff'"):
            permutation_test(x, y, mean_diff, n_resamples=99, seed=1,
                             backend='gpu')

    def test_gpu_2d_groups_raise(self):
        x = np.arange(1.0, 11.0).reshape(5, 2)
        y = np.arange(11.0, 21.0).reshape(5, 2)
        with pytest.raises(ValidationError, match="cannot run on the GPU"):
            permutation_test(x, y, mean_diff, n_resamples=99, seed=1,
                             backend='gpu', gpu_statistic='mean_diff')

    def test_invalid_gpu_statistic_raises(self):
        x, y = np.arange(1.0, 11.0), np.arange(11.0, 21.0)
        with pytest.raises(ValidationError,
                           match="gpu_statistic must be 'mean_diff'"):
            permutation_test(x, y, mean_diff, n_resamples=99, seed=1,
                             gpu_statistic='mean')


# ---------------------------------------------------------------------------
# Declaration guard — a declared statistic that isn't the mean must RAISE,
# never be silently computed as the mean (the R16 showstopper this closes).
# ---------------------------------------------------------------------------

class TestDeclarationGuard:
    def test_boot_declared_mean_but_median_raises(self, gpu_available):
        data = np.random.default_rng(0).gamma(2, 3, 300)  # skewed: median != mean
        with pytest.raises(ValidationError,
                           match="does not equal mean"):
            boot(data, median_stat, n_resamples=99, seed=1, backend='gpu',
                 gpu_statistic='mean')

    def test_boot_declared_mean_but_multivalued_raises(self, gpu_available):
        data = np.arange(1.0, 51.0)
        with pytest.raises(ValidationError, match="does not equal mean"):
            boot(data, mean_var_stat, n_resamples=99, seed=1, backend='gpu',
                 gpu_statistic='mean')

    def test_perm_declared_meandiff_but_mediandiff_raises(self, gpu_available):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 60)
        y = rng.gamma(2, 1, 60)  # skewed -> mean-diff != median-diff
        with pytest.raises(ValidationError, match="does not equal mean"):
            permutation_test(x, y, median_diff, n_resamples=99, seed=1,
                             backend='gpu', gpu_statistic='mean_diff')


# ---------------------------------------------------------------------------
# GPU-executing tests (correctness + reproducibility on real hardware)
# ---------------------------------------------------------------------------

class TestGPUBootstrap:
    def test_gpu_bootstrap_mean(self, gpu_available):
        data = np.arange(1.0, 51.0)
        result = boot(data, mean_stat, n_resamples=500, seed=42,
                      backend='gpu', gpu_statistic='mean')
        assert 'gpu' in result.backend_name
        assert result.t0[0] == pytest.approx(25.5, rel=1e-10)
        assert result.t.shape == (500, 1)
        assert 1.0 < result.se[0] < 5.0

    def test_gpu_bootstrap_matches_cpu_statistically(self, gpu_available):
        """Independent RNGs -> t0 identical, SE within Monte-Carlo error."""
        data = np.arange(1.0, 51.0)
        cpu = boot(data, mean_stat, n_resamples=4000, seed=42, backend='cpu')
        gpu = boot(data, mean_stat, n_resamples=4000, seed=42, backend='gpu',
                   gpu_statistic='mean')
        np.testing.assert_array_equal(gpu.t0, cpu.t0)
        np.testing.assert_allclose(gpu.se, cpu.se, rtol=0.15)

    def test_gpu_bootstrap_seed_reproducibility(self, gpu_available):
        data = np.arange(1.0, 31.0)
        r1 = boot(data, mean_stat, n_resamples=100, seed=123, backend='gpu',
                  gpu_statistic='mean')
        r2 = boot(data, mean_stat, n_resamples=100, seed=123, backend='gpu',
                  gpu_statistic='mean')
        np.testing.assert_array_equal(r1.t, r2.t)


class TestGPUPermutation:
    def test_gpu_permutation_significant(self, gpu_available):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 30)
        y = rng.normal(3, 1, 30)
        result = permutation_test(x, y, mean_diff, n_resamples=999, seed=42,
                                  backend='gpu', gpu_statistic='mean_diff')
        assert 'gpu' in result.backend_name
        assert result.p_value < 0.05

    def test_gpu_permutation_alternatives(self, gpu_available):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 20)
        y = rng.normal(2, 1, 20)
        for alt in ["two-sided", "less", "greater"]:
            result = permutation_test(x, y, mean_diff, n_resamples=500,
                                      alternative=alt, seed=42, backend='gpu',
                                      gpu_statistic='mean_diff')
            assert 0 <= result.p_value <= 1
            assert result.alternative == alt

    def test_gpu_permutation_seed_reproducibility(self, gpu_available):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 15)
        y = rng.normal(1, 1, 15)
        r1 = permutation_test(x, y, mean_diff, n_resamples=200, seed=99,
                              backend='gpu', gpu_statistic='mean_diff')
        r2 = permutation_test(x, y, mean_diff, n_resamples=200, seed=99,
                              backend='gpu', gpu_statistic='mean_diff')
        np.testing.assert_array_equal(r1.perm_stats, r2.perm_stats)
        assert r1.p_value == r2.p_value


# ---------------------------------------------------------------------------
# backend='auto' — no preference, so a CPU fallback is legitimate (disclosed),
# and it must NEVER silently compute the wrong statistic.
# ---------------------------------------------------------------------------

class TestAutoBackend:
    def test_auto_arbitrary_statistic_uses_cpu(self):
        data = np.arange(1.0, 11.0)
        result = boot(data, median_stat, n_resamples=50, seed=42, backend='auto')
        assert 'cpu' in result.backend_name

    def test_auto_declared_mean_runs_and_is_correct(self):
        """auto with a declared mean: GPU on CUDA, CPU elsewhere — either way
        the answer is the mean (never silently a different statistic)."""
        data = np.arange(1.0, 51.0)
        result = boot(data, mean_stat, n_resamples=200, seed=42, backend='auto',
                      gpu_statistic='mean')
        assert result.t0[0] == pytest.approx(25.5, rel=1e-10)
        assert 'bootstrap' in result.backend_name
