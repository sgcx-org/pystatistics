"""
GPU backend tests for descriptive statistics.

Validates GPU results against CPU reference backend.
Skipped if no GPU is available.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    HAS_GPU = torch.cuda.is_available() or (
        hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    )
except ImportError:
    HAS_GPU = False

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="No GPU available")

from pystatistics.descriptive import describe, cor, cov, var, quantile, summary


class TestGPUvsCPU:
    """Compare GPU results against CPU reference for all statistics."""

    @pytest.fixture
    def data_2d(self):
        """Well-conditioned 100x5 data."""
        rng = np.random.default_rng(42)
        return rng.standard_normal((100, 5))

    @pytest.fixture
    def data_1d(self):
        """1D data."""
        rng = np.random.default_rng(42)
        return rng.standard_normal(100)

    def test_describe_mean(self, data_2d):
        """GPU mean matches CPU to rtol <= 1e-5."""
        cpu = describe(data_2d, backend='cpu')
        gpu = describe(data_2d, backend='gpu')
        np.testing.assert_allclose(gpu.mean, cpu.mean, rtol=1e-5)

    def test_describe_variance(self, data_2d):
        """GPU variance matches CPU to rtol <= 1e-4."""
        cpu = describe(data_2d, backend='cpu')
        gpu = describe(data_2d, backend='gpu')
        np.testing.assert_allclose(gpu.variance, cpu.variance, rtol=1e-4)

    def test_describe_sd(self, data_2d):
        """GPU SD matches CPU to rtol <= 1e-4."""
        cpu = describe(data_2d, backend='cpu')
        gpu = describe(data_2d, backend='gpu')
        np.testing.assert_allclose(gpu.sd, cpu.sd, rtol=1e-4)

    def test_describe_skewness(self, data_2d):
        """GPU skewness matches CPU to rtol <= 1e-4."""
        cpu = describe(data_2d, backend='cpu')
        gpu = describe(data_2d, backend='gpu')
        np.testing.assert_allclose(gpu.skewness, cpu.skewness, rtol=1e-4)

    def test_describe_kurtosis(self, data_2d):
        """GPU kurtosis matches CPU to rtol <= 1e-4."""
        cpu = describe(data_2d, backend='cpu')
        gpu = describe(data_2d, backend='gpu')
        np.testing.assert_allclose(gpu.kurtosis, cpu.kurtosis, rtol=1e-4)

    def test_cov(self, data_2d):
        """GPU covariance matches CPU to rtol <= 1e-4."""
        cpu = cov(data_2d, backend='cpu')
        gpu = cov(data_2d, backend='gpu')
        np.testing.assert_allclose(
            gpu.covariance_matrix, cpu.covariance_matrix, rtol=1e-4
        )

    def test_cor_pearson(self, data_2d):
        """GPU Pearson correlation matches CPU to rtol <= 1e-4."""
        cpu = cor(data_2d, method='pearson', backend='cpu')
        gpu = cor(data_2d, method='pearson', backend='gpu')
        np.testing.assert_allclose(
            gpu.correlation_pearson, cpu.correlation_pearson, rtol=1e-4
        )

    def test_cor_spearman(self, data_2d):
        """GPU Spearman (CPU fallback) matches CPU exactly."""
        cpu = cor(data_2d, method='spearman', backend='cpu')
        gpu = cor(data_2d, method='spearman', backend='gpu')
        np.testing.assert_allclose(
            gpu.correlation_spearman, cpu.correlation_spearman, rtol=1e-10
        )

    def test_cor_kendall(self, data_2d):
        """GPU Kendall (CPU fallback) matches CPU exactly."""
        cpu = cor(data_2d, method='kendall', backend='cpu')
        gpu = cor(data_2d, method='kendall', backend='gpu')
        np.testing.assert_allclose(
            gpu.correlation_kendall, cpu.correlation_kendall, rtol=1e-10
        )

    def test_quantiles(self, data_2d):
        """GPU quantiles (CPU fallback) match CPU exactly."""
        cpu = quantile(data_2d, type=7, backend='cpu')
        gpu = quantile(data_2d, type=7, backend='gpu')
        np.testing.assert_allclose(gpu.quantiles, cpu.quantiles, rtol=1e-10)

    def test_summary(self, data_2d):
        """GPU summary matches CPU."""
        cpu = summary(data_2d, backend='cpu')
        gpu = summary(data_2d, backend='gpu')
        np.testing.assert_allclose(gpu.summary_table, cpu.summary_table, rtol=1e-5)

    def test_var(self, data_2d):
        """GPU var() matches CPU."""
        cpu = var(data_2d, backend='cpu')
        gpu = var(data_2d, backend='gpu')
        np.testing.assert_allclose(gpu.variance, cpu.variance, rtol=1e-4)

    def test_1d_input(self, data_1d):
        """GPU handles 1D input."""
        cpu = describe(data_1d, backend='cpu')
        gpu = describe(data_1d, backend='gpu')
        np.testing.assert_allclose(gpu.mean, cpu.mean, rtol=1e-5)
        np.testing.assert_allclose(gpu.variance, cpu.variance, rtol=1e-4)

    def test_backend_name(self, data_2d):
        """GPU backend reports correct name."""
        result = describe(data_2d, backend='gpu')
        assert 'gpu' in result.backend_name

    def test_all_quantile_types(self, data_2d):
        """All 9 quantile types match CPU exactly."""
        for qtype in range(1, 10):
            cpu = quantile(data_2d, type=qtype, backend='cpu')
            gpu = quantile(data_2d, type=qtype, backend='gpu')
            np.testing.assert_allclose(
                gpu.quantiles, cpu.quantiles, rtol=1e-10,
                err_msg=f"Type {qtype} mismatch"
            )
