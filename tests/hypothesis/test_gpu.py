"""
Tests for GPU Monte Carlo backend for hypothesis tests.

GPU only accelerates Monte Carlo simulation for chi-squared and
Fisher r×c tests. All other tests fall back to CPU automatically.

Skipped if no GPU (CUDA or MPS) is available.
"""

import numpy as np
import pytest

# Try importing torch to check for GPU availability
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    HAS_MPS = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    HAS_GPU = HAS_CUDA or HAS_MPS
except ImportError:
    HAS_GPU = False

pytestmark = pytest.mark.skipif(
    not HAS_GPU,
    reason="No GPU available (need CUDA or MPS)"
)


class TestGPUChisqMonteCarlo:
    """GPU Monte Carlo for chi-squared test."""

    def test_independence_mc_vs_cpu(self):
        """GPU and CPU Monte Carlo give similar p-values."""
        from pystatistics.hypothesis import chisq_test

        table = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

        cpu_result = chisq_test(table, simulate_p_value=True, B=5000,
                                backend='cpu')
        gpu_result = chisq_test(table, simulate_p_value=True, B=5000,
                                backend='gpu')

        # Monte Carlo p-values should be close (stochastic)
        assert gpu_result.p_value == pytest.approx(
            cpu_result.p_value, abs=0.05,
        ), "GPU vs CPU Monte Carlo p-values differ too much"

        # Statistic should be identical (computed deterministically)
        assert gpu_result.statistic == pytest.approx(
            cpu_result.statistic, rel=1e-4,
        )

    def test_gof_mc_vs_cpu(self):
        """GPU and CPU Monte Carlo GOF give similar p-values."""
        from pystatistics.hypothesis import chisq_test

        observed = np.array([16, 18, 16, 14, 12, 12])
        p = np.ones(6) / 6

        cpu_result = chisq_test(observed, p=p, simulate_p_value=True,
                                B=5000, backend='cpu')
        gpu_result = chisq_test(observed, p=p, simulate_p_value=True,
                                B=5000, backend='gpu')

        assert gpu_result.p_value == pytest.approx(
            cpu_result.p_value, abs=0.05,
        )
        assert gpu_result.statistic == pytest.approx(
            cpu_result.statistic, rel=1e-4,
        )

    def test_independence_mc_backend_name(self):
        """GPU backend name is reported."""
        from pystatistics.hypothesis import chisq_test

        table = np.array([[10, 20], [30, 40]])
        result = chisq_test(table, simulate_p_value=True, B=1000,
                            backend='gpu')
        assert "gpu" in result.backend_name

    def test_non_mc_falls_back_to_cpu(self):
        """Non-Monte Carlo chi-squared falls back to CPU."""
        from pystatistics.hypothesis import chisq_test

        table = np.array([[10, 20], [30, 40]])
        result = chisq_test(table, backend='gpu')
        assert "cpu_fallback" in result.backend_name


class TestGPUFisherMonteCarlo:
    """GPU Monte Carlo for Fisher's exact test."""

    def test_fisher_rxc_mc_vs_cpu(self):
        """GPU and CPU Fisher r×c give similar p-values."""
        from pystatistics.hypothesis import fisher_test

        table = np.array([
            [5, 10, 15],
            [10, 5, 20],
            [15, 20, 5],
        ])

        cpu_result = fisher_test(table, simulate_p_value=True, B=5000,
                                 backend='cpu')
        gpu_result = fisher_test(table, simulate_p_value=True, B=5000,
                                 backend='gpu')

        assert gpu_result.p_value == pytest.approx(
            cpu_result.p_value, abs=0.05,
        )

    def test_fisher_2x2_falls_back_to_cpu(self):
        """Fisher 2x2 (exact, no MC) falls back to CPU."""
        from pystatistics.hypothesis import fisher_test

        table = np.array([[1, 9], [11, 3]])
        result = fisher_test(table, backend='gpu')
        assert "cpu_fallback" in result.backend_name


class TestGPUFallback:
    """Tests that non-MC tests correctly fall back to CPU."""

    def test_t_test_fallback(self):
        """t-test falls back to CPU on GPU backend."""
        from pystatistics.hypothesis import t_test

        result = t_test([1, 2, 3, 4, 5], mu=3, backend='gpu')
        assert "cpu_fallback" in result.backend_name
        assert result.statistic is not None

    def test_wilcox_fallback(self):
        """Wilcoxon falls back to CPU."""
        from pystatistics.hypothesis import wilcox_test

        result = wilcox_test([1, 2, 3, 4, 5], mu=3, backend='gpu')
        assert "cpu_fallback" in result.backend_name

    def test_var_test_fallback(self):
        """var.test falls back to CPU."""
        from pystatistics.hypothesis import var_test

        result = var_test([1, 2, 3, 4, 5], [6, 7, 8, 9, 10],
                          backend='gpu')
        assert "cpu_fallback" in result.backend_name
