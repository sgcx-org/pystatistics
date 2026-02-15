"""
GPU tests for discrete-time survival model.

discrete_time() is the only GPU-accelerated survival method.
It delegates to regression.fit(family='binomial', backend='gpu').
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.survival import discrete_time

# Check GPU availability
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False


# ── Fixtures ─────────────────────────────────────────────────────────

DT_TIME = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=np.float64)
DT_EVENT = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1], dtype=np.float64)
DT_X = np.column_stack([
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1.2, 0.5, -0.3, 0.8, -1.0, 0.3, -0.5, 1.1, -0.8, 0.2],
]).astype(np.float64)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA not available")
class TestDiscreteTimeGPU:
    """GPU-accelerated discrete-time survival tests."""

    def test_gpu_basic_fit(self):
        """GPU backend produces valid results."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X, backend="gpu")
        assert result.n_observations == 10
        assert len(result.coefficients) == 2
        assert np.all(np.isfinite(result.coefficients))

    def test_gpu_matches_cpu(self):
        """GPU results match CPU within tolerance."""
        result_cpu = discrete_time(DT_TIME, DT_EVENT, DT_X, backend="cpu")
        result_gpu = discrete_time(DT_TIME, DT_EVENT, DT_X, backend="gpu")

        assert_allclose(result_gpu.coefficients, result_cpu.coefficients,
                       rtol=1e-4)
        assert_allclose(result_gpu.standard_errors, result_cpu.standard_errors,
                       rtol=1e-3)
        assert_allclose(result_gpu.hazard_ratios, result_cpu.hazard_ratios,
                       rtol=1e-4)
        assert_allclose(result_gpu.baseline_hazard, result_cpu.baseline_hazard,
                       rtol=1e-3)

    def test_gpu_large_dataset(self):
        """GPU handles larger datasets."""
        rng = np.random.default_rng(42)
        n = 500
        time = rng.exponential(5, n)
        event = rng.binomial(1, 0.5, n).astype(np.float64)
        X = rng.standard_normal((n, 3))

        result = discrete_time(time, event, X, backend="gpu")
        assert result.n_observations == 500
        assert len(result.coefficients) == 3
        assert np.all(np.isfinite(result.coefficients))

    def test_gpu_hazard_ratios(self):
        """GPU hazard ratios = exp(coef)."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X, backend="gpu")
        assert_allclose(result.hazard_ratios, np.exp(result.coefficients),
                       rtol=1e-10)

    def test_gpu_custom_intervals(self):
        """GPU with custom intervals."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X,
                              intervals=[1, 3, 5], backend="gpu")
        assert result.n_intervals == 3

    def test_gpu_single_covariate(self):
        """GPU with single covariate."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X[:, :1], backend="gpu")
        assert len(result.coefficients) == 1

    def test_gpu_glm_diagnostics(self):
        """GPU reports GLM diagnostics."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X, backend="gpu")
        assert result.glm_deviance > 0
        assert result.glm_aic > 0
