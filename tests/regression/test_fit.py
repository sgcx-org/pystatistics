"""
Tests for regression fit().

Tests the complete pipeline: Design construction, backend selection,
and solution properties.
"""

import pytest
import numpy as np

from pystatistics.regression import fit, Design
from pystatistics.regression.solution import LinearSolution
from pystatistics.core.exceptions import NumericalError


class TestFitBasic:
    """Basic fit() functionality tests."""

    def test_fit_from_arrays(self, simple_regression_data):
        X, y, _ = simple_regression_data
        result = fit(X, y)
        assert isinstance(result, LinearSolution)
        assert result.coefficients.shape == (3,)

    def test_fit_from_design(self, simple_regression_data):
        X, y, _ = simple_regression_data
        design = Design.from_arrays(X, y)
        result = fit(design)
        assert isinstance(result, LinearSolution)

    def test_fit_requires_y_with_arrays(self, simple_regression_data):
        X, _, _ = simple_regression_data
        with pytest.raises(ValueError, match="y required"):
            fit(X)

    def test_coefficients_close_to_truth(self, simple_regression_data):
        X, y, beta_true = simple_regression_data
        result = fit(X, y)
        # With low noise (sigma=0.1), coefficients should be close to truth
        np.testing.assert_allclose(result.coefficients, beta_true, atol=0.5)

    def test_r_squared_range(self, simple_regression_data):
        X, y, _ = simple_regression_data
        result = fit(X, y)
        assert 0.0 <= result.r_squared <= 1.0

    def test_residuals_sum_to_near_zero_with_intercept(self, rng):
        """For models with intercept, residuals should sum to ~0."""
        n = 100
        X = np.column_stack([
            np.ones(n),
            rng.standard_normal(n),
            rng.standard_normal(n),
        ])
        y = X @ [1.0, 2.0, -0.5] + rng.standard_normal(n) * 0.1
        result = fit(X, y)
        assert abs(result.residuals.sum()) < 1e-10


class TestFitProperties:
    """Test derived properties of LinearSolution."""

    def test_standard_errors_positive(self, simple_regression_data):
        X, y, _ = simple_regression_data
        result = fit(X, y)
        se = result.standard_errors
        assert np.all(se > 0)
        assert np.all(np.isfinite(se))

    def test_t_statistics_finite(self, simple_regression_data):
        X, y, _ = simple_regression_data
        result = fit(X, y)
        assert np.all(np.isfinite(result.t_statistics))

    def test_p_values_in_zero_one(self, simple_regression_data):
        X, y, _ = simple_regression_data
        result = fit(X, y)
        pv = result.p_values
        assert np.all(pv >= 0.0)
        assert np.all(pv <= 1.0)

    def test_fitted_plus_residuals_equals_y(self, simple_regression_data):
        X, y, _ = simple_regression_data
        result = fit(X, y)
        np.testing.assert_allclose(
            result.fitted_values + result.residuals, y, atol=1e-12
        )

    def test_rss_matches_residuals(self, simple_regression_data):
        X, y, _ = simple_regression_data
        result = fit(X, y)
        expected_rss = float(result.residuals @ result.residuals)
        assert abs(result.rss - expected_rss) < 1e-12

    def test_r_squared_formula(self, simple_regression_data):
        X, y, _ = simple_regression_data
        result = fit(X, y)
        expected = 1.0 - result.rss / result.tss
        assert abs(result.r_squared - expected) < 1e-15

    def test_summary_runs(self, simple_regression_data):
        X, y, _ = simple_regression_data
        result = fit(X, y)
        s = result.summary()
        assert "R-squared" in s
        assert "Pr(>|t|)" in s
        assert "Backend" in s


class TestFitRankDeficient:
    """Test behavior with rank-deficient data."""

    def test_collinear_rank_detection(self, collinear_data):
        X, y = collinear_data
        result = fit(X, y)
        assert result.rank < X.shape[1]

    def test_collinear_has_nan_se(self, collinear_data):
        X, y = collinear_data
        result = fit(X, y)
        # At least one SE should be NaN (aliased coefficient)
        assert np.any(np.isnan(result.standard_errors))

    def test_collinear_has_nan_t(self, collinear_data):
        X, y = collinear_data
        result = fit(X, y)
        assert np.any(np.isnan(result.t_statistics))

    def test_collinear_has_nan_pv(self, collinear_data):
        X, y = collinear_data
        result = fit(X, y)
        assert np.any(np.isnan(result.p_values))


class TestBackendSelection:
    """Test backend dispatch logic."""

    def test_cpu_backend(self, simple_regression_data):
        X, y, _ = simple_regression_data
        result = fit(X, y, backend='cpu')
        assert result.backend_name == 'cpu_qr'

    def test_cpu_qr_backend(self, simple_regression_data):
        X, y, _ = simple_regression_data
        result = fit(X, y, backend='cpu_qr')
        assert result.backend_name == 'cpu_qr'

    def test_auto_backend_works(self, simple_regression_data):
        X, y, _ = simple_regression_data
        result = fit(X, y, backend='auto')
        # auto should succeed regardless of GPU availability
        assert result.backend_name in ('cpu_qr', 'gpu_qr_fp32', 'gpu_qr_fp64')

    def test_invalid_backend_raises(self, simple_regression_data):
        X, y, _ = simple_regression_data
        with pytest.raises(ValueError, match="Unknown backend"):
            fit(X, y, backend='nonsense')

    def test_cpu_svd_not_implemented(self, simple_regression_data):
        X, y, _ = simple_regression_data
        with pytest.raises(NotImplementedError):
            fit(X, y, backend='cpu_svd')


def _gpu_available():
    """Check if any GPU is available for testing."""
    try:
        import torch
        return (torch.cuda.is_available() or
                (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()))
    except ImportError:
        return False


@pytest.mark.skipif(not _gpu_available(), reason="No GPU available")
class TestGPUBackend:
    """GPU-specific tests."""

    def test_gpu_backend_runs(self, simple_regression_data):
        X, y, _ = simple_regression_data
        result = fit(X, y, backend='gpu')
        assert 'gpu' in result.backend_name

    def test_gpu_matches_cpu(self, simple_regression_data):
        """GPU coefficients should be close to CPU reference."""
        X, y, _ = simple_regression_data
        cpu_result = fit(X, y, backend='cpu')
        gpu_result = fit(X, y, backend='gpu')
        np.testing.assert_allclose(
            gpu_result.coefficients,
            cpu_result.coefficients,
            rtol=1e-4, atol=1e-5,
        )

    def test_gpu_r_squared_matches_cpu(self, simple_regression_data):
        X, y, _ = simple_regression_data
        cpu_result = fit(X, y, backend='cpu')
        gpu_result = fit(X, y, backend='gpu')
        np.testing.assert_allclose(
            gpu_result.r_squared,
            cpu_result.r_squared,
            rtol=1e-4,
        )

    def test_gpu_refuses_ill_conditioned(self, rng):
        """GPU should refuse ill-conditioned matrices without force=True."""
        n, p = 100, 5
        # Create ill-conditioned X (cond > 1e6)
        U, _ = np.linalg.qr(rng.standard_normal((n, p)))
        V, _ = np.linalg.qr(rng.standard_normal((p, p)))
        sv = np.array([1e7, 1e4, 1e2, 1e1, 1.0])
        X = U @ np.diag(sv) @ V.T
        y = rng.standard_normal(n)

        with pytest.raises(NumericalError, match="ill-conditioned"):
            fit(X, y, backend='gpu')

    def test_gpu_proceeds_with_force(self, rng):
        """GPU should proceed on ill-conditioned if force=True."""
        n, p = 100, 5
        U, _ = np.linalg.qr(rng.standard_normal((n, p)))
        V, _ = np.linalg.qr(rng.standard_normal((p, p)))
        sv = np.array([1e7, 1e4, 1e2, 1e1, 1.0])
        X = U @ np.diag(sv) @ V.T
        y = rng.standard_normal(n)

        result = fit(X, y, backend='gpu', force=True)
        assert isinstance(result, LinearSolution)
        assert 'condition_number' in result.info

    def test_gpu_timing_populated(self, simple_regression_data):
        X, y, _ = simple_regression_data
        result = fit(X, y, backend='gpu')
        assert result.timing is not None
        assert 'total_seconds' in result.timing
        assert 'condition_check' in result.timing
