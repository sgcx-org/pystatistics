"""
GPU tests for GLM (Generalized Linear Models).

Validates GPU IRLS backend against CPU reference for all three families:
Gaussian, Binomial, and Poisson.

GPU uses FP32 for IRLS iterations, so tolerances are wider than CPU vs R.
The key guarantee: GPU produces statistically equivalent results to CPU,
validated by:
    - Coefficient correlation > 0.9999
    - Element-wise rtol ≤ 1e-3 on well-conditioned problems
    - Deviance within rtol=1e-4

Skipped automatically when no GPU is available.
"""

import pytest
import numpy as np

from pystatistics.regression import Design, fit, GLMSolution


def _gpu_available():
    try:
        import torch
        return (torch.cuda.is_available() or
                (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()))
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _gpu_available(), reason="No GPU available"
)


# =====================================================================
# Test data generators
# =====================================================================

def _make_gaussian_data(n=500, p=5, seed=42):
    """Generate Gaussian regression data."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    beta = rng.standard_normal(p) * 2
    y = X @ beta + rng.standard_normal(n)
    return X, y


def _make_binomial_data(n=500, p=4, seed=42):
    """Generate binary classification data."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    beta = rng.standard_normal(p)
    eta = X @ beta
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(np.float64)
    return X, y


def _make_poisson_data(n=500, p=4, seed=42):
    """Generate Poisson count data."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1)) * 0.5])
    beta = np.array([1.0] + list(rng.standard_normal(p - 1) * 0.3))
    eta = X @ beta
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(np.float64)
    return X, y


# =====================================================================
# GPU vs CPU correctness tests
# =====================================================================

class TestGLMGPUGaussian:
    """GPU Gaussian GLM matches CPU reference."""

    def test_coefficients_match(self):
        X, y = _make_gaussian_data()
        design = Design.from_arrays(X, y)

        cpu_result = fit(design, family='gaussian', backend='cpu')
        gpu_result = fit(design, family='gaussian', backend='gpu')

        assert isinstance(gpu_result, GLMSolution)
        np.testing.assert_allclose(
            gpu_result.coefficients, cpu_result.coefficients,
            rtol=1e-3, err_msg="Gaussian GPU coefficients differ from CPU"
        )

    def test_deviance_matches(self):
        X, y = _make_gaussian_data()
        design = Design.from_arrays(X, y)

        cpu_result = fit(design, family='gaussian', backend='cpu')
        gpu_result = fit(design, family='gaussian', backend='gpu')

        np.testing.assert_allclose(
            gpu_result.deviance, cpu_result.deviance,
            rtol=1e-4, err_msg="Gaussian GPU deviance differs from CPU"
        )

    def test_fitted_values_match(self):
        X, y = _make_gaussian_data()
        design = Design.from_arrays(X, y)

        cpu_result = fit(design, family='gaussian', backend='cpu')
        gpu_result = fit(design, family='gaussian', backend='gpu')

        np.testing.assert_allclose(
            gpu_result.fitted_values, cpu_result.fitted_values,
            rtol=1e-3, err_msg="Gaussian GPU fitted values differ from CPU"
        )

    def test_converged(self):
        X, y = _make_gaussian_data()
        result = fit(X, y, family='gaussian', backend='gpu')
        assert result.converged


class TestGLMGPUBinomial:
    """GPU Binomial GLM matches CPU reference."""

    def test_coefficients_match(self):
        X, y = _make_binomial_data()
        design = Design.from_arrays(X, y)

        cpu_result = fit(design, family='binomial', backend='cpu')
        gpu_result = fit(design, family='binomial', backend='gpu')

        assert isinstance(gpu_result, GLMSolution)
        np.testing.assert_allclose(
            gpu_result.coefficients, cpu_result.coefficients,
            rtol=1e-3, err_msg="Binomial GPU coefficients differ from CPU"
        )

    def test_deviance_matches(self):
        X, y = _make_binomial_data()
        design = Design.from_arrays(X, y)

        cpu_result = fit(design, family='binomial', backend='cpu')
        gpu_result = fit(design, family='binomial', backend='gpu')

        np.testing.assert_allclose(
            gpu_result.deviance, cpu_result.deviance,
            rtol=1e-4, err_msg="Binomial GPU deviance differs from CPU"
        )

    def test_fitted_values_match(self):
        X, y = _make_binomial_data()
        design = Design.from_arrays(X, y)

        cpu_result = fit(design, family='binomial', backend='cpu')
        gpu_result = fit(design, family='binomial', backend='gpu')

        np.testing.assert_allclose(
            gpu_result.fitted_values, cpu_result.fitted_values,
            rtol=1e-3, err_msg="Binomial GPU fitted values differ from CPU"
        )

    def test_converged(self):
        X, y = _make_binomial_data()
        result = fit(X, y, family='binomial', backend='gpu')
        assert result.converged

    def test_standard_errors_match(self):
        X, y = _make_binomial_data()
        design = Design.from_arrays(X, y)

        cpu_result = fit(design, family='binomial', backend='cpu')
        gpu_result = fit(design, family='binomial', backend='gpu')

        np.testing.assert_allclose(
            gpu_result.standard_errors, cpu_result.standard_errors,
            rtol=1e-2, err_msg="Binomial GPU SEs differ from CPU"
        )


class TestGLMGPUPoisson:
    """GPU Poisson GLM matches CPU reference."""

    def test_coefficients_match(self):
        X, y = _make_poisson_data()
        design = Design.from_arrays(X, y)

        cpu_result = fit(design, family='poisson', backend='cpu')
        gpu_result = fit(design, family='poisson', backend='gpu')

        assert isinstance(gpu_result, GLMSolution)
        np.testing.assert_allclose(
            gpu_result.coefficients, cpu_result.coefficients,
            rtol=1e-3, err_msg="Poisson GPU coefficients differ from CPU"
        )

    def test_deviance_matches(self):
        X, y = _make_poisson_data()
        design = Design.from_arrays(X, y)

        cpu_result = fit(design, family='poisson', backend='cpu')
        gpu_result = fit(design, family='poisson', backend='gpu')

        np.testing.assert_allclose(
            gpu_result.deviance, cpu_result.deviance,
            rtol=1e-4, err_msg="Poisson GPU deviance differs from CPU"
        )

    def test_fitted_values_match(self):
        X, y = _make_poisson_data()
        design = Design.from_arrays(X, y)

        cpu_result = fit(design, family='poisson', backend='cpu')
        gpu_result = fit(design, family='poisson', backend='gpu')

        np.testing.assert_allclose(
            gpu_result.fitted_values, cpu_result.fitted_values,
            rtol=1e-3, err_msg="Poisson GPU fitted values differ from CPU"
        )

    def test_converged(self):
        X, y = _make_poisson_data()
        result = fit(X, y, family='poisson', backend='gpu')
        assert result.converged

    def test_residuals_match(self):
        X, y = _make_poisson_data()
        design = Design.from_arrays(X, y)

        cpu_result = fit(design, family='poisson', backend='cpu')
        gpu_result = fit(design, family='poisson', backend='gpu')

        np.testing.assert_allclose(
            gpu_result.residuals_deviance, cpu_result.residuals_deviance,
            rtol=1e-3, err_msg="Poisson GPU deviance residuals differ from CPU"
        )
        np.testing.assert_allclose(
            gpu_result.residuals_pearson, cpu_result.residuals_pearson,
            rtol=1e-3, err_msg="Poisson GPU Pearson residuals differ from CPU"
        )


# =====================================================================
# GPU metadata tests
# =====================================================================

class TestGLMGPUMetadata:
    """Verify GPU backend reports correct metadata."""

    def test_backend_name(self):
        X, y = _make_gaussian_data(n=100)
        result = fit(X, y, family='gaussian', backend='gpu')
        assert 'gpu' in result.backend_name

    def test_timing_sections(self):
        X, y = _make_binomial_data(n=200)
        result = fit(X, y, family='binomial', backend='gpu')
        assert result.timing is not None
        assert 'total_seconds' in result.timing
        for section in ('data_transfer_to_gpu', 'irls',
                        'data_transfer_to_cpu', 'residuals'):
            assert section in result.timing, f"Missing timing section: {section}"

    def test_info_contains_device(self):
        X, y = _make_poisson_data(n=200)
        result = fit(X, y, family='poisson', backend='gpu')
        assert 'device' in result.info
        assert 'method' in result.info

    def test_family_and_link_reported(self):
        X, y = _make_binomial_data(n=200)
        result = fit(X, y, family='binomial', backend='gpu')
        assert result.family_name == 'binomial'
        assert result.link_name == 'logit'


# =====================================================================
# GPU stress tests for GLM
# =====================================================================

class TestGLMGPUStress:
    """GLM GPU correctness at larger scale."""

    def test_large_binomial(self):
        """Large logistic regression: GPU matches CPU."""
        n, p = 50_000, 20
        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
        beta = rng.standard_normal(p) * 0.5
        eta = X @ beta
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob).astype(np.float64)
        design = Design.from_arrays(X, y)

        cpu_result = fit(design, family='binomial', backend='cpu')
        gpu_result = fit(design, family='binomial', backend='gpu')

        # At n=50k, FP32 accumulation can drift slightly
        np.testing.assert_allclose(
            gpu_result.coefficients, cpu_result.coefficients,
            rtol=1e-2,
            err_msg="Large binomial: GPU coefficients differ from CPU"
        )
        # Correlation should be near-perfect
        corr = np.corrcoef(
            gpu_result.coefficients, cpu_result.coefficients
        )[0, 1]
        assert corr > 0.9999

    def test_large_poisson(self):
        """Large Poisson regression: GPU matches CPU."""
        n, p = 50_000, 15
        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1)) * 0.3])
        beta = np.array([2.0] + list(rng.standard_normal(p - 1) * 0.2))
        eta = X @ beta
        mu = np.exp(eta)
        y = rng.poisson(mu).astype(np.float64)
        design = Design.from_arrays(X, y)

        cpu_result = fit(design, family='poisson', backend='cpu')
        gpu_result = fit(design, family='poisson', backend='gpu')

        np.testing.assert_allclose(
            gpu_result.coefficients, cpu_result.coefficients,
            rtol=1e-2,
            err_msg="Large Poisson: GPU coefficients differ from CPU"
        )
        corr = np.corrcoef(
            gpu_result.coefficients, cpu_result.coefficients
        )[0, 1]
        assert corr > 0.9999
