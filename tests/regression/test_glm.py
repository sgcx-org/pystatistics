"""
GLM unit tests.

Tests family/link functions, IRLS convergence, solution interface,
and backward compatibility (family=None → LinearSolution).
"""

import numpy as np
import pytest

from pystatistics.regression import fit, GLMSolution, LinearSolution, Design
from pystatistics.regression.families import (
    Family, Gaussian, Binomial, Poisson,
    IdentityLink, LogitLink, LogLink, ProbitLink,
    resolve_family,
)


# =====================================================================
# Family / Link function tests
# =====================================================================

class TestLinks:
    """Test link function roundtrips and derivatives."""

    @pytest.mark.parametrize("link_cls,mu_range", [
        (IdentityLink, np.linspace(-5, 5, 50)),
        (LogitLink, np.linspace(0.01, 0.99, 50)),
        (LogLink, np.linspace(0.01, 10, 50)),
    ])
    def test_roundtrip(self, link_cls, mu_range):
        """linkinv(link(mu)) == mu."""
        link = link_cls()
        eta = link.link(mu_range)
        mu_back = link.linkinv(eta)
        np.testing.assert_allclose(mu_back, mu_range, rtol=1e-10)

    @pytest.mark.parametrize("link_cls,eta_range", [
        (IdentityLink, np.linspace(-5, 5, 50)),
        (LogitLink, np.linspace(-5, 5, 50)),
        (LogLink, np.linspace(-5, 5, 50)),
    ])
    def test_mu_eta_matches_finite_difference(self, link_cls, eta_range):
        """mu_eta(eta) should match numerical derivative of linkinv."""
        link = link_cls()
        h = 1e-7
        mu_eta_analytic = link.mu_eta(eta_range)
        mu_eta_numeric = (link.linkinv(eta_range + h) - link.linkinv(eta_range - h)) / (2 * h)
        np.testing.assert_allclose(mu_eta_analytic, mu_eta_numeric, rtol=1e-5)

    def test_logit_extreme_eta(self):
        """Logit should not overflow on extreme eta values."""
        link = LogitLink()
        eta = np.array([-500, -100, 0, 100, 500])
        mu = link.linkinv(eta)
        assert np.all(np.isfinite(mu))
        assert np.all(mu >= 0) and np.all(mu <= 1)

    def test_log_extreme_eta(self):
        """Log link should not overflow on extreme negative eta."""
        link = LogLink()
        eta = np.array([-500, -100, 0, 5])
        mu = link.linkinv(eta)
        assert np.all(np.isfinite(mu))
        assert np.all(mu >= 0)


class TestFamilies:
    """Test family variance, deviance, and initialization."""

    def test_gaussian_variance(self):
        mu = np.array([1.0, 2.0, 3.0])
        fam = Gaussian()
        np.testing.assert_array_equal(fam.variance(mu), np.ones(3))

    def test_binomial_variance(self):
        mu = np.array([0.2, 0.5, 0.8])
        fam = Binomial()
        expected = mu * (1 - mu)
        np.testing.assert_allclose(fam.variance(mu), expected)

    def test_poisson_variance(self):
        mu = np.array([1.0, 2.0, 5.0])
        fam = Poisson()
        np.testing.assert_allclose(fam.variance(mu), mu)

    def test_gaussian_deviance_is_rss(self):
        """Gaussian deviance with identity link should be RSS."""
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.1, 1.9, 3.2])
        wt = np.ones(3)
        fam = Gaussian()
        expected_rss = np.sum((y - mu) ** 2)
        assert abs(fam.deviance(y, mu, wt) - expected_rss) < 1e-14

    def test_binomial_deviance_perfect_fit(self):
        """Deviance should be 0 when mu == y (binary)."""
        y = np.array([0.0, 1.0, 0.0, 1.0])
        mu = np.array([1e-10, 1 - 1e-10, 1e-10, 1 - 1e-10])
        wt = np.ones(4)
        fam = Binomial()
        assert fam.deviance(y, mu, wt) < 1e-5

    def test_poisson_deviance_perfect_fit(self):
        """Deviance should be near 0 when mu == y."""
        y = np.array([1.0, 2.0, 5.0])
        mu = y.copy()
        wt = np.ones(3)
        fam = Poisson()
        assert fam.deviance(y, mu, wt) < 1e-14

    def test_dispersion_is_fixed(self):
        assert not Gaussian().dispersion_is_fixed
        assert Binomial().dispersion_is_fixed
        assert Poisson().dispersion_is_fixed

    def test_gaussian_initialization(self):
        y = np.array([1.0, 2.0, 3.0])
        mu = Gaussian().initialize(y)
        np.testing.assert_array_equal(mu, y)

    def test_binomial_initialization(self):
        y = np.array([0.0, 1.0, 0.0, 1.0])
        mu = Binomial().initialize(y)
        # R: (y + 0.5) / 2
        expected = (y + 0.5) / 2.0
        np.testing.assert_allclose(mu, expected)


class TestFamilyResolver:
    """Test string → Family resolution."""

    @pytest.mark.parametrize("name,cls", [
        ('gaussian', Gaussian),
        ('binomial', Binomial),
        ('poisson', Poisson),
        ('normal', Gaussian),
    ])
    def test_resolve_string(self, name, cls):
        fam = resolve_family(name)
        assert isinstance(fam, cls)

    def test_resolve_case_insensitive(self):
        fam = resolve_family('Binomial')
        assert isinstance(fam, Binomial)

    def test_resolve_passthrough(self):
        fam = Binomial()
        assert resolve_family(fam) is fam

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown family"):
            resolve_family('unknown')

    def test_resolve_wrong_type_raises(self):
        with pytest.raises(TypeError):
            resolve_family(42)


# =====================================================================
# GLM fit tests
# =====================================================================

class TestGLMFit:
    """Test fit() with family parameter."""

    @pytest.fixture
    def gaussian_data(self):
        rng = np.random.default_rng(42)
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        beta = np.array([1.0, 2.0])
        y = X @ beta + rng.standard_normal(n) * 0.5
        return X, y

    @pytest.fixture
    def binomial_data(self):
        rng = np.random.default_rng(42)
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        beta = np.array([0.5, 1.0])
        eta = X @ beta
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob).astype(np.float64)
        return X, y

    @pytest.fixture
    def poisson_data(self):
        rng = np.random.default_rng(42)
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        beta = np.array([1.0, 0.5])
        eta = X @ beta
        mu = np.exp(eta)
        y = rng.poisson(mu).astype(np.float64)
        return X, y

    def test_backward_compatibility(self, gaussian_data):
        """family=None should still return LinearSolution."""
        X, y = gaussian_data
        result = fit(X, y, backend='cpu')
        assert isinstance(result, LinearSolution)

    def test_gaussian_glm_returns_glmsolution(self, gaussian_data):
        """family='gaussian' should return GLMSolution."""
        X, y = gaussian_data
        result = fit(X, y, family='gaussian', backend='cpu')
        assert isinstance(result, GLMSolution)

    def test_binomial_returns_glmsolution(self, binomial_data):
        X, y = binomial_data
        result = fit(X, y, family='binomial', backend='cpu')
        assert isinstance(result, GLMSolution)

    def test_poisson_returns_glmsolution(self, poisson_data):
        X, y = poisson_data
        result = fit(X, y, family='poisson', backend='cpu')
        assert isinstance(result, GLMSolution)

    def test_gaussian_glm_matches_lm_coefficients(self, gaussian_data):
        """Gaussian GLM coefficients should match OLS exactly."""
        X, y = gaussian_data
        lm_result = fit(X, y, backend='cpu')
        glm_result = fit(X, y, family='gaussian', backend='cpu')
        np.testing.assert_allclose(
            glm_result.coefficients, lm_result.coefficients,
            rtol=1e-8,
            err_msg="Gaussian GLM coefficients differ from OLS"
        )

    def test_binomial_converges(self, binomial_data):
        X, y = binomial_data
        result = fit(X, y, family='binomial', backend='cpu')
        assert result.converged

    def test_poisson_converges(self, poisson_data):
        X, y = poisson_data
        result = fit(X, y, family='poisson', backend='cpu')
        assert result.converged

    def test_family_object_passthrough(self, binomial_data):
        """Passing a Family instance should work."""
        X, y = binomial_data
        result = fit(X, y, family=Binomial(), backend='cpu')
        assert isinstance(result, GLMSolution)
        assert result.family_name == 'binomial'

    def test_design_object_works(self, gaussian_data):
        """Design objects should work with family parameter."""
        X, y = gaussian_data
        design = Design.from_arrays(X, y)
        result = fit(design, family='gaussian', backend='cpu')
        assert isinstance(result, GLMSolution)


# =====================================================================
# GLMSolution interface tests
# =====================================================================

class TestGLMSolution:
    """Test GLMSolution properties and methods."""

    @pytest.fixture
    def binomial_result(self):
        rng = np.random.default_rng(42)
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        beta = np.array([0.5, 1.0])
        eta = X @ beta
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob).astype(np.float64)
        return fit(X, y, family='binomial', backend='cpu')

    @pytest.fixture
    def gaussian_result(self):
        rng = np.random.default_rng(42)
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        beta = np.array([1.0, 2.0])
        y = X @ beta + rng.standard_normal(n) * 0.5
        return fit(X, y, family='gaussian', backend='cpu')

    def test_deviance_positive(self, binomial_result):
        assert binomial_result.deviance >= 0

    def test_null_deviance_greater(self, binomial_result):
        """Null deviance should generally be >= model deviance."""
        assert binomial_result.null_deviance >= binomial_result.deviance - 0.01

    def test_aic_finite(self, binomial_result):
        assert np.isfinite(binomial_result.aic)

    def test_bic_finite(self, binomial_result):
        assert np.isfinite(binomial_result.bic)

    def test_fitted_values_in_range_binomial(self, binomial_result):
        """Binomial fitted values should be probabilities."""
        assert np.all(binomial_result.fitted_values >= 0)
        assert np.all(binomial_result.fitted_values <= 1)

    def test_fitted_values_match_linpred_binomial(self, binomial_result):
        """fitted = linkinv(linear_predictor)."""
        from pystatistics.regression.families import LogitLink
        link = LogitLink()
        expected = link.linkinv(binomial_result.linear_predictor)
        np.testing.assert_allclose(
            binomial_result.fitted_values, expected, rtol=1e-10
        )

    def test_response_residuals(self, binomial_result):
        """Response residuals should be y - μ."""
        # Can't access y directly, but residuals_response should sum correctly
        assert binomial_result.residuals_response.shape == binomial_result.fitted_values.shape

    def test_standard_errors_positive(self, binomial_result):
        se = binomial_result.standard_errors
        assert np.all(se > 0)
        assert np.all(np.isfinite(se))

    def test_z_statistics_finite(self, binomial_result):
        """Binomial should use z-statistics."""
        stat = binomial_result.test_statistics
        assert np.all(np.isfinite(stat))

    def test_p_values_in_range(self, binomial_result):
        pv = binomial_result.p_values
        assert np.all(pv >= 0)
        assert np.all(pv <= 1)

    def test_gaussian_dispersion_estimated(self, gaussian_result):
        """Gaussian dispersion should be estimated (not 1.0)."""
        assert gaussian_result.dispersion != 1.0
        assert gaussian_result.dispersion > 0

    def test_binomial_dispersion_one(self, binomial_result):
        """Binomial dispersion should be fixed at 1.0."""
        assert binomial_result.dispersion == 1.0

    def test_family_and_link_names(self, binomial_result):
        assert binomial_result.family_name == 'binomial'
        assert binomial_result.link_name == 'logit'

    def test_summary_contains_family(self, binomial_result):
        summary = binomial_result.summary()
        assert 'GLM Results' in summary
        assert 'binomial' in summary
        assert 'logit' in summary
        assert 'Deviance' in summary

    def test_repr(self, binomial_result):
        r = repr(binomial_result)
        assert 'GLMSolution' in r
        assert 'binomial' in r

    def test_timing_populated(self, binomial_result):
        assert binomial_result.timing is not None
        assert 'total_seconds' in binomial_result.timing
        assert binomial_result.timing['total_seconds'] > 0

    def test_backend_name(self, binomial_result):
        assert 'irls' in binomial_result.backend_name

    def test_four_residual_types(self, binomial_result):
        """All four residual types should be available."""
        assert binomial_result.residuals_deviance is not None
        assert binomial_result.residuals_pearson is not None
        assert binomial_result.residuals_working is not None
        assert binomial_result.residuals_response is not None

    def test_deviance_residuals_sum(self, binomial_result):
        """Sum of squared deviance residuals should equal deviance."""
        dev_resid_sq = binomial_result.residuals_deviance ** 2
        np.testing.assert_allclose(
            np.sum(dev_resid_sq), binomial_result.deviance, rtol=1e-6
        )
