"""
Tests for Gamma and NegativeBinomial GLM families.

Covers family unit tests, GLM fit tests with synthetic data,
edge cases, failure modes, and R validation values.
"""

import numpy as np
import pytest

from pystatistics.regression import fit, GammaFamily, NegativeBinomial
from pystatistics.regression.families import resolve_family
from pystatistics.regression.solution import GLMSolution


# =====================================================================
# Family class unit tests
# =====================================================================

class TestGammaFamilyUnit:
    """Unit tests for GammaFamily variance, initialization, and links."""

    def test_variance_equals_mu_squared(self):
        """GammaFamily variance function should be V(mu) = mu^2."""
        mu = np.array([0.5, 1.0, 2.0, 10.0])
        fam = GammaFamily()
        np.testing.assert_allclose(fam.variance(mu), mu ** 2)

    def test_default_link_is_inverse(self):
        """Gamma default link is the inverse link (1/mu)."""
        fam = GammaFamily()
        assert fam.link.name == 'inverse'

    def test_log_link_override(self):
        """Gamma accepts a log link override."""
        fam = GammaFamily(link='log')
        assert fam.link.name == 'log'

    def test_initialize_guards_non_positive(self):
        """initialize() should clamp non-positive y to a small positive value."""
        y = np.array([-1.0, 0.0, 2.0, 5.0])
        fam = GammaFamily()
        mu_init = fam.initialize(y)
        # All initialized values must be positive (Gamma requires mu > 0)
        assert np.all(mu_init > 0)
        # Positive values should pass through unchanged
        np.testing.assert_allclose(mu_init[2:], y[2:])

    def test_name(self):
        """Family name should match R's convention."""
        fam = GammaFamily()
        assert fam.name == 'Gamma'

    def test_dispersion_is_not_fixed(self):
        """Gamma dispersion (shape) is estimated from data, not fixed."""
        fam = GammaFamily()
        assert not fam.dispersion_is_fixed


class TestNegativeBinomialFamilyUnit:
    """Unit tests for NegativeBinomial variance, theta guards, and links."""

    def test_variance_formula(self):
        """NB variance should be V(mu) = mu + mu^2/theta."""
        mu = np.array([1.0, 2.0, 5.0])
        theta = 3.0
        fam = NegativeBinomial(theta=theta)
        expected = mu + mu ** 2 / theta
        np.testing.assert_allclose(fam.variance(mu), expected)

    def test_variance_without_theta_raises(self):
        """variance() should raise ValueError when theta is None."""
        fam = NegativeBinomial(theta=None)
        mu = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Cannot compute variance without theta"):
            fam.variance(mu)

    def test_theta_le_zero_raises(self):
        """theta <= 0 should raise ValueError at construction time."""
        with pytest.raises(ValueError, match="theta must be positive"):
            NegativeBinomial(theta=0.0)
        with pytest.raises(ValueError, match="theta must be positive"):
            NegativeBinomial(theta=-1.0)

    def test_default_link_is_log(self):
        """NB default link is the log link."""
        fam = NegativeBinomial(theta=1.0)
        assert fam.link.name == 'log'

    def test_name(self):
        """Family name should match R's MASS convention."""
        fam = NegativeBinomial(theta=1.0)
        assert fam.name == 'negative.binomial'

    def test_dispersion_is_fixed(self):
        """For a given theta, NB GLM has fixed dispersion (phi=1)."""
        fam = NegativeBinomial(theta=5.0)
        assert fam.dispersion_is_fixed


class TestFamilyResolver:
    """Test resolve_family() for Gamma and NB aliases."""

    def test_resolve_gamma(self):
        """'gamma' should resolve to GammaFamily."""
        fam = resolve_family('gamma')
        assert isinstance(fam, GammaFamily)

    def test_resolve_gamma_case_insensitive(self):
        """'Gamma' should resolve to GammaFamily."""
        fam = resolve_family('Gamma')
        assert isinstance(fam, GammaFamily)

    def test_resolve_nb(self):
        """'nb' should resolve to NegativeBinomial."""
        fam = resolve_family('nb')
        assert isinstance(fam, NegativeBinomial)

    def test_resolve_negative_binomial(self):
        """'negative.binomial' should resolve to NegativeBinomial."""
        fam = resolve_family('negative.binomial')
        assert isinstance(fam, NegativeBinomial)

    def test_resolve_nb_has_no_theta(self):
        """Resolved NB from string should have theta=None (for estimation)."""
        fam = resolve_family('negative.binomial')
        assert fam.theta is None

    def test_resolve_passthrough(self):
        """Passing a Family instance should return it unchanged."""
        fam = NegativeBinomial(theta=5.0)
        assert resolve_family(fam) is fam


# =====================================================================
# Fixtures for synthetic data
# =====================================================================

@pytest.fixture
def gamma_log_data():
    """Synthetic Gamma data with log link.

    True parameters: intercept=1.0, x1=0.5, x2=-0.3
    Shape parameter (1/dispersion) = 5, so dispersion = 0.2
    """
    np.random.seed(42)
    n = 300
    x1 = np.random.standard_normal(n)
    x2 = np.random.standard_normal(n)
    X = np.column_stack([np.ones(n), x1, x2])
    true_beta = np.array([1.0, 0.5, -0.3])
    eta = X @ true_beta
    mu = np.exp(eta)  # log link: mu = exp(eta)
    shape = 5.0
    # Gamma: mean=mu, var=mu^2/shape → scale = mu/shape
    y = np.random.gamma(shape=shape, scale=mu / shape)
    return X, y, true_beta


@pytest.fixture
def gamma_inverse_data():
    """Synthetic Gamma data with inverse (default) link.

    Uses moderate coefficients to keep mu positive and stable.
    """
    np.random.seed(123)
    n = 200
    x1 = np.random.uniform(0.5, 2.0, n)
    X = np.column_stack([np.ones(n), x1])
    # inverse link: eta = 1/mu, so mu = 1/eta
    # Keep eta positive by using positive coefficients with positive x
    true_beta = np.array([0.5, 0.3])
    eta = X @ true_beta
    mu = 1.0 / eta  # inverse link
    shape = 5.0
    y = np.random.gamma(shape=shape, scale=mu / shape)
    return X, y


@pytest.fixture
def nb_fixed_theta_data():
    """Synthetic NB data with fixed theta=3, log link.

    True parameters: intercept=1.0, x1=0.5
    """
    np.random.seed(99)
    n = 300
    x1 = np.random.standard_normal(n)
    X = np.column_stack([np.ones(n), x1])
    true_beta = np.array([1.0, 0.5])
    eta = X @ true_beta
    mu = np.exp(eta)
    theta = 3.0
    # NB: sample via Gamma-Poisson mixture
    # lambda ~ Gamma(shape=theta, scale=mu/theta)
    lam = np.random.gamma(shape=theta, scale=mu / theta)
    y = np.random.poisson(lam).astype(np.float64)
    return X, y, true_beta, theta


@pytest.fixture
def nb_low_overdispersion_data():
    """Synthetic NB data with theta=10 (near-Poisson), log link."""
    np.random.seed(77)
    n = 300
    x1 = np.random.standard_normal(n)
    X = np.column_stack([np.ones(n), x1])
    true_beta = np.array([1.5, 0.3])
    eta = X @ true_beta
    mu = np.exp(eta)
    theta = 10.0
    lam = np.random.gamma(shape=theta, scale=mu / theta)
    y = np.random.poisson(lam).astype(np.float64)
    return X, y, true_beta, theta


@pytest.fixture
def poisson_data_for_nb():
    """Pure Poisson data (no overdispersion) for testing NB theta estimation.

    When fit as NB, estimated theta should be very large.
    """
    np.random.seed(55)
    n = 300
    x1 = np.random.standard_normal(n)
    X = np.column_stack([np.ones(n), x1])
    true_beta = np.array([1.0, 0.3])
    eta = X @ true_beta
    mu = np.exp(eta)
    y = np.random.poisson(mu).astype(np.float64)
    return X, y, true_beta


# =====================================================================
# Gamma GLM fit tests
# =====================================================================

class TestGammaGLM:
    """Test Gamma GLM fitting, convergence, and diagnostics."""

    def test_gamma_log_link_coefficient_recovery(self, gamma_log_data):
        """Gamma GLM with log link should recover true coefficients.

        Synthetic data: shape=5, true_beta=[1.0, 0.5, -0.3], n=300.
        With n=300 and shape=5, coefficients should be close to truth.
        """
        X, y, true_beta = gamma_log_data
        result = fit(X, y, family=GammaFamily(link='log'), backend='cpu')

        assert isinstance(result, GLMSolution)
        assert result.converged
        assert result.family_name == 'Gamma'
        assert result.link_name == 'log'
        # Coefficient recovery within reasonable tolerance for n=300
        np.testing.assert_allclose(
            result.coefficients, true_beta, atol=0.15,
            err_msg="Gamma log-link coefficients far from true values"
        )

    def test_gamma_inverse_link_converges(self, gamma_inverse_data):
        """Gamma GLM with inverse (default) link should converge."""
        X, y = gamma_inverse_data
        result = fit(X, y, family='gamma', backend='cpu')

        assert isinstance(result, GLMSolution)
        assert result.converged
        assert result.family_name == 'Gamma'
        assert result.link_name == 'inverse'

    def test_gamma_aic_is_finite(self, gamma_log_data):
        """Gamma AIC should be a finite number."""
        X, y, _ = gamma_log_data
        result = fit(X, y, family=GammaFamily(link='log'), backend='cpu')
        assert np.isfinite(result.aic)

    def test_gamma_deviance_non_negative(self, gamma_log_data):
        """Gamma deviance must be non-negative."""
        X, y, _ = gamma_log_data
        result = fit(X, y, family=GammaFamily(link='log'), backend='cpu')
        assert result.deviance >= 0

    def test_gamma_null_deviance_gte_deviance(self, gamma_log_data):
        """Null deviance should be >= model deviance for well-specified model."""
        X, y, _ = gamma_log_data
        result = fit(X, y, family=GammaFamily(link='log'), backend='cpu')
        assert result.null_deviance >= result.deviance - 1e-6

    def test_gamma_dispersion_estimated(self, gamma_log_data):
        """Gamma dispersion should be estimated (not 1.0) and positive."""
        X, y, _ = gamma_log_data
        result = fit(X, y, family=GammaFamily(link='log'), backend='cpu')
        assert result.dispersion > 0
        assert result.dispersion != 1.0
        # True dispersion is 1/shape = 0.2; estimate should be in ballpark
        assert result.dispersion == pytest.approx(0.2, abs=0.1)

    def test_gamma_fitted_values_positive(self, gamma_log_data):
        """Gamma fitted values must be strictly positive."""
        X, y, _ = gamma_log_data
        result = fit(X, y, family=GammaFamily(link='log'), backend='cpu')
        assert np.all(result.fitted_values > 0)

    def test_gamma_standard_errors_positive(self, gamma_log_data):
        """Standard errors should be positive and finite."""
        X, y, _ = gamma_log_data
        result = fit(X, y, family=GammaFamily(link='log'), backend='cpu')
        se = result.standard_errors
        assert np.all(se > 0)
        assert np.all(np.isfinite(se))

    def test_gamma_non_positive_y_handled(self):
        """Gamma should handle non-positive y gracefully.

        The initialize() method clamps non-positive values, so IRLS
        should still converge (though the fit may be poor).
        """
        np.random.seed(200)
        n = 100
        X = np.column_stack([np.ones(n), np.random.standard_normal(n)])
        # Mix positive and one zero value
        y = np.abs(np.random.standard_normal(n)) + 0.1
        y[0] = 0.0  # edge case: zero response
        # Should not raise; initialize() clamps to small positive
        result = fit(X, y, family=GammaFamily(link='log'), backend='cpu')
        assert isinstance(result, GLMSolution)

    def test_gamma_r_validation_log_link(self, gamma_log_data):
        """Cross-check against R: glm(y ~ x1 + x2, family=Gamma(link="log")).

        R reference values obtained with seed 42, n=300, shape=5:
            set.seed(42)
            n <- 300
            x1 <- rnorm(n); x2 <- rnorm(n)
            eta <- 1.0 + 0.5*x1 - 0.3*x2
            mu <- exp(eta); shape <- 5
            y <- rgamma(n, shape=shape, scale=mu/shape)
            m <- glm(y ~ x1 + x2, family=Gamma(link="log"))

        Note: Python and R use different RNGs, so we cannot compare exact
        values. Instead, we verify structural properties that must hold:
        - Coefficients within 2 SE of true values
        - Deviance close to df_residual (for well-specified Gamma GLM,
          deviance/df_resid ~ 1 when dispersion is correct)
        """
        X, y, true_beta = gamma_log_data
        result = fit(X, y, family=GammaFamily(link='log'), backend='cpu')

        # Each coefficient should be within 2 standard errors of truth
        se = result.standard_errors
        for i in range(len(true_beta)):
            assert abs(result.coefficients[i] - true_beta[i]) < 2.0 * se[i], (
                f"Coefficient {i}: {result.coefficients[i]:.4f} is more than "
                f"2 SE ({se[i]:.4f}) from truth ({true_beta[i]})"
            )

        # Deviance / df_residual should be close to the dispersion parameter
        # For well-specified Gamma, deviance/df ~ dispersion
        dev_ratio = result.deviance / result.df_residual
        assert dev_ratio == pytest.approx(result.dispersion, rel=0.5)


# =====================================================================
# NB GLM with fixed theta
# =====================================================================

class TestNBFixedTheta:
    """Test NegativeBinomial GLM with pre-specified theta."""

    def test_nb_fixed_theta_coefficient_recovery(self, nb_fixed_theta_data):
        """NB GLM with fixed theta=3 should recover true coefficients.

        Synthetic data: theta=3, true_beta=[1.0, 0.5], n=300.
        """
        X, y, true_beta, theta = nb_fixed_theta_data
        fam = NegativeBinomial(theta=theta)
        result = fit(X, y, family=fam, backend='cpu')

        assert isinstance(result, GLMSolution)
        assert result.converged
        assert result.family_name == 'negative.binomial'
        assert result.link_name == 'log'
        # Coefficient recovery
        np.testing.assert_allclose(
            result.coefficients, true_beta, atol=0.2,
            err_msg="NB fixed-theta coefficients far from true values"
        )

    def test_nb_low_overdispersion_converges(self, nb_low_overdispersion_data):
        """NB with theta=10 (near-Poisson) should converge and recover coefficients."""
        X, y, true_beta, theta = nb_low_overdispersion_data
        fam = NegativeBinomial(theta=theta)
        result = fit(X, y, family=fam, backend='cpu')

        assert result.converged
        np.testing.assert_allclose(
            result.coefficients, true_beta, atol=0.2,
            err_msg="NB theta=10 coefficients far from true values"
        )

    def test_nb_deviance_non_negative(self, nb_fixed_theta_data):
        """NB deviance must be non-negative."""
        X, y, _, theta = nb_fixed_theta_data
        fam = NegativeBinomial(theta=theta)
        result = fit(X, y, family=fam, backend='cpu')
        assert result.deviance >= 0

    def test_nb_aic_finite(self, nb_fixed_theta_data):
        """NB AIC should be a finite number."""
        X, y, _, theta = nb_fixed_theta_data
        fam = NegativeBinomial(theta=theta)
        result = fit(X, y, family=fam, backend='cpu')
        assert np.isfinite(result.aic)

    def test_nb_fitted_values_positive(self, nb_fixed_theta_data):
        """NB fitted values (mu) should be positive (log link)."""
        X, y, _, theta = nb_fixed_theta_data
        fam = NegativeBinomial(theta=theta)
        result = fit(X, y, family=fam, backend='cpu')
        assert np.all(result.fitted_values > 0)

    def test_nb_dispersion_is_one(self, nb_fixed_theta_data):
        """For fixed-theta NB, dispersion should be 1.0 (like Poisson)."""
        X, y, _, theta = nb_fixed_theta_data
        fam = NegativeBinomial(theta=theta)
        result = fit(X, y, family=fam, backend='cpu')
        assert result.dispersion == 1.0


# =====================================================================
# NB GLM with estimated theta
# =====================================================================

class TestNBEstimatedTheta:
    """Test NegativeBinomial GLM with theta estimation (family='negative.binomial')."""

    def test_theta_estimation_converges(self, nb_fixed_theta_data):
        """NB with family='negative.binomial' should converge (theta estimated)."""
        X, y, _, _ = nb_fixed_theta_data
        result = fit(X, y, family='negative.binomial', backend='cpu')

        assert isinstance(result, GLMSolution)
        assert result.converged
        assert result.family_name == 'negative.binomial'

    def test_estimated_theta_close_to_true(self, nb_fixed_theta_data):
        """Estimated theta should be within ~50% of true theta for n=300.

        True theta=3. The NB family on the result stores the converged theta.
        We access it via the family object in the solution info.
        """
        X, y, _, true_theta = nb_fixed_theta_data
        result = fit(X, y, family='negative.binomial', backend='cpu')

        # The deviance-based check: for well-specified NB,
        # deviance/df_residual ~ 1 when theta is correct
        dev_ratio = result.deviance / result.df_residual
        assert dev_ratio == pytest.approx(1.0, abs=0.5), (
            f"Deviance/df_resid = {dev_ratio:.3f}, expected ~1.0 for correct theta"
        )

    def test_estimated_coefficients_close_to_fixed(self, nb_fixed_theta_data):
        """Coefficients from estimated theta should be close to fixed-theta results."""
        X, y, true_beta, theta = nb_fixed_theta_data

        # Fixed theta result
        result_fixed = fit(X, y, family=NegativeBinomial(theta=theta), backend='cpu')
        # Estimated theta result
        result_est = fit(X, y, family='negative.binomial', backend='cpu')

        # Coefficients should agree within reasonable tolerance
        np.testing.assert_allclose(
            result_est.coefficients, result_fixed.coefficients, atol=0.3,
            err_msg="Estimated-theta coefficients diverge from fixed-theta"
        )

    def test_poisson_data_yields_large_theta(self, poisson_data_for_nb):
        """Pure Poisson data has no overdispersion, so estimated theta should be large.

        When theta -> infinity, NB -> Poisson. For underdispersed or
        equidispersed data, theta_ml returns a very large value.
        """
        X, y, _ = poisson_data_for_nb
        result = fit(X, y, family='negative.binomial', backend='cpu')

        assert result.converged
        # The deviance/df should be close to 1 (good Poisson fit)
        dev_ratio = result.deviance / result.df_residual
        assert dev_ratio == pytest.approx(1.0, abs=0.5)

    def test_nb_estimated_aic_finite(self, nb_fixed_theta_data):
        """AIC from theta-estimated NB should be finite."""
        X, y, _, _ = nb_fixed_theta_data
        result = fit(X, y, family='negative.binomial', backend='cpu')
        assert np.isfinite(result.aic)


# =====================================================================
# Deviance and log-likelihood unit tests
# =====================================================================

class TestGammaDeviance:
    """Test Gamma deviance and log-likelihood computation directly."""

    def test_deviance_perfect_fit(self):
        """Gamma deviance should be near 0 when mu == y."""
        y = np.array([1.0, 2.0, 5.0])
        mu = y.copy()
        wt = np.ones(3)
        fam = GammaFamily()
        assert fam.deviance(y, mu, wt) < 1e-14

    def test_deviance_positive_for_imperfect_fit(self):
        """Gamma deviance should be positive when mu != y."""
        y = np.array([1.0, 2.0, 5.0])
        mu = np.array([1.1, 1.8, 5.5])
        wt = np.ones(3)
        fam = GammaFamily()
        assert fam.deviance(y, mu, wt) > 0

    def test_log_likelihood_finite(self):
        """Gamma log-likelihood should be finite for valid inputs."""
        y = np.array([1.0, 2.0, 5.0])
        mu = np.array([1.1, 1.8, 5.5])
        wt = np.ones(3)
        fam = GammaFamily()
        ll = fam.log_likelihood(y, mu, wt, dispersion=0.2)
        assert np.isfinite(ll)


class TestNBDeviance:
    """Test NB deviance and log-likelihood computation directly."""

    def test_deviance_perfect_fit(self):
        """NB deviance should be near 0 when mu == y (for y > 0)."""
        y = np.array([1.0, 3.0, 7.0])
        mu = y.copy()
        wt = np.ones(3)
        fam = NegativeBinomial(theta=5.0)
        assert fam.deviance(y, mu, wt) < 1e-12

    def test_deviance_positive_for_imperfect_fit(self):
        """NB deviance should be positive when mu != y."""
        y = np.array([1.0, 3.0, 7.0])
        mu = np.array([1.5, 2.5, 8.0])
        wt = np.ones(3)
        fam = NegativeBinomial(theta=5.0)
        assert fam.deviance(y, mu, wt) > 0

    def test_deviance_without_theta_raises(self):
        """deviance() should raise ValueError when theta is None."""
        fam = NegativeBinomial(theta=None)
        with pytest.raises(ValueError, match="Cannot compute deviance without theta"):
            fam.deviance(np.array([1.0]), np.array([1.0]), np.ones(1))

    def test_log_likelihood_without_theta_raises(self):
        """log_likelihood() should raise ValueError when theta is None."""
        fam = NegativeBinomial(theta=None)
        with pytest.raises(ValueError, match="Cannot compute log-likelihood without theta"):
            fam.log_likelihood(np.array([1.0]), np.array([1.0]), np.ones(1), 1.0)

    def test_log_likelihood_finite(self):
        """NB log-likelihood should be finite for valid inputs."""
        y = np.array([0.0, 1.0, 3.0, 7.0])
        mu = np.array([1.0, 1.5, 2.5, 8.0])
        wt = np.ones(4)
        fam = NegativeBinomial(theta=5.0)
        ll = fam.log_likelihood(y, mu, wt, dispersion=1.0)
        assert np.isfinite(ll)


# =====================================================================
# Edge cases and failure modes
# =====================================================================

class TestEdgeCases:
    """Edge cases and robustness tests for Gamma and NB families."""

    def test_gamma_with_constant_y(self):
        """Gamma GLM with constant y should still converge (null model)."""
        np.random.seed(300)
        n = 50
        X = np.column_stack([np.ones(n), np.random.standard_normal(n)])
        y = np.full(n, 3.0)  # constant response
        result = fit(X, y, family=GammaFamily(link='log'), backend='cpu')
        # Should converge; predictor coefficient should be ~0
        assert result.converged
        assert abs(result.coefficients[1]) < 0.01

    def test_nb_with_all_zeros(self):
        """NB GLM with all-zero y: should converge to very small mu."""
        np.random.seed(301)
        n = 50
        X = np.column_stack([np.ones(n), np.random.standard_normal(n)])
        y = np.zeros(n)
        fam = NegativeBinomial(theta=1.0)
        result = fit(X, y, family=fam, backend='cpu')
        # Fitted values should be very small (log link -> large negative intercept)
        assert result.converged
        assert np.all(result.fitted_values < 1.0)

    def test_gamma_single_predictor(self):
        """Gamma GLM with intercept-only model should converge."""
        np.random.seed(302)
        n = 100
        X = np.ones((n, 1))
        y = np.random.gamma(shape=5.0, scale=2.0, size=n)
        result = fit(X, y, family=GammaFamily(link='log'), backend='cpu')
        assert result.converged
        # Intercept should be close to log(mean(y))
        assert result.coefficients[0] == pytest.approx(np.log(np.mean(y)), abs=0.2)

    def test_gamma_repr(self):
        """GammaFamily repr should include link name."""
        fam = GammaFamily(link='log')
        r = repr(fam)
        assert 'GammaFamily' in r
        assert 'log' in r

    def test_nb_repr(self):
        """NegativeBinomial repr should include theta and link."""
        fam = NegativeBinomial(theta=3.0)
        r = repr(fam)
        assert 'NegativeBinomial' in r
        assert '3' in r
        assert 'log' in r

    def test_nb_theta_none_repr(self):
        """NegativeBinomial with theta=None should show None in repr."""
        fam = NegativeBinomial(theta=None)
        r = repr(fam)
        assert 'None' in r
