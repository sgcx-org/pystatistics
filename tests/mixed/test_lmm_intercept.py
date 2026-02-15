"""Tests for LMM with random intercepts."""

import numpy as np
import pytest

from pystatistics.mixed import lmm


class TestLMMRandomIntercept:
    """Tests for the basic random intercept model."""

    def test_basic_fit(self, random_intercept_simple):
        """LMM fits and returns LMMSolution."""
        d = random_intercept_simple
        result = lmm(d['y'], d['X'], groups={'group': d['group']})

        assert result.converged
        assert len(result.coefficients) == 2

    def test_fixed_effects_close_to_truth(self, random_intercept_simple):
        """Fixed effects recover true values within reasonable tolerance."""
        d = random_intercept_simple
        result = lmm(d['y'], d['X'], groups={'group': d['group']})

        # Intercept ≈ 5.0, slope ≈ 2.0
        np.testing.assert_allclose(
            result.coefficients[0], d['beta0'], atol=2.0
        )
        np.testing.assert_allclose(
            result.coefficients[1], d['beta1'], atol=0.5
        )

    def test_variance_components(self, random_intercept_simple):
        """Variance components are positive and reasonable."""
        d = random_intercept_simple
        result = lmm(d['y'], d['X'], groups={'group': d['group']})

        vc = result.var_components
        assert len(vc) == 1  # one random intercept
        assert vc[0].group == 'group'
        assert vc[0].name == '(Intercept)'
        assert vc[0].variance > 0
        assert vc[0].std_dev > 0
        assert vc[0].corr is None  # no second term to correlate with

    def test_residual_variance_positive(self, random_intercept_simple):
        """Residual variance is positive."""
        d = random_intercept_simple
        result = lmm(d['y'], d['X'], groups={'group': d['group']})
        assert result.params.residual_variance > 0

    def test_blups_shape(self, random_intercept_simple):
        """BLUPs have correct shape: (n_groups, 1)."""
        d = random_intercept_simple
        result = lmm(d['y'], d['X'], groups={'group': d['group']})

        ranef = result.ranef
        assert 'group' in ranef
        assert ranef['group'].shape == (d['n_groups'], 1)

    def test_blups_sum_near_zero(self, random_intercept_simple):
        """BLUPs should approximately sum to zero (shrinkage property)."""
        d = random_intercept_simple
        result = lmm(d['y'], d['X'], groups={'group': d['group']})

        blups = result.ranef['group'][:, 0]
        assert abs(np.mean(blups)) < 1.0  # not exactly zero due to shrinkage

    def test_icc(self, random_intercept_simple):
        """ICC is between 0 and 1."""
        d = random_intercept_simple
        result = lmm(d['y'], d['X'], groups={'group': d['group']})

        icc = result.icc
        assert 'group' in icc
        assert 0 < icc['group'] < 1

    def test_fitted_plus_residuals_equals_y(self, random_intercept_simple):
        """Fitted values + residuals = y."""
        d = random_intercept_simple
        result = lmm(d['y'], d['X'], groups={'group': d['group']})

        np.testing.assert_allclose(
            result.fitted_values + result.residuals, d['y'], atol=1e-8
        )

    def test_model_fit_stats(self, random_intercept_simple):
        """Log-likelihood, AIC, BIC are finite."""
        d = random_intercept_simple
        result = lmm(d['y'], d['X'], groups={'group': d['group']})

        assert np.isfinite(result.log_likelihood)
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)

    def test_reml_vs_ml(self, random_intercept_simple):
        """REML and ML give different but close results."""
        d = random_intercept_simple
        result_reml = lmm(d['y'], d['X'], groups={'group': d['group']}, reml=True)
        result_ml = lmm(d['y'], d['X'], groups={'group': d['group']}, reml=False)

        # Fixed effects should be very close
        np.testing.assert_allclose(
            result_reml.coefficients, result_ml.coefficients, rtol=0.1
        )

        # REML variance should be >= ML variance (on average)
        # They should differ
        assert result_reml.params.residual_variance != result_ml.params.residual_variance

    def test_summary_output(self, random_intercept_simple):
        """summary() produces non-empty string."""
        d = random_intercept_simple
        result = lmm(d['y'], d['X'], groups={'group': d['group']})

        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 100
        assert 'Random effects:' in s
        assert 'Fixed effects:' in s

    def test_satterthwaite_df_range(self, random_intercept_simple):
        """Satterthwaite df should be >= 1 and finite."""
        d = random_intercept_simple
        result = lmm(d['y'], d['X'], groups={'group': d['group']})

        df = result.df_satterthwaite
        assert len(df) == 2
        for i in range(len(df)):
            assert df[i] >= 1.0
            assert np.isfinite(df[i])

    def test_p_values_defined(self, random_intercept_simple):
        """p-values should be between 0 and 1."""
        d = random_intercept_simple
        result = lmm(d['y'], d['X'], groups={'group': d['group']})

        for p in result.p_values:
            assert 0 <= p <= 1

    def test_fixef_dict(self, random_intercept_simple):
        """fixef property returns correct dict."""
        d = random_intercept_simple
        result = lmm(d['y'], d['X'], groups={'group': d['group']})

        fixef = result.fixef
        assert '(Intercept)' in fixef
        assert 'X1' in fixef


class TestLMMNoEffect:
    """Test LMM when there is no fixed effect signal."""

    def test_no_signal(self, rng):
        """With no signal, intercept-only model should work."""
        n_groups = 10
        n_per = 10
        n = n_groups * n_per

        group = np.repeat(np.arange(n_groups), n_per)
        X = np.ones((n, 1))  # intercept only
        y = rng.normal(0, 1, n)

        result = lmm(y, X, groups={'group': group})
        assert result.converged
        assert len(result.coefficients) == 1
