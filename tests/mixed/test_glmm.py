"""Tests for Generalized Linear Mixed Models (GLMM)."""

import numpy as np
import pytest

from pystatistics.mixed import glmm


class TestGLMMBinomial:
    """Tests for GLMM with binomial family."""

    def test_basic_fit(self, glmm_binomial):
        """Binomial GLMM fits and converges."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        assert result.converged
        assert len(result.coefficients) == 2

    def test_family_info(self, glmm_binomial):
        """Family and link recorded correctly."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        assert result.params.family_name == 'binomial'
        assert result.params.link_name == 'logit'

    def test_fixed_effects_direction(self, glmm_binomial):
        """Fixed effects should have correct sign."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        # True beta1 = 1.0 (positive)
        assert result.coefficients[1] > 0

    def test_variance_component(self, glmm_binomial):
        """Random intercept variance should be positive."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        assert len(result.var_components) == 1
        assert result.var_components[0].variance > 0

    def test_fitted_values_in_range(self, glmm_binomial):
        """Fitted values (probabilities) should be in [0, 1]."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        assert np.all(result.fitted_values >= 0)
        assert np.all(result.fitted_values <= 1)

    def test_deviance_positive(self, glmm_binomial):
        """Deviance should be positive."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        assert result.deviance > 0

    def test_wald_z_statistics(self, glmm_binomial):
        """GLMM uses Wald z-statistics, not t."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        # z = coef / se
        np.testing.assert_allclose(
            result.z_values, result.coefficients / result.se, atol=1e-10
        )

    def test_p_values_defined(self, glmm_binomial):
        """p-values should be between 0 and 1."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        for p in result.p_values:
            assert 0 <= p <= 1

    def test_summary_output(self, glmm_binomial):
        """Summary should mention family and link."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        s = result.summary()
        assert 'binomial' in s
        assert 'logit' in s

    def test_icc_logistic(self, glmm_binomial):
        """ICC on logistic scale: uses π²/3 for residual variance."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        icc = result.icc
        assert 'group' in icc
        assert 0 < icc['group'] < 1


class TestGLMMPoisson:
    """Tests for GLMM with Poisson family."""

    def test_basic_fit(self, glmm_poisson):
        """Poisson GLMM fits and converges."""
        d = glmm_poisson
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='poisson',
        )
        assert result.converged

    def test_family_info(self, glmm_poisson):
        """Family and link recorded correctly."""
        d = glmm_poisson
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='poisson',
        )
        assert result.params.family_name == 'poisson'
        assert result.params.link_name == 'log'

    def test_fixed_effects_direction(self, glmm_poisson):
        """Fixed effects should have correct sign."""
        d = glmm_poisson
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='poisson',
        )
        # True beta1 = 0.5 (positive)
        assert result.coefficients[1] > 0

    def test_fitted_values_positive(self, glmm_poisson):
        """Poisson fitted values (counts) should be positive."""
        d = glmm_poisson
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='poisson',
        )
        assert np.all(result.fitted_values > 0)
