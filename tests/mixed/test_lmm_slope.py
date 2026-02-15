"""Tests for LMM with random intercept + slope."""

import numpy as np
import pytest

from pystatistics.mixed import lmm


class TestLMMRandomSlope:
    """Tests for random intercept + slope models."""

    def test_basic_fit(self, sleepstudy_like):
        """LMM with random slope fits and converges."""
        d = sleepstudy_like
        result = lmm(
            d['y'], d['X'],
            groups={'subject': d['subject']},
            random_effects={'subject': ['1', 'days']},
            random_data={'days': d['days']},
        )
        assert result.converged
        assert len(result.coefficients) == 2

    def test_fixed_effects_recovery(self, sleepstudy_like):
        """Fixed effects approximately recover true parameters."""
        d = sleepstudy_like
        result = lmm(
            d['y'], d['X'],
            groups={'subject': d['subject']},
            random_effects={'subject': ['1', 'days']},
            random_data={'days': d['days']},
        )
        # True: intercept=250, slope=10
        np.testing.assert_allclose(result.coefficients[0], 250.0, atol=20.0)
        np.testing.assert_allclose(result.coefficients[1], 10.0, atol=5.0)

    def test_two_variance_components(self, sleepstudy_like):
        """Model should have 2 variance components (intercept + slope)."""
        d = sleepstudy_like
        result = lmm(
            d['y'], d['X'],
            groups={'subject': d['subject']},
            random_effects={'subject': ['1', 'days']},
            random_data={'days': d['days']},
        )
        vc = result.var_components
        assert len(vc) == 2
        assert vc[0].name == '(Intercept)'
        assert vc[1].name == 'days'
        assert vc[0].variance > 0
        assert vc[1].variance > 0

    def test_correlation_estimated(self, sleepstudy_like):
        """Slope component should have a correlation with intercept."""
        d = sleepstudy_like
        result = lmm(
            d['y'], d['X'],
            groups={'subject': d['subject']},
            random_effects={'subject': ['1', 'days']},
            random_data={'days': d['days']},
        )
        vc = result.var_components
        # First term has no correlation (it's the baseline)
        assert vc[0].corr is None
        # Second term should have correlation with first
        assert vc[1].corr is not None
        assert -1.0 <= vc[1].corr <= 1.0

    def test_blups_shape(self, sleepstudy_like):
        """BLUPs should be (n_subjects, 2) for intercept + slope."""
        d = sleepstudy_like
        result = lmm(
            d['y'], d['X'],
            groups={'subject': d['subject']},
            random_effects={'subject': ['1', 'days']},
            random_data={'days': d['days']},
        )
        ranef = result.ranef['subject']
        assert ranef.shape == (d['n_subjects'], 2)

    def test_theta_length(self, sleepstudy_like):
        """θ should have 3 elements for 2×2 Cholesky factor."""
        d = sleepstudy_like
        result = lmm(
            d['y'], d['X'],
            groups={'subject': d['subject']},
            random_effects={'subject': ['1', 'days']},
            random_data={'days': d['days']},
        )
        assert len(result.params.theta) == 3

    def test_summary_shows_correlation(self, sleepstudy_like):
        """Summary should include correlation in random effects table."""
        d = sleepstudy_like
        result = lmm(
            d['y'], d['X'],
            groups={'subject': d['subject']},
            random_effects={'subject': ['1', 'days']},
            random_data={'days': d['days']},
        )
        s = result.summary()
        assert 'days' in s
        assert '(Intercept)' in s
