"""Tests for LMM with crossed random effects."""

import numpy as np
import pytest

from pystatistics.mixed import lmm


class TestLMMCrossed:
    """Tests for crossed random effects: (1|subject) + (1|item)."""

    def test_basic_fit(self, crossed_effects):
        """Crossed effects model fits and converges."""
        d = crossed_effects
        result = lmm(
            d['y'], d['X'],
            groups={'subject': d['subject'], 'item': d['item']},
        )
        assert result.converged

    def test_two_grouping_factors(self, crossed_effects):
        """Should have 2 variance components (subject + item)."""
        d = crossed_effects
        result = lmm(
            d['y'], d['X'],
            groups={'subject': d['subject'], 'item': d['item']},
        )
        vc = result.var_components
        assert len(vc) == 2
        groups_in_vc = {v.group for v in vc}
        assert groups_in_vc == {'subject', 'item'}

    def test_both_variances_positive(self, crossed_effects):
        """Both subject and item variances should be positive."""
        d = crossed_effects
        result = lmm(
            d['y'], d['X'],
            groups={'subject': d['subject'], 'item': d['item']},
        )
        for vc in result.var_components:
            assert vc.variance > 0

    def test_n_groups_dict(self, crossed_effects):
        """n_groups should report correct counts for both factors."""
        d = crossed_effects
        result = lmm(
            d['y'], d['X'],
            groups={'subject': d['subject'], 'item': d['item']},
        )
        assert result.params.n_groups['subject'] == d['n_subjects']
        assert result.params.n_groups['item'] == d['n_items']

    def test_blups_for_both_factors(self, crossed_effects):
        """BLUPs should exist for both subject and item."""
        d = crossed_effects
        result = lmm(
            d['y'], d['X'],
            groups={'subject': d['subject'], 'item': d['item']},
        )
        assert 'subject' in result.ranef
        assert 'item' in result.ranef
        assert result.ranef['subject'].shape == (d['n_subjects'], 1)
        assert result.ranef['item'].shape == (d['n_items'], 1)

    def test_theta_has_two_elements(self, crossed_effects):
        """Two random intercepts → θ has 2 elements."""
        d = crossed_effects
        result = lmm(
            d['y'], d['X'],
            groups={'subject': d['subject'], 'item': d['item']},
        )
        assert len(result.params.theta) == 2

    def test_fixed_effects_recovery(self, crossed_effects):
        """Fixed effects should be close to true values."""
        d = crossed_effects
        result = lmm(
            d['y'], d['X'],
            groups={'subject': d['subject'], 'item': d['item']},
        )
        # True: beta0=3, beta1=1.5
        np.testing.assert_allclose(result.coefficients[0], d['beta0'], atol=2.0)
        np.testing.assert_allclose(result.coefficients[1], d['beta1'], atol=0.5)
