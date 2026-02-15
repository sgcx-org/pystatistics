"""Tests for LMM with nested random effects."""

import numpy as np
import pytest

from pystatistics.mixed import lmm


class TestLMMNested:
    """Tests for nested random effects: (1|classroom) + (1|student)."""

    def test_basic_fit(self, nested_effects):
        """Nested effects model fits and converges."""
        d = nested_effects
        result = lmm(
            d['y'], d['X'],
            groups={
                'classroom': d['classroom'],
                'student': d['student'],
            },
        )
        assert result.converged

    def test_two_variance_components(self, nested_effects):
        """Should have 2 variance components."""
        d = nested_effects
        result = lmm(
            d['y'], d['X'],
            groups={
                'classroom': d['classroom'],
                'student': d['student'],
            },
        )
        vc = result.var_components
        assert len(vc) == 2

    def test_both_variances_estimated(self, nested_effects):
        """Both classroom and student variance should be non-negative."""
        d = nested_effects
        result = lmm(
            d['y'], d['X'],
            groups={
                'classroom': d['classroom'],
                'student': d['student'],
            },
        )
        for vc in result.var_components:
            assert vc.variance >= 0

    def test_fixed_effects_recovery(self, nested_effects):
        """Fixed effects should be approximately correct."""
        d = nested_effects
        result = lmm(
            d['y'], d['X'],
            groups={
                'classroom': d['classroom'],
                'student': d['student'],
            },
        )
        np.testing.assert_allclose(result.coefficients[0], d['beta0'], atol=3.0)
        np.testing.assert_allclose(result.coefficients[1], d['beta1'], atol=1.0)

    def test_n_groups_correct(self, nested_effects):
        """n_groups should match true structure."""
        d = nested_effects
        result = lmm(
            d['y'], d['X'],
            groups={
                'classroom': d['classroom'],
                'student': d['student'],
            },
        )
        assert result.params.n_groups['classroom'] == d['n_classrooms']
        assert result.params.n_groups['student'] == d['n_students']
