"""
Tests for Levene's test / Brown-Forsythe test.

Validates:
    - Equal variances → high p-value
    - Unequal variances → low p-value
    - Both center options (mean, median)
    - Degrees of freedom
    - Group variances in output
"""

import numpy as np
import pytest

from pystatistics.anova import levene_test


class TestLeveneEqualVariances:
    """Groups with equal variance → non-significant."""

    def test_high_p_value(self):
        rng = np.random.default_rng(42)
        y = np.concatenate([
            rng.normal(10, 2, 30),
            rng.normal(15, 2, 30),
            rng.normal(20, 2, 30),
        ])
        group = np.array(['A'] * 30 + ['B'] * 30 + ['C'] * 30)
        result = levene_test(y, group)
        assert result.p_value > 0.05

    def test_degrees_of_freedom(self):
        rng = np.random.default_rng(42)
        y = np.concatenate([rng.normal(0, 1, 20), rng.normal(0, 1, 20)])
        group = np.array(['A'] * 20 + ['B'] * 20)
        result = levene_test(y, group)
        assert result.df_between == 1   # k - 1 = 2 - 1
        assert result.df_within == 38   # n - k = 40 - 2


class TestLeveneUnequalVariances:
    """Groups with very different variances → significant."""

    def test_low_p_value(self):
        rng = np.random.default_rng(42)
        y = np.concatenate([
            rng.normal(10, 1, 30),    # sd = 1
            rng.normal(10, 5, 30),    # sd = 5
            rng.normal(10, 10, 30),   # sd = 10
        ])
        group = np.array(['A'] * 30 + ['B'] * 30 + ['C'] * 30)
        result = levene_test(y, group)
        assert result.p_value < 0.01


class TestLeveneCenterOptions:
    """center='mean' vs center='median'."""

    def test_median_default(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        group = np.array(['A', 'A', 'B', 'B'])
        result = levene_test(y, group)
        assert result.center == 'median'

    def test_mean_center(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        group = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
        result = levene_test(y, group, center='mean')
        assert result.center == 'mean'

    def test_invalid_center_raises(self):
        y = np.array([1.0, 2.0, 3.0])
        group = np.array(['A', 'B', 'C'])
        with pytest.raises(ValueError, match="center"):
            levene_test(y, group, center='mode')


class TestLeveneOutput:
    """Result structure and output."""

    def test_group_vars_present(self):
        rng = np.random.default_rng(42)
        y = np.concatenate([rng.normal(0, 1, 20), rng.normal(0, 3, 20)])
        group = np.array(['A'] * 20 + ['B'] * 20)
        result = levene_test(y, group)
        assert 'A' in result.group_vars
        assert 'B' in result.group_vars

    def test_f_value_positive(self):
        rng = np.random.default_rng(42)
        y = np.concatenate([rng.normal(0, 1, 20), rng.normal(0, 5, 20)])
        group = np.array(['A'] * 20 + ['B'] * 20)
        result = levene_test(y, group)
        assert result.f_value >= 0

    def test_summary_output(self):
        rng = np.random.default_rng(42)
        y = np.concatenate([rng.normal(0, 1, 20), rng.normal(0, 1, 20)])
        group = np.array(['A'] * 20 + ['B'] * 20)
        result = levene_test(y, group)
        text = result.summary()
        assert 'Brown-Forsythe' in text  # default center='median'

    def test_summary_levene_variant(self):
        rng = np.random.default_rng(42)
        y = np.concatenate([rng.normal(0, 1, 20), rng.normal(0, 1, 20)])
        group = np.array(['A'] * 20 + ['B'] * 20)
        result = levene_test(y, group, center='mean')
        text = result.summary()
        assert 'Levene' in text

    def test_repr(self):
        rng = np.random.default_rng(42)
        y = np.concatenate([rng.normal(0, 1, 10), rng.normal(0, 1, 10)])
        group = np.array(['A'] * 10 + ['B'] * 10)
        result = levene_test(y, group)
        r = repr(result)
        assert 'LeveneSolution' in r
