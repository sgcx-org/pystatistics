"""
Tests for one-way ANOVA.

Validates:
    - Table structure (terms, df, SS, MS, F, p)
    - 2-group matches F-test / t-test equivalence
    - 3-group balanced and unbalanced
    - Equal-means case (F ≈ 0)
    - Effect sizes (eta-squared, partial eta-squared)
    - Summary output
"""

import numpy as np
import pytest
from scipy import stats as sp_stats

from pystatistics.anova import anova_oneway


class TestOneWayBalanced:
    """3-group balanced design with clear differences."""

    def test_table_structure(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        table = result.table

        # Should have 2 rows: group + Residuals
        assert len(table) == 2
        assert table[0].term == 'group'
        assert table[1].term == 'Residuals'

    def test_degrees_of_freedom(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)

        # k=3 groups, n=30 total
        assert result.table[0].df == 2    # k - 1
        assert result.table[1].df == 27   # n - k

    def test_significant_effect(self, oneway_balanced):
        """Groups have means 10, 15, 20 — should be highly significant."""
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        assert result.table[0].p_value < 0.001

    def test_f_value_positive(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        assert result.table[0].f_value > 0

    def test_ss_partition(self, oneway_balanced):
        """SS_group + SS_residual = SS_total."""
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        ss_group = result.table[0].sum_sq
        ss_resid = result.table[1].sum_sq
        ss_total = np.sum((y - np.mean(y)) ** 2)
        np.testing.assert_allclose(ss_group + ss_resid, ss_total, rtol=1e-10)

    def test_mean_sq_computation(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        for row in result.table:
            np.testing.assert_allclose(
                row.mean_sq, row.sum_sq / row.df, rtol=1e-12
            )

    def test_f_is_ms_ratio(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        expected_f = result.table[0].mean_sq / result.table[1].mean_sq
        np.testing.assert_allclose(result.table[0].f_value, expected_f, rtol=1e-12)

    def test_residuals_no_f(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        assert result.table[1].f_value is None
        assert result.table[1].p_value is None


class TestOneWayUnbalanced:
    """3-group unbalanced design (n=5, 10, 15)."""

    def test_degrees_of_freedom(self, oneway_unbalanced):
        y, group = oneway_unbalanced
        result = anova_oneway(y, group)
        assert result.table[0].df == 2    # k - 1
        assert result.table[1].df == 27   # 30 - 3

    def test_significant_effect(self, oneway_unbalanced):
        y, group = oneway_unbalanced
        result = anova_oneway(y, group)
        assert result.table[0].p_value < 0.001

    def test_ss_partition(self, oneway_unbalanced):
        y, group = oneway_unbalanced
        result = anova_oneway(y, group)
        ss_group = result.table[0].sum_sq
        ss_resid = result.table[1].sum_sq
        ss_total = np.sum((y - np.mean(y)) ** 2)
        np.testing.assert_allclose(ss_group + ss_resid, ss_total, rtol=1e-10)


class TestOneWayTwoGroups:
    """2-group ANOVA should match F-test from t-test."""

    def test_f_equals_t_squared(self, oneway_two_groups):
        """For 2 groups: F = t^2."""
        y, group = oneway_two_groups
        result = anova_oneway(y, group)

        # Independent t-test
        mask_ctrl = group == 'control'
        t_stat, t_p = sp_stats.ttest_ind(y[mask_ctrl], y[~mask_ctrl])

        np.testing.assert_allclose(
            result.table[0].f_value, t_stat ** 2, rtol=1e-8
        )
        np.testing.assert_allclose(
            result.table[0].p_value, t_p, rtol=1e-8
        )


class TestOneWayNoEffect:
    """All groups have same population mean — F should be small."""

    def test_not_significant(self, oneway_no_effect):
        y, group = oneway_no_effect
        result = anova_oneway(y, group)
        # Should NOT be significant (p > 0.05 with high probability)
        # We use a fixed seed so this is deterministic
        assert result.table[0].p_value > 0.01


class TestOneWayEffectSizes:
    """eta-squared and partial eta-squared computation."""

    def test_eta_squared_between_0_and_1(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        for term, eta in result.eta_squared.items():
            assert 0 <= eta <= 1

    def test_partial_eta_squared_between_0_and_1(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        for term, peta in result.partial_eta_squared.items():
            assert 0 <= peta <= 1

    def test_oneway_eta_equals_partial_eta(self, oneway_balanced):
        """For one-way ANOVA, eta^2 == partial eta^2."""
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        for term in result.eta_squared:
            np.testing.assert_allclose(
                result.eta_squared[term],
                result.partial_eta_squared[term],
                rtol=1e-10,
            )

    def test_eta_squared_formula(self, oneway_balanced):
        """eta^2 = SS_effect / SS_total."""
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        ss_total = sum(row.sum_sq for row in result.table)
        for term in result.eta_squared:
            row = [r for r in result.table if r.term == term][0]
            expected = row.sum_sq / ss_total
            np.testing.assert_allclose(result.eta_squared[term], expected, rtol=1e-10)


class TestOneWayMetadata:
    """Metadata and summary output."""

    def test_n_obs(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        assert result.n_obs == 30

    def test_grand_mean(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        np.testing.assert_allclose(result.grand_mean, np.mean(y), rtol=1e-10)

    def test_group_means(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        means = result.group_means['group']
        assert set(means.keys()) == {'A', 'B', 'C'}

    def test_summary_contains_key_info(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        text = result.summary()
        assert 'Analysis of Variance' in text
        assert 'group' in text
        assert 'Residuals' in text
        assert 'Pr(>F)' in text

    def test_repr(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        r = repr(result)
        assert 'AnovaSolution' in r
        assert 'n=30' in r


class TestOneWaySStypes:
    """For one-way ANOVA, all SS types should give identical results."""

    def test_type1_equals_type2(self, oneway_balanced):
        y, group = oneway_balanced
        r1 = anova_oneway(y, group, ss_type=1)
        r2 = anova_oneway(y, group, ss_type=2)
        np.testing.assert_allclose(
            r1.table[0].sum_sq, r2.table[0].sum_sq, rtol=1e-8
        )
        np.testing.assert_allclose(
            r1.table[0].f_value, r2.table[0].f_value, rtol=1e-8
        )

    def test_type1_equals_type3(self, oneway_balanced):
        y, group = oneway_balanced
        r1 = anova_oneway(y, group, ss_type=1)
        r3 = anova_oneway(y, group, ss_type=3)
        np.testing.assert_allclose(
            r1.table[0].sum_sq, r3.table[0].sum_sq, rtol=1e-8
        )

    def test_type2_equals_type3(self, oneway_unbalanced):
        """Even for unbalanced, one-way gives same results for all types."""
        y, group = oneway_unbalanced
        r2 = anova_oneway(y, group, ss_type=2)
        r3 = anova_oneway(y, group, ss_type=3)
        np.testing.assert_allclose(
            r2.table[0].sum_sq, r3.table[0].sum_sq, rtol=1e-8
        )
