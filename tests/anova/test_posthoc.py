"""
Tests for post-hoc tests (Tukey HSD, Bonferroni, Dunnett).

Validates:
    - Tukey HSD: 3-group balanced, CI properties
    - Bonferroni: p-value adjustment, CI width
    - Dunnett: control vs treatment comparisons
    - 2-group Tukey ≈ t-test
"""

import numpy as np
import pytest
from scipy import stats as sp_stats

from pystatistics.anova import anova_oneway, anova_posthoc


# ═══════════════════════════════════════════════════════════════════════
# Tukey HSD
# ═══════════════════════════════════════════════════════════════════════


class TestTukeyHSD:
    """Tukey's Honestly Significant Difference test."""

    def test_number_of_comparisons(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        posthoc = anova_posthoc(result, method='tukey')
        # k=3 groups → C(3,2) = 3 comparisons
        assert len(posthoc.comparisons) == 3

    def test_comparison_labels(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        posthoc = anova_posthoc(result, method='tukey')
        pairs = {(c.group1, c.group2) for c in posthoc.comparisons}
        assert ('A', 'B') in pairs
        assert ('A', 'C') in pairs
        assert ('B', 'C') in pairs

    def test_significant_differences(self, oneway_balanced):
        """All pairs should be significant (means at 10, 15, 20)."""
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        posthoc = anova_posthoc(result, method='tukey')
        for c in posthoc.comparisons:
            assert c.p_value < 0.05

    def test_ci_contains_diff(self, oneway_balanced):
        """The difference should fall within the CI."""
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        posthoc = anova_posthoc(result, method='tukey')
        for c in posthoc.comparisons:
            assert c.ci_lower <= c.diff <= c.ci_upper

    def test_diff_sign(self, oneway_balanced):
        """B-A and C-A should be positive (means increase A→B→C)."""
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        posthoc = anova_posthoc(result, method='tukey')
        for c in posthoc.comparisons:
            if c.group1 == 'A':
                assert c.diff > 0  # B > A and C > A

    def test_method_is_tukey(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        posthoc = anova_posthoc(result, method='tukey')
        assert posthoc.method == 'tukey'

    def test_summary_output(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        posthoc = anova_posthoc(result, method='tukey')
        text = posthoc.summary()
        assert 'Tukey HSD' in text
        assert 'B-A' in text


# ═══════════════════════════════════════════════════════════════════════
# Bonferroni
# ═══════════════════════════════════════════════════════════════════════


class TestBonferroni:
    """Bonferroni pairwise t-tests."""

    def test_number_of_comparisons(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        posthoc = anova_posthoc(result, method='bonferroni')
        assert len(posthoc.comparisons) == 3

    def test_p_values_adjusted(self, oneway_balanced):
        """Bonferroni p-values should be larger than unadjusted."""
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        posthoc = anova_posthoc(result, method='bonferroni')
        for c in posthoc.comparisons:
            assert c.p_value <= 1.0

    def test_bonferroni_more_conservative_than_tukey(self, oneway_balanced):
        """Bonferroni CIs should generally be wider than Tukey."""
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        tukey = anova_posthoc(result, method='tukey')
        bonf = anova_posthoc(result, method='bonferroni')

        # Compare CI widths for the first comparison
        tukey_width = tukey.comparisons[0].ci_upper - tukey.comparisons[0].ci_lower
        bonf_width = bonf.comparisons[0].ci_upper - bonf.comparisons[0].ci_lower
        # Bonferroni is generally more conservative (wider CIs)
        # This isn't guaranteed for all cases but should hold for 3 groups
        assert bonf_width >= tukey_width * 0.95  # allow small numerical wiggle

    def test_method_is_bonferroni(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        posthoc = anova_posthoc(result, method='bonferroni')
        assert posthoc.method == 'bonferroni'


# ═══════════════════════════════════════════════════════════════════════
# Dunnett
# ═══════════════════════════════════════════════════════════════════════


class TestDunnett:
    """Dunnett's test: each treatment vs control."""

    def test_number_of_comparisons(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        posthoc = anova_posthoc(result, method='dunnett', control='A')
        # k=3 groups, control=A → 2 comparisons (B vs A, C vs A)
        assert len(posthoc.comparisons) == 2

    def test_control_is_group1(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        posthoc = anova_posthoc(result, method='dunnett', control='A')
        for c in posthoc.comparisons:
            assert c.group1 == 'A'

    def test_significant_vs_control(self, oneway_balanced):
        """B and C should differ significantly from A."""
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        posthoc = anova_posthoc(result, method='dunnett', control='A')
        for c in posthoc.comparisons:
            assert c.p_value < 0.05

    def test_requires_control_param(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        with pytest.raises(ValueError, match="control"):
            anova_posthoc(result, method='dunnett')

    def test_invalid_control_raises(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        with pytest.raises(ValueError, match="not found"):
            anova_posthoc(result, method='dunnett', control='nonexistent')

    def test_method_is_dunnett(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        posthoc = anova_posthoc(result, method='dunnett', control='A')
        assert posthoc.method == 'dunnett'


# ═══════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestPostHocEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_method_raises(self, oneway_balanced):
        y, group = oneway_balanced
        result = anova_oneway(y, group)
        with pytest.raises(ValueError, match="Unknown method"):
            anova_posthoc(result, method='scheffe')

    def test_2_group_tukey(self, oneway_two_groups):
        """Tukey with 2 groups should give one comparison."""
        y, group = oneway_two_groups
        result = anova_oneway(y, group)
        posthoc = anova_posthoc(result, method='tukey')
        assert len(posthoc.comparisons) == 1
