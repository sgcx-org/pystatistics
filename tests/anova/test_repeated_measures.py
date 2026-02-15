"""
Tests for repeated-measures ANOVA.

Validates:
    - Within-subjects factor with 3 conditions
    - 2-condition case (sphericity trivially satisfied, epsilon=1)
    - Mauchly's test values
    - GG/HF corrected p-values
    - Mixed design (between + within)
    - Summary output
"""

import numpy as np
import pytest

from pystatistics.anova import anova_rm


class TestRMWithin3:
    """10 subjects × 3 conditions."""

    def test_table_structure(self, rm_within_3):
        y, subj, cond = rm_within_3
        result = anova_rm(y, subj, within={'condition': cond})
        terms = [row.term for row in result.table]
        assert 'condition' in terms
        assert 'Error' in terms

    def test_degrees_of_freedom(self, rm_within_3):
        y, subj, cond = rm_within_3
        result = anova_rm(y, subj, within={'condition': cond})
        df_dict = {row.term: row.df for row in result.table}
        # k=3 conditions, n=10 subjects
        assert df_dict['condition'] == 2.0    # k - 1
        assert df_dict['Error'] == 18.0       # (n-1)(k-1)

    def test_significant_effect(self, rm_within_3):
        """Condition means at 10, 15, 20 — should be significant."""
        y, subj, cond = rm_within_3
        result = anova_rm(y, subj, within={'condition': cond})
        cond_row = [r for r in result.table if r.term == 'condition'][0]
        assert cond_row.p_value < 0.001

    def test_sphericity_present(self, rm_within_3):
        y, subj, cond = rm_within_3
        result = anova_rm(y, subj, within={'condition': cond})
        assert len(result.sphericity) == 1
        assert result.sphericity[0].factor == 'condition'

    def test_gg_epsilon_range(self, rm_within_3):
        y, subj, cond = rm_within_3
        result = anova_rm(y, subj, within={'condition': cond})
        eps = result.sphericity[0].gg_epsilon
        # GG epsilon in [1/(k-1), 1] = [0.5, 1]
        assert 0.5 <= eps <= 1.0

    def test_hf_epsilon_ge_gg(self, rm_within_3):
        """HF epsilon should be >= GG epsilon."""
        y, subj, cond = rm_within_3
        result = anova_rm(y, subj, within={'condition': cond})
        gg = result.sphericity[0].gg_epsilon
        hf = result.sphericity[0].hf_epsilon
        assert hf >= gg - 1e-10

    def test_corrected_p_values_present(self, rm_within_3):
        y, subj, cond = rm_within_3
        result = anova_rm(y, subj, within={'condition': cond})
        cond_row = [r for r in result.table if r.term == 'condition'][0]
        assert cond_row.gg_p_value is not None
        assert cond_row.hf_p_value is not None

    def test_corrected_p_ge_uncorrected(self, rm_within_3):
        """Corrected p-values should be >= uncorrected (less power)."""
        y, subj, cond = rm_within_3
        result = anova_rm(y, subj, within={'condition': cond})
        cond_row = [r for r in result.table if r.term == 'condition'][0]
        assert cond_row.gg_p_value >= cond_row.p_value - 1e-10

    def test_effect_sizes(self, rm_within_3):
        y, subj, cond = rm_within_3
        result = anova_rm(y, subj, within={'condition': cond})
        assert 'condition' in result.eta_squared
        assert 'condition' in result.partial_eta_squared
        assert 0 < result.eta_squared['condition'] <= 1
        assert 0 < result.partial_eta_squared['condition'] <= 1

    def test_n_subjects(self, rm_within_3):
        y, subj, cond = rm_within_3
        result = anova_rm(y, subj, within={'condition': cond})
        assert result.n_subjects == 10
        assert result.n_obs == 30


class TestRMWithin2:
    """2 conditions — sphericity trivially satisfied."""

    def test_epsilon_equals_1(self, rm_within_2):
        """With 2 conditions, epsilon is always 1."""
        y, subj, cond = rm_within_2
        result = anova_rm(y, subj, within={'condition': cond})
        assert result.sphericity[0].gg_epsilon == 1.0
        assert result.sphericity[0].hf_epsilon == 1.0

    def test_mauchly_w_equals_1(self, rm_within_2):
        y, subj, cond = rm_within_2
        result = anova_rm(y, subj, within={'condition': cond})
        assert result.sphericity[0].mauchly_w == 1.0

    def test_corrected_equals_uncorrected(self, rm_within_2):
        """With epsilon=1, corrected and uncorrected p should match."""
        y, subj, cond = rm_within_2
        result = anova_rm(y, subj, within={'condition': cond})
        cond_row = [r for r in result.table if r.term == 'condition'][0]
        np.testing.assert_allclose(
            cond_row.gg_p_value, cond_row.p_value, rtol=1e-10
        )


class TestRMMixed:
    """Mixed design: between-subjects factor + within-subjects factor."""

    def test_table_has_between_and_within(self, rm_mixed):
        y, subj, within, between = rm_mixed
        result = anova_rm(
            y, subj,
            within={'time': within},
            between={'group': between},
        )
        terms = [row.term for row in result.table]
        assert 'group' in terms
        assert 'time' in terms

    def test_between_subjects_effect(self, rm_mixed):
        """Treatment group should differ from control."""
        y, subj, within, between = rm_mixed
        result = anova_rm(
            y, subj,
            within={'time': within},
            between={'group': between},
        )
        group_row = [r for r in result.table if r.term == 'group'][0]
        assert group_row.p_value < 0.05

    def test_within_subjects_effect(self, rm_mixed):
        """Time effect should be significant."""
        y, subj, within, between = rm_mixed
        result = anova_rm(
            y, subj,
            within={'time': within},
            between={'group': between},
        )
        time_row = [r for r in result.table if r.term == 'time'][0]
        assert time_row.p_value < 0.001


class TestRMSummary:
    """Summary output."""

    def test_summary_contains_key_info(self, rm_within_3):
        y, subj, cond = rm_within_3
        result = anova_rm(y, subj, within={'condition': cond})
        text = result.summary()
        assert 'Repeated-Measures ANOVA' in text
        assert 'condition' in text
        assert 'Mauchly' in text
        assert 'GG eps' in text

    def test_repr(self, rm_within_3):
        y, subj, cond = rm_within_3
        result = anova_rm(y, subj, within={'condition': cond})
        r = repr(result)
        assert 'AnovaRMSolution' in r
        assert 'n_subjects=10' in r


class TestRMCorrection:
    """Correction parameter options."""

    def test_correction_none(self, rm_within_3):
        y, subj, cond = rm_within_3
        result = anova_rm(y, subj, within={'condition': cond}, correction='none')
        assert result.correction == 'none'

    def test_correction_gg(self, rm_within_3):
        y, subj, cond = rm_within_3
        result = anova_rm(y, subj, within={'condition': cond}, correction='gg')
        assert result.correction == 'gg'

    def test_correction_hf(self, rm_within_3):
        y, subj, cond = rm_within_3
        result = anova_rm(y, subj, within={'condition': cond}, correction='hf')
        assert result.correction == 'hf'

    def test_invalid_correction_raises(self, rm_within_3):
        y, subj, cond = rm_within_3
        with pytest.raises(ValueError, match="correction"):
            anova_rm(y, subj, within={'condition': cond}, correction='invalid')
