"""
Tests for AnovaDesign factory validation.

Validates:
    - for_oneway: input validation, group requirements
    - for_factorial: multi-factor validation
    - for_repeated_measures: subject/within validation
"""

import numpy as np
import pytest

from pystatistics.core.exceptions import ValidationError, DimensionError
from pystatistics.anova.design import AnovaDesign


# ═══════════════════════════════════════════════════════════════════════
# for_oneway
# ═══════════════════════════════════════════════════════════════════════


class TestForOneway:
    """AnovaDesign.for_oneway() validates inputs."""

    def test_basic_creation(self, oneway_balanced):
        y, group = oneway_balanced
        design = AnovaDesign.for_oneway(y, group)
        assert design.design_type == 'oneway'
        assert design.n == 30
        assert 'group' in design.factors

    def test_rejects_single_group(self):
        y = np.array([1.0, 2.0, 3.0])
        group = np.array(['A', 'A', 'A'])
        with pytest.raises(ValidationError, match="at least 2 groups"):
            AnovaDesign.for_oneway(y, group)

    def test_rejects_mismatched_lengths(self):
        y = np.array([1.0, 2.0, 3.0])
        group = np.array(['A', 'B'])
        with pytest.raises(DimensionError):
            AnovaDesign.for_oneway(y, group)

    def test_rejects_non_numeric_y(self):
        with pytest.raises(ValidationError):
            AnovaDesign.for_oneway(['a', 'b', 'c'], ['A', 'B', 'C'])

    def test_rejects_2d_y(self):
        y = np.array([[1.0, 2.0], [3.0, 4.0]])
        group = np.array(['A', 'B'])
        with pytest.raises(DimensionError):
            AnovaDesign.for_oneway(y, group)

    def test_rejects_nan_y(self):
        y = np.array([1.0, np.nan, 3.0])
        group = np.array(['A', 'B', 'C'])
        with pytest.raises(ValidationError):
            AnovaDesign.for_oneway(y, group)

    def test_integer_groups(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        group = np.array([1, 1, 2, 2])
        design = AnovaDesign.for_oneway(y, group)
        assert design.n == 4

    def test_converts_to_string_groups(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        group = np.array([1, 1, 2, 2])
        design = AnovaDesign.for_oneway(y, group)
        # Groups should be stored as strings
        assert all(isinstance(v, (str, np.str_)) for v in design.factors['group'])


# ═══════════════════════════════════════════════════════════════════════
# for_factorial
# ═══════════════════════════════════════════════════════════════════════


class TestForFactorial:
    """AnovaDesign.for_factorial() validates multi-factor inputs."""

    def test_basic_creation(self, twoway_balanced):
        y, a, b = twoway_balanced
        design = AnovaDesign.for_factorial(y, {'A': a, 'B': b})
        assert design.design_type == 'factorial'
        assert 'A' in design.factors
        assert 'B' in design.factors

    def test_rejects_single_level_factor(self):
        y = np.array([1.0, 2.0, 3.0])
        factors = {'group': np.array(['A', 'A', 'A'])}
        with pytest.raises(ValidationError, match="at least 2 levels"):
            AnovaDesign.for_factorial(y, factors)

    def test_rejects_mismatched_factor_length(self):
        y = np.array([1.0, 2.0, 3.0])
        factors = {'group': np.array(['A', 'B'])}
        with pytest.raises(ValidationError, match="doesn't match"):
            AnovaDesign.for_factorial(y, factors)

    def test_with_covariate(self, ancova_data):
        y, group, age = ancova_data
        design = AnovaDesign.for_factorial(
            y, {'group': group}, covariates={'age': age}
        )
        assert design.covariates is not None
        assert 'age' in design.covariates

    def test_rejects_nan_covariate(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        factors = {'g': np.array(['A', 'A', 'B', 'B'])}
        cov = np.array([1.0, np.nan, 3.0, 4.0])
        with pytest.raises(ValidationError):
            AnovaDesign.for_factorial(y, factors, covariates={'x': cov})


# ═══════════════════════════════════════════════════════════════════════
# for_repeated_measures
# ═══════════════════════════════════════════════════════════════════════


class TestForRepeatedMeasures:
    """AnovaDesign.for_repeated_measures() validates RM inputs."""

    def test_basic_creation(self, rm_within_3):
        y, subj, cond = rm_within_3
        design = AnovaDesign.for_repeated_measures(
            y, subj, within={'condition': cond}
        )
        assert design.design_type == 'rm'
        assert design.subject is not None

    def test_rejects_mismatched_subject_length(self):
        y = np.array([1.0, 2.0, 3.0])
        subj = np.array(['S1', 'S2'])
        cond = np.array(['A', 'B', 'A'])
        with pytest.raises(ValidationError, match="subject"):
            AnovaDesign.for_repeated_measures(y, subj, within={'cond': cond})

    def test_rejects_mismatched_within_length(self):
        y = np.array([1.0, 2.0, 3.0])
        subj = np.array(['S1', 'S1', 'S2'])
        cond = np.array(['A', 'B'])
        with pytest.raises(ValidationError):
            AnovaDesign.for_repeated_measures(y, subj, within={'cond': cond})

    def test_mixed_design(self, rm_mixed):
        y, subj, within, between = rm_mixed
        design = AnovaDesign.for_repeated_measures(
            y, subj, within={'time': within}, between={'group': between}
        )
        assert 'time' in design.factors
        assert 'group' in design.factors
