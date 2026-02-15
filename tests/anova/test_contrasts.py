"""
Tests for contrast coding and model matrix construction.

Validates:
    - Treatment (dummy) coding shapes, baseline selection, indicator values
    - Deviation (sum-to-zero) coding properties
    - Interaction column computation
    - Full model matrix: term slices, shapes, metadata
"""

import numpy as np
import pytest

from pystatistics.anova._contrasts import (
    encode_treatment,
    encode_deviation,
    interaction_columns,
    build_model_matrix,
    ModelMatrix,
)


# ═══════════════════════════════════════════════════════════════════════
# encode_treatment
# ═══════════════════════════════════════════════════════════════════════


class TestEncodeTreatment:
    """Treatment coding produces (n, k-1) indicators dropping first level."""

    def test_shape_3_levels(self):
        factor = np.array(['A', 'B', 'C', 'A', 'B', 'C'])
        X, levels, baseline = encode_treatment(factor)
        assert X.shape == (6, 2)
        assert baseline == 'A'
        assert levels == ['B', 'C']

    def test_shape_2_levels(self):
        factor = np.array(['low', 'high', 'low', 'high'])
        X, levels, baseline = encode_treatment(factor)
        assert X.shape == (4, 1)
        assert baseline == 'high'   # sorted first
        assert levels == ['low']

    def test_indicator_values(self):
        factor = np.array(['A', 'B', 'C', 'A', 'B', 'C'])
        X, _, _ = encode_treatment(factor)
        # Column 0 = indicator for B, Column 1 = indicator for C
        expected_B = np.array([0, 1, 0, 0, 1, 0], dtype=np.float64)
        expected_C = np.array([0, 0, 1, 0, 0, 1], dtype=np.float64)
        np.testing.assert_array_equal(X[:, 0], expected_B)
        np.testing.assert_array_equal(X[:, 1], expected_C)

    def test_baseline_is_first_sorted_level(self):
        factor = np.array(['Z', 'A', 'M'])
        _, _, baseline = encode_treatment(factor)
        assert baseline == 'A'

    def test_integer_labels(self):
        factor = np.array([1, 2, 3, 1, 2])
        X, levels, baseline = encode_treatment(factor)
        assert X.shape == (5, 2)
        assert baseline == '1'

    def test_float_dtype(self):
        factor = np.array(['A', 'B', 'A'])
        X, _, _ = encode_treatment(factor)
        assert X.dtype == np.float64


# ═══════════════════════════════════════════════════════════════════════
# encode_deviation
# ═══════════════════════════════════════════════════════════════════════


class TestEncodeDeviation:
    """Deviation coding sums to zero across levels."""

    def test_shape_3_levels(self):
        factor = np.array(['A', 'B', 'C', 'A', 'B', 'C'])
        X, levels = encode_deviation(factor)
        assert X.shape == (6, 2)
        assert levels == ['A', 'B']  # last level (C) is reference

    def test_sums_to_zero(self):
        """Each column sums to zero when all levels equally represented."""
        factor = np.array(['A', 'B', 'C', 'A', 'B', 'C'])
        X, _ = encode_deviation(factor)
        np.testing.assert_allclose(X.sum(axis=0), 0.0, atol=1e-10)

    def test_reference_gets_minus_one(self):
        """Last level (reference) gets -1 in all columns."""
        factor = np.array(['A', 'B', 'C'])
        X, _ = encode_deviation(factor)
        # C is reference (last sorted level)
        assert X[2, 0] == -1.0
        assert X[2, 1] == -1.0

    def test_own_level_gets_plus_one(self):
        factor = np.array(['A', 'B', 'C'])
        X, _ = encode_deviation(factor)
        # A gets +1 in column 0, B gets +1 in column 1
        assert X[0, 0] == 1.0
        assert X[1, 1] == 1.0

    def test_2_levels(self):
        factor = np.array(['X', 'Y', 'X', 'Y'])
        X, levels = encode_deviation(factor)
        assert X.shape == (4, 1)
        # X gets +1, Y gets -1
        np.testing.assert_array_equal(X[:, 0], [1, -1, 1, -1])


# ═══════════════════════════════════════════════════════════════════════
# interaction_columns
# ═══════════════════════════════════════════════════════════════════════


class TestInteractionColumns:
    """Interaction = element-wise product of all column pairs."""

    def test_shape(self):
        X_a = np.ones((6, 2))
        X_b = np.ones((6, 3))
        X_int = interaction_columns(X_a, X_b)
        assert X_int.shape == (6, 6)  # 2 * 3 = 6

    def test_values(self):
        """Product of specific indicator columns."""
        n = 4
        X_a = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float64)
        X_b = np.array([[1], [1], [0], [0]], dtype=np.float64)
        X_int = interaction_columns(X_a, X_b)
        assert X_int.shape == (4, 2)
        np.testing.assert_array_equal(X_int[:, 0], [1, 0, 0, 0])
        np.testing.assert_array_equal(X_int[:, 1], [0, 1, 0, 0])

    def test_single_column_each(self):
        X_a = np.array([[1], [0], [1]], dtype=np.float64)
        X_b = np.array([[0], [1], [1]], dtype=np.float64)
        X_int = interaction_columns(X_a, X_b)
        assert X_int.shape == (3, 1)
        np.testing.assert_array_equal(X_int[:, 0], [0, 0, 1])


# ═══════════════════════════════════════════════════════════════════════
# build_model_matrix
# ═══════════════════════════════════════════════════════════════════════


class TestBuildModelMatrix:
    """Full model matrix construction with metadata."""

    def test_oneway_treatment_shape(self):
        factors = {'group': np.array(['A', 'B', 'C'] * 5)}
        mm = build_model_matrix(factors, coding='treatment')
        # intercept(1) + group(2) = 3 columns
        assert mm.X.shape == (15, 3)
        assert mm.p == 3
        assert mm.n == 15

    def test_oneway_deviation_shape(self):
        factors = {'group': np.array(['A', 'B', 'C'] * 5)}
        mm = build_model_matrix(factors, coding='deviation')
        assert mm.X.shape == (15, 3)

    def test_term_slices(self):
        factors = {'group': np.array(['A', 'B', 'C'] * 5)}
        mm = build_model_matrix(factors, coding='treatment')
        assert 'Intercept' in mm.term_slices
        assert 'group' in mm.term_slices
        assert mm.term_df['Intercept'] == 1
        assert mm.term_df['group'] == 2

    def test_intercept_column(self):
        factors = {'group': np.array(['A', 'B'] * 5)}
        mm = build_model_matrix(factors, coding='treatment', include_intercept=True)
        np.testing.assert_array_equal(mm.X[:, 0], 1.0)

    def test_no_intercept(self):
        factors = {'group': np.array(['A', 'B'] * 5)}
        mm = build_model_matrix(factors, coding='treatment', include_intercept=False)
        assert 'Intercept' not in mm.term_slices
        # Just group columns (1 for 2-level treatment)
        assert mm.X.shape == (10, 1)

    def test_two_factors_with_interaction(self):
        factors = {
            'A': np.array(['a1', 'a2'] * 6),
            'B': np.array(['b1', 'b1', 'b2', 'b2', 'b3', 'b3'] * 2),
        }
        mm = build_model_matrix(factors, coding='treatment')
        # intercept(1) + A(1) + B(2) + A:B(2) = 6
        assert mm.p == 6
        assert 'A:B' in mm.term_slices
        assert mm.term_df['A:B'] == 2  # 1 * 2 = 2

    def test_no_interactions_when_empty_list(self):
        factors = {
            'A': np.array(['a1', 'a2'] * 6),
            'B': np.array(['b1', 'b2', 'b3'] * 4),
        }
        mm = build_model_matrix(factors, coding='treatment', interactions=[])
        assert 'A:B' not in mm.term_slices

    def test_covariate(self):
        factors = {'group': np.array(['A', 'B'] * 5)}
        covariates = {'age': np.arange(10, dtype=np.float64)}
        mm = build_model_matrix(factors, covariates=covariates, coding='treatment')
        assert 'age' in mm.term_slices
        assert mm.term_df['age'] == 1

    def test_factor_levels_stored(self):
        factors = {'color': np.array(['red', 'blue', 'green'] * 3)}
        mm = build_model_matrix(factors, coding='treatment')
        assert mm.factor_levels == {'color': ['blue', 'green', 'red']}

    def test_coding_stored(self):
        factors = {'g': np.array(['X', 'Y'] * 5)}
        mm = build_model_matrix(factors, coding='deviation')
        assert mm.coding == 'deviation'

    def test_invalid_coding_raises(self):
        factors = {'g': np.array(['X', 'Y'] * 5)}
        with pytest.raises(ValueError, match="coding"):
            build_model_matrix(factors, coding='helmert')

    def test_term_names_order(self):
        """Terms should be: Intercept, factors (sorted), covariates, interactions."""
        factors = {
            'B': np.array(['b1', 'b2'] * 5),
            'A': np.array(['a1', 'a2'] * 5),
        }
        mm = build_model_matrix(factors, coding='treatment')
        assert mm.term_names[0] == 'Intercept'
        assert mm.term_names[1] == 'A'
        assert mm.term_names[2] == 'B'
        assert mm.term_names[3] == 'A:B'
