"""
Tests for input validation utilities.

Validates every function in core/validation.py:
    - check_array: conversion, dtype coercion, object rejection
    - check_finite: NaN/Inf detection
    - check_ndim / check_1d / check_2d: dimensionality checks
    - check_consistent_length: multi-array length matching
    - check_min_samples: minimum sample count
    - check_no_zero_variance_columns: constant column detection
    - check_column_rank: rank deficiency detection
"""

import numpy as np
import pytest

from pystatistics.core.exceptions import DimensionError, ValidationError
from pystatistics.core.validation import (
    check_1d,
    check_2d,
    check_array,
    check_column_rank,
    check_consistent_length,
    check_finite,
    check_min_samples,
    check_ndim,
    check_no_zero_variance_columns,
)


# ═══════════════════════════════════════════════════════════════════════
# check_array
# ═══════════════════════════════════════════════════════════════════════


class TestCheckArray:
    """check_array converts to ndarray and rejects non-numeric data."""

    def test_list_to_float_array(self):
        result = check_array([1, 2, 3], "X")
        assert isinstance(result, np.ndarray)
        assert np.issubdtype(result.dtype, np.floating)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_int_array_promoted_to_float(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = check_array(arr, "X")
        assert np.issubdtype(result.dtype, np.floating)

    def test_float_array_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = check_array(arr, "X")
        assert result.dtype == np.float64

    def test_float32_preserved(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = check_array(arr, "X")
        assert np.issubdtype(result.dtype, np.floating)

    def test_nested_list_to_2d(self):
        result = check_array([[1, 2], [3, 4]], "X")
        assert result.shape == (2, 2)
        assert np.issubdtype(result.dtype, np.floating)

    def test_rejects_mixed_types(self):
        """Mixed types (str + numeric) produce object dtype → ValidationError."""
        with pytest.raises(ValidationError, match="object dtype"):
            check_array([None, 1, 2.0], "X")

    def test_rejects_homogeneous_strings(self):
        """Homogeneous string arrays rejected as non-numeric dtype."""
        with pytest.raises(ValidationError, match="non-numeric dtype"):
            check_array(["a", "b", "c"], "X")

    def test_rejects_string_2d(self):
        with pytest.raises(ValidationError, match="non-numeric dtype"):
            check_array([["a", "b"], ["c", "d"]], "X")

    def test_empty_array(self):
        result = check_array([], "X")
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_scalar_becomes_0d(self):
        result = check_array(5.0, "X")
        assert isinstance(result, np.ndarray)

    def test_error_message_includes_name_object(self):
        """Object-dtype inputs get name in the error message."""
        with pytest.raises(ValidationError, match="my_var"):
            check_array([None, 1], "my_var")

    def test_error_message_includes_name_string(self):
        """String-dtype inputs get name in the error message."""
        with pytest.raises(ValidationError, match="my_var"):
            check_array(["a", "b"], "my_var")


# ═══════════════════════════════════════════════════════════════════════
# check_finite
# ═══════════════════════════════════════════════════════════════════════


class TestCheckFinite:
    """check_finite rejects NaN and Inf values."""

    def test_finite_passes(self):
        check_finite(np.array([1.0, 2.0, 3.0]), "X")  # no exception

    def test_nan_rejected(self):
        with pytest.raises(ValidationError, match="1 NaN"):
            check_finite(np.array([1.0, np.nan, 3.0]), "X")

    def test_inf_rejected(self):
        with pytest.raises(ValidationError, match="1 Inf"):
            check_finite(np.array([1.0, np.inf, 3.0]), "X")

    def test_neg_inf_rejected(self):
        with pytest.raises(ValidationError, match="1 Inf"):
            check_finite(np.array([1.0, -np.inf, 3.0]), "X")

    def test_mixed_nan_inf(self):
        with pytest.raises(ValidationError, match="2 NaN.*1 Inf"):
            check_finite(np.array([np.nan, np.inf, np.nan]), "X")

    def test_2d_finite_passes(self):
        check_finite(np.array([[1.0, 2.0], [3.0, 4.0]]), "X")

    def test_2d_nan_rejected(self):
        with pytest.raises(ValidationError):
            check_finite(np.array([[1.0, np.nan], [3.0, 4.0]]), "X")

    def test_error_message_includes_name(self):
        with pytest.raises(ValidationError, match="my_data"):
            check_finite(np.array([np.nan]), "my_data")


# ═══════════════════════════════════════════════════════════════════════
# check_ndim / check_1d / check_2d
# ═══════════════════════════════════════════════════════════════════════


class TestCheckNdim:
    """check_ndim, check_1d, check_2d enforce dimensionality."""

    def test_1d_passes_ndim_1(self):
        check_ndim(np.array([1.0, 2.0]), 1, "X")

    def test_2d_passes_ndim_2(self):
        check_ndim(np.array([[1.0, 2.0]]), 2, "X")

    def test_wrong_ndim_raises(self):
        with pytest.raises(DimensionError, match="expected 1D.*got 2D"):
            check_ndim(np.array([[1.0, 2.0]]), 1, "X")

    def test_error_includes_shape(self):
        with pytest.raises(DimensionError, match=r"shape \(3, 2\)"):
            check_ndim(np.ones((3, 2)), 1, "X")

    def test_check_1d_passes(self):
        check_1d(np.array([1.0, 2.0, 3.0]), "y")

    def test_check_1d_rejects_2d(self):
        with pytest.raises(DimensionError):
            check_1d(np.array([[1.0, 2.0]]), "y")

    def test_check_2d_passes(self):
        check_2d(np.array([[1.0, 2.0], [3.0, 4.0]]), "X")

    def test_check_2d_rejects_1d(self):
        with pytest.raises(DimensionError):
            check_2d(np.array([1.0, 2.0, 3.0]), "X")

    def test_check_2d_rejects_3d(self):
        with pytest.raises(DimensionError):
            check_2d(np.ones((2, 3, 4)), "X")

    def test_error_message_includes_name(self):
        with pytest.raises(DimensionError, match="my_matrix"):
            check_2d(np.array([1.0]), "my_matrix")


# ═══════════════════════════════════════════════════════════════════════
# check_consistent_length
# ═══════════════════════════════════════════════════════════════════════


class TestCheckConsistentLength:
    """check_consistent_length ensures first dimensions match."""

    def test_same_length_passes(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        check_consistent_length(a, b, names=("X", "y"))

    def test_different_length_raises(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        with pytest.raises(DimensionError, match="Inconsistent lengths"):
            check_consistent_length(a, b, names=("X", "y"))

    def test_error_includes_names_and_lengths(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        with pytest.raises(DimensionError, match=r"X=3.*y=2"):
            check_consistent_length(a, b, names=("X", "y"))

    def test_three_arrays_all_same(self):
        a = np.ones(10)
        b = np.ones(10)
        c = np.ones(10)
        check_consistent_length(a, b, c, names=("a", "b", "c"))

    def test_three_arrays_one_different(self):
        a = np.ones(10)
        b = np.ones(10)
        c = np.ones(5)
        with pytest.raises(DimensionError):
            check_consistent_length(a, b, c, names=("a", "b", "c"))

    def test_2d_arrays_check_first_dim(self):
        a = np.ones((10, 3))
        b = np.ones(10)
        check_consistent_length(a, b, names=("X", "y"))

    def test_2d_arrays_different_first_dim(self):
        a = np.ones((10, 3))
        b = np.ones(5)
        with pytest.raises(DimensionError):
            check_consistent_length(a, b, names=("X", "y"))

    def test_wrong_number_of_names(self):
        a = np.ones(5)
        b = np.ones(5)
        with pytest.raises(ValueError, match="Number of arrays"):
            check_consistent_length(a, b, names=("X",))

    def test_single_array_passes(self):
        """Single array always passes (nothing to compare)."""
        check_consistent_length(np.ones(5), names=("X",))

    def test_empty_arrays_pass(self):
        check_consistent_length(
            np.array([]), np.array([]), names=("a", "b")
        )


# ═══════════════════════════════════════════════════════════════════════
# check_min_samples
# ═══════════════════════════════════════════════════════════════════════


class TestCheckMinSamples:
    """check_min_samples enforces minimum sample count."""

    def test_sufficient_samples_pass(self):
        check_min_samples(np.ones(10), 5, "X")

    def test_exact_minimum_passes(self):
        check_min_samples(np.ones(5), 5, "X")

    def test_too_few_raises(self):
        with pytest.raises(ValidationError, match="at least 5.*got 3"):
            check_min_samples(np.ones(3), 5, "X")

    def test_empty_array_raises(self):
        with pytest.raises(ValidationError, match="at least 1.*got 0"):
            check_min_samples(np.array([]), 1, "X")

    def test_2d_checks_first_dim(self):
        check_min_samples(np.ones((10, 3)), 5, "X")

    def test_2d_too_few_rows(self):
        with pytest.raises(ValidationError):
            check_min_samples(np.ones((2, 10)), 5, "X")

    def test_error_message_includes_name(self):
        with pytest.raises(ValidationError, match="my_data"):
            check_min_samples(np.ones(1), 10, "my_data")


# ═══════════════════════════════════════════════════════════════════════
# check_no_zero_variance_columns
# ═══════════════════════════════════════════════════════════════════════


class TestCheckNoZeroVarianceColumns:
    """check_no_zero_variance_columns detects constant columns."""

    def test_varying_columns_pass(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        check_no_zero_variance_columns(X, "X")

    def test_constant_column_raises(self):
        X = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
        with pytest.raises(ValidationError, match="zero variance"):
            check_no_zero_variance_columns(X, "X")

    def test_error_reports_column_indices(self):
        X = np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 3.0], [1.0, 6.0, 3.0]])
        with pytest.raises(ValidationError, match=r"\[0, 2\]"):
            check_no_zero_variance_columns(X, "X")

    def test_all_constant_raises(self):
        X = np.array([[5.0, 3.0], [5.0, 3.0], [5.0, 3.0]])
        with pytest.raises(ValidationError, match=r"\[0, 1\]"):
            check_no_zero_variance_columns(X, "X")

    def test_single_column_varying_passes(self):
        X = np.array([[1.0], [2.0], [3.0]])
        check_no_zero_variance_columns(X, "X")

    def test_single_column_constant_raises(self):
        X = np.array([[1.0], [1.0], [1.0]])
        with pytest.raises(ValidationError, match="zero variance"):
            check_no_zero_variance_columns(X, "X")


# ═══════════════════════════════════════════════════════════════════════
# check_column_rank
# ═══════════════════════════════════════════════════════════════════════


class TestCheckColumnRank:
    """check_column_rank detects rank deficiency."""

    def test_full_rank_passes(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        check_column_rank(X, "X")

    def test_rank_deficient_raises(self):
        # Column 2 = 2 * Column 1 → perfect multicollinearity
        X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        with pytest.raises(ValidationError, match="rank-deficient"):
            check_column_rank(X, "X")

    def test_error_includes_rank_info(self):
        X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        with pytest.raises(ValidationError, match="rank=1"):
            check_column_rank(X, "X")

    def test_error_mentions_multicollinearity(self):
        X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        with pytest.raises(ValidationError, match="multicollinearity"):
            check_column_rank(X, "X")

    def test_single_column_nonzero_passes(self):
        X = np.array([[1.0], [2.0], [3.0]])
        check_column_rank(X, "X")

    def test_single_column_zero_raises(self):
        X = np.array([[0.0], [0.0], [0.0]])
        with pytest.raises(ValidationError, match="rank-deficient"):
            check_column_rank(X, "X")

    def test_wide_matrix_full_rank(self):
        """More columns than rows, but still full column rank for p <= n."""
        # Actually can't have full column rank if p > n
        # p=3, n=2 → rank <= 2 < p=3
        X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with pytest.raises(ValidationError, match="rank-deficient"):
            check_column_rank(X, "X")

    def test_square_identity_passes(self):
        X = np.eye(5)
        check_column_rank(X, "X")

    def test_near_singular_passes(self):
        """Numerically near-singular but still full rank."""
        X = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-10], [1.0, 2.0]])
        # This should have rank 2 (numerically)
        check_column_rank(X, "X")
