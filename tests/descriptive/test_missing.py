"""
Tests for missing data handling.
"""

import numpy as np
import pytest

from pystatistics.descriptive._missing import (
    apply_use_policy, pairwise_mask, columnwise_clean,
)
from pystatistics.core.exceptions import ValidationError


class TestApplyUsePolicy:
    """Test the apply_use_policy function."""

    def test_everything_preserves_data(self):
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        clean, n_complete = apply_use_policy(data, 'everything')
        np.testing.assert_array_equal(clean, data)
        assert n_complete == 1  # Only one row has no NaN

    def test_complete_obs_removes_nan_rows(self):
        data = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        clean, n_complete = apply_use_policy(data, 'complete.obs')
        assert clean.shape == (2, 2)
        assert n_complete == 2
        np.testing.assert_array_equal(clean, [[3.0, 4.0], [5.0, 6.0]])

    def test_complete_obs_all_nan_raises(self):
        data = np.array([[1.0, np.nan], [np.nan, 4.0]])
        with pytest.raises(ValidationError, match="No complete observations"):
            apply_use_policy(data, 'complete.obs')

    def test_complete_obs_no_nan(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        clean, n_complete = apply_use_policy(data, 'complete.obs')
        assert n_complete == 2
        np.testing.assert_array_equal(clean, data)

    def test_pairwise_preserves_data(self):
        """Pairwise just passes data through for column-wise ops."""
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        clean, n_complete = apply_use_policy(data, 'pairwise.complete.obs')
        np.testing.assert_array_equal(clean, data)

    def test_invalid_use_raises(self):
        data = np.array([[1.0]])
        with pytest.raises(ValidationError, match="Invalid use="):
            apply_use_policy(data, 'invalid')


class TestPairwiseMask:
    """Test pairwise_mask helper."""

    def test_no_nan(self):
        xi = np.array([1.0, 2.0, 3.0])
        xj = np.array([4.0, 5.0, 6.0])
        mask = pairwise_mask(xi, xj)
        assert mask.all()

    def test_with_nan_in_first(self):
        xi = np.array([1.0, np.nan, 3.0])
        xj = np.array([4.0, 5.0, 6.0])
        mask = pairwise_mask(xi, xj)
        np.testing.assert_array_equal(mask, [True, False, True])

    def test_with_nan_in_both(self):
        xi = np.array([np.nan, 2.0, 3.0])
        xj = np.array([4.0, np.nan, 6.0])
        mask = pairwise_mask(xi, xj)
        np.testing.assert_array_equal(mask, [False, False, True])


class TestColumnwiseClean:
    """Test columnwise_clean helper."""

    def test_removes_nan(self):
        col = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        clean = columnwise_clean(col)
        np.testing.assert_array_equal(clean, [1.0, 3.0, 5.0])

    def test_no_nan(self):
        col = np.array([1.0, 2.0, 3.0])
        clean = columnwise_clean(col)
        np.testing.assert_array_equal(clean, col)

    def test_all_nan(self):
        col = np.array([np.nan, np.nan])
        clean = columnwise_clean(col)
        assert len(clean) == 0
