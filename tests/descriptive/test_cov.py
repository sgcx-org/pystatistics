"""
Tests for cov() — covariance matrix with all use= modes.
"""

import numpy as np
import pytest

from pystatistics.descriptive import cov, DescriptiveDesign


class TestCovarianceBasic:
    """Test covariance matrix fundamentals."""

    def test_bessel_correction(self):
        """cov() must use n-1 denominator to match R."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = cov(data, backend='cpu')
        expected = np.cov(data, rowvar=False, ddof=1)
        np.testing.assert_allclose(result.covariance_matrix, expected, rtol=1e-12)

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 4))
        result = cov(data, backend='cpu')
        C = result.covariance_matrix
        np.testing.assert_allclose(C, C.T, rtol=1e-12)

    def test_diagonal_equals_variance(self):
        """Diagonal of cov matrix should equal per-column variance."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 3))
        result = cov(data, backend='cpu')
        expected_var = np.var(data, axis=0, ddof=1)
        np.testing.assert_allclose(
            np.diag(result.covariance_matrix), expected_var, rtol=1e-12
        )

    def test_single_column(self):
        """cov() on 1D data returns 1x1 matrix."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cov(data, backend='cpu')
        assert result.covariance_matrix.shape == (1, 1)
        np.testing.assert_allclose(
            result.covariance_matrix[0, 0],
            np.var(data, ddof=1),
            rtol=1e-12,
        )

    def test_perfect_correlation(self):
        """Two perfectly correlated columns: off-diagonal = var."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0
        data = np.column_stack([x, y])
        result = cov(data, backend='cpu')
        C = result.covariance_matrix
        # cov(x, 2x+1) = 2*var(x)
        np.testing.assert_allclose(C[0, 1], 2.0 * np.var(x, ddof=1), rtol=1e-12)

    def test_cov_xy_shorthand(self):
        """cov(x, y) stacks into 2-column matrix."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([10.0, 20.0, 30.0, 40.0])
        result = cov(x, y, backend='cpu')
        assert result.covariance_matrix.shape == (2, 2)


class TestCovarianceWithNaN:
    """Test covariance with missing data."""

    def test_everything_propagates_nan(self):
        """use='everything' propagates NaN."""
        data = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        result = cov(data, use='everything', backend='cpu')
        C = result.covariance_matrix
        # Column 1 has NaN, so any covariance involving col 1 is NaN
        assert np.isnan(C[0, 1])
        assert np.isnan(C[1, 0])

    def test_complete_obs_listwise_deletion(self):
        """use='complete.obs' removes rows with any NaN."""
        data = np.array([
            [1.0, 10.0],
            [2.0, np.nan],
            [3.0, 30.0],
            [4.0, 40.0],
        ])
        result = cov(data, use='complete.obs', backend='cpu')
        # After removing row 1: [[1,10], [3,30], [4,40]]
        clean = np.array([[1.0, 10.0], [3.0, 30.0], [4.0, 40.0]])
        expected = np.cov(clean, rowvar=False, ddof=1)
        np.testing.assert_allclose(result.covariance_matrix, expected, rtol=1e-12)

    def test_pairwise_uses_available_pairs(self):
        """use='pairwise.complete.obs' uses shared non-NaN rows per pair."""
        data = np.array([
            [1.0, 10.0],
            [2.0, np.nan],
            [3.0, 30.0],
        ])
        result = cov(data, use='pairwise.complete.obs', backend='cpu')
        C = result.covariance_matrix

        # Var(col0): all 3 rows available → var([1,2,3]) = 1.0
        np.testing.assert_allclose(C[0, 0], 1.0, rtol=1e-12)

        # Var(col1): rows 0,2 available → var([10,30]) = 200.0
        np.testing.assert_allclose(C[1, 1], 200.0, rtol=1e-12)

        # Cov(col0, col1): rows 0,2 → cov([1,3], [10,30])
        # = sum((x-mean(x))*(y-mean(y))) / (n-1)
        # = ((1-2)*(10-20) + (3-2)*(30-20)) / 1 = (10 + 10) / 1 = 20.0
        np.testing.assert_allclose(C[0, 1], 20.0, rtol=1e-12)

    def test_pairwise_n_counts(self):
        """Pairwise mode returns observation counts per pair."""
        data = np.array([
            [1.0, 10.0],
            [2.0, np.nan],
            [3.0, 30.0],
        ])
        result = cov(data, use='pairwise.complete.obs', backend='cpu')
        assert result.pairwise_n is not None
        # col0 has all 3 values, col1 has 2
        # (0,0): 3, (1,1): 2, (0,1): 2
        assert result.pairwise_n[0, 0] == 3
        assert result.pairwise_n[1, 1] == 2
        assert result.pairwise_n[0, 1] == 2

    def test_pairwise_no_shared_obs_returns_nan(self):
        """When no shared observations for a pair, return NaN."""
        data = np.array([
            [1.0, np.nan],
            [np.nan, 20.0],
        ])
        result = cov(data, use='pairwise.complete.obs', backend='cpu')
        assert np.isnan(result.covariance_matrix[0, 1])
