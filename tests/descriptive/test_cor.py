"""
Tests for cor() — Pearson, Spearman, Kendall correlation.

All expected values are verified against R 4.5.2.
"""

import numpy as np
import pytest

from pystatistics.descriptive import cor


class TestPearsonCorrelation:
    """Test Pearson correlation matrix."""

    def test_identity_diagonal(self):
        """Diagonal must be exactly 1.0."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 4))
        result = cor(data, method='pearson', backend='cpu')
        np.testing.assert_array_equal(np.diag(result.correlation_pearson), 1.0)

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 4))
        result = cor(data, method='pearson', backend='cpu')
        C = result.correlation_pearson
        np.testing.assert_allclose(C, C.T, rtol=1e-12)

    def test_range_minus1_to_1(self):
        """All values must be in [-1, 1]."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 5))
        result = cor(data, method='pearson', backend='cpu')
        assert np.all(result.correlation_pearson >= -1.0 - 1e-10)
        assert np.all(result.correlation_pearson <= 1.0 + 1e-10)

    def test_perfect_positive_correlation(self):
        """Two perfectly correlated columns: r = 1.0."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0
        result = cor(x, y, method='pearson', backend='cpu')
        np.testing.assert_allclose(result.correlation_pearson[0, 1], 1.0, rtol=1e-12)

    def test_perfect_negative_correlation(self):
        """Two perfectly anti-correlated columns: r = -1.0."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = -3.0 * x + 10.0
        result = cor(x, y, method='pearson', backend='cpu')
        np.testing.assert_allclose(result.correlation_pearson[0, 1], -1.0, rtol=1e-12)

    def test_uncorrelated(self):
        """Orthogonal columns: r ≈ 0."""
        rng = np.random.default_rng(42)
        n = 10000
        data = rng.standard_normal((n, 2))
        result = cor(data, method='pearson', backend='cpu')
        assert abs(result.correlation_pearson[0, 1]) < 0.05

    def test_matches_numpy_corrcoef(self):
        """Must match numpy corrcoef."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 3))
        result = cor(data, method='pearson', backend='cpu')
        expected = np.corrcoef(data, rowvar=False)
        np.testing.assert_allclose(result.correlation_pearson, expected, rtol=1e-10)

    def test_single_column(self):
        """cor() on 1D data returns 1x1 matrix with value 1."""
        data = np.array([1.0, 2.0, 3.0])
        result = cor(data, method='pearson', backend='cpu')
        assert result.correlation_pearson.shape == (1, 1)
        np.testing.assert_allclose(result.correlation_pearson[0, 0], 1.0)

    def test_constant_column_nan(self):
        """Correlation with a constant column should be NaN."""
        data = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        result = cor(data, method='pearson', backend='cpu')
        # Col 1 is constant, var=0, so cor involving it should be NaN
        assert np.isnan(result.correlation_pearson[0, 1])

    def test_correlation_matrix_property(self):
        """correlation_matrix returns Pearson when Pearson is computed."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = cor(data, method='pearson', backend='cpu')
        assert result.correlation_matrix is not None
        np.testing.assert_array_equal(result.correlation_matrix, result.correlation_pearson)


class TestPearsonWithNaN:
    """Test Pearson correlation with missing data."""

    def test_everything_propagates(self):
        data = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        result = cor(data, method='pearson', use='everything', backend='cpu')
        # NaN in any element of a column propagates
        assert np.isnan(result.correlation_pearson[0, 1])

    def test_complete_obs(self):
        data = np.array([
            [1.0, 10.0],
            [2.0, np.nan],
            [3.0, 30.0],
            [4.0, 40.0],
        ])
        result = cor(data, method='pearson', use='complete.obs', backend='cpu')
        clean = np.array([[1.0, 10.0], [3.0, 30.0], [4.0, 40.0]])
        expected = np.corrcoef(clean, rowvar=False)
        np.testing.assert_allclose(result.correlation_pearson, expected, rtol=1e-10)

    def test_pairwise_complete_obs(self):
        data = np.array([
            [1.0, 10.0],
            [2.0, np.nan],
            [3.0, 30.0],
        ])
        result = cor(data, method='pearson', use='pairwise.complete.obs', backend='cpu')
        C = result.correlation_pearson
        # Diagonal must be 1
        np.testing.assert_allclose(C[0, 0], 1.0)
        np.testing.assert_allclose(C[1, 1], 1.0)
        # Off-diagonal: cor([1,3], [10,30]) = 1.0 (perfectly correlated)
        np.testing.assert_allclose(C[0, 1], 1.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# Spearman correlation
# ---------------------------------------------------------------------------

class TestSpearmanCorrelation:
    """Test Spearman rank correlation. Values verified against R cor(method='spearman')."""

    def test_matches_r_basic(self):
        """Matches R: cor(cbind(1:5, c(2,4,5,4,5)), method='spearman')[1,2]."""
        data = np.array([[1, 2], [2, 4], [3, 5], [4, 4], [5, 5]], dtype=np.float64)
        result = cor(data, method='spearman', backend='cpu')
        np.testing.assert_allclose(
            result.correlation_spearman[0, 1], 0.73786478737262184, rtol=1e-12
        )

    def test_identity_diagonal(self):
        """Diagonal must be exactly 1.0."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 3))
        result = cor(data, method='spearman', backend='cpu')
        np.testing.assert_array_equal(np.diag(result.correlation_spearman), 1.0)

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 3))
        result = cor(data, method='spearman', backend='cpu')
        C = result.correlation_spearman
        np.testing.assert_allclose(C, C.T, rtol=1e-12)

    def test_perfect_monotone(self):
        """Monotonically increasing: Spearman = 1.0."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 100.0, 1000.0, 10000.0, 100000.0])
        result = cor(x, y, method='spearman', backend='cpu')
        np.testing.assert_allclose(result.correlation_spearman[0, 1], 1.0, rtol=1e-12)

    def test_ties(self):
        """Data with ties: R gives cor=0 for this orthogonal-rank case."""
        data = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 1], [3, 2]], dtype=np.float64)
        result = cor(data, method='spearman', backend='cpu')
        np.testing.assert_allclose(result.correlation_spearman[0, 1], 0.0, atol=1e-12)

    def test_correlation_matrix_property(self):
        """correlation_matrix returns Spearman when Spearman is computed."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = cor(data, method='spearman', backend='cpu')
        assert result.correlation_matrix is not None
        np.testing.assert_array_equal(result.correlation_matrix, result.correlation_spearman)

    def test_xy_interface(self):
        """cor(x, y, method='spearman') works with two 1D arrays."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 5.0, 4.0, 5.0])
        result = cor(x, y, method='spearman', backend='cpu')
        np.testing.assert_allclose(
            result.correlation_spearman[0, 1], 0.73786478737262184, rtol=1e-12
        )


class TestSpearmanWithNaN:
    """Test Spearman correlation with missing data."""

    def test_pairwise_complete_obs(self):
        """Pairwise Spearman with NaN. R gives all 1s for perfectly monotone data."""
        data = np.array([
            [1, 10, 100],
            [2, np.nan, 200],
            [np.nan, 30, 300],
            [4, 40, np.nan],
            [5, 50, 500],
        ], dtype=np.float64)
        result = cor(data, method='spearman', use='pairwise.complete.obs', backend='cpu')
        # All pairwise correlations are perfect (monotone) after NaN removal
        C = result.correlation_spearman
        for i in range(3):
            for j in range(3):
                np.testing.assert_allclose(C[i, j], 1.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# Kendall correlation
# ---------------------------------------------------------------------------

class TestKendallCorrelation:
    """Test Kendall tau-b correlation. Values verified against R cor(method='kendall')."""

    def test_matches_r_basic(self):
        """Matches R: cor(cbind(1:5, c(2,4,5,4,5)), method='kendall')[1,2]."""
        data = np.array([[1, 2], [2, 4], [3, 5], [4, 4], [5, 5]], dtype=np.float64)
        result = cor(data, method='kendall', backend='cpu')
        np.testing.assert_allclose(
            result.correlation_kendall[0, 1], 0.67082039324993692, rtol=1e-12
        )

    def test_identity_diagonal(self):
        """Diagonal must be exactly 1.0."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((30, 3))
        result = cor(data, method='kendall', backend='cpu')
        np.testing.assert_array_equal(np.diag(result.correlation_kendall), 1.0)

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((30, 3))
        result = cor(data, method='kendall', backend='cpu')
        C = result.correlation_kendall
        np.testing.assert_allclose(C, C.T, rtol=1e-12)

    def test_perfect_concordance(self):
        """Monotonically increasing: Kendall = 1.0."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = cor(x, y, method='kendall', backend='cpu')
        np.testing.assert_allclose(result.correlation_kendall[0, 1], 1.0, rtol=1e-12)

    def test_perfect_discordance(self):
        """Monotonically decreasing: Kendall = -1.0."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        result = cor(x, y, method='kendall', backend='cpu')
        np.testing.assert_allclose(result.correlation_kendall[0, 1], -1.0, rtol=1e-12)

    def test_ties(self):
        """Data with ties: R gives 0 for this orthogonal case."""
        data = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 1], [3, 2]], dtype=np.float64)
        result = cor(data, method='kendall', backend='cpu')
        np.testing.assert_allclose(result.correlation_kendall[0, 1], 0.0, atol=1e-12)

    def test_correlation_matrix_property(self):
        """correlation_matrix returns Kendall when Kendall is computed."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = cor(data, method='kendall', backend='cpu')
        assert result.correlation_matrix is not None
        np.testing.assert_array_equal(result.correlation_matrix, result.correlation_kendall)


class TestKendallWithNaN:
    """Test Kendall correlation with missing data."""

    def test_pairwise_complete_obs(self):
        """Pairwise Kendall with NaN. Perfectly monotone pairwise => tau=1."""
        data = np.array([
            [1, 10, 100],
            [2, np.nan, 200],
            [np.nan, 30, 300],
            [4, 40, np.nan],
            [5, 50, 500],
        ], dtype=np.float64)
        result = cor(data, method='kendall', use='pairwise.complete.obs', backend='cpu')
        C = result.correlation_kendall
        for i in range(3):
            for j in range(3):
                np.testing.assert_allclose(C[i, j], 1.0, rtol=1e-10)
