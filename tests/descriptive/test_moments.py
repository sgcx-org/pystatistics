"""
Tests for skewness and kurtosis.

Expected values verified against R e1071::skewness(type=2) and
e1071::kurtosis(type=2) (bias-adjusted).
"""

import numpy as np
import pytest

from pystatistics.descriptive import describe


class TestSkewness:
    """Test bias-adjusted skewness (e1071 type 2)."""

    def test_symmetric_zero_skewness(self):
        """Symmetric data (1:5) has skewness = 0."""
        result = describe([1, 2, 3, 4, 5], backend='cpu')
        np.testing.assert_allclose(result.skewness[0], 0.0, atol=1e-12)

    def test_matches_r_negative_skew(self):
        """R: skewness(c(2,4,5,4,5), type=2) = -1.3608276348795434."""
        result = describe([2, 4, 5, 4, 5], backend='cpu')
        np.testing.assert_allclose(
            result.skewness[0], -1.3608276348795434, rtol=1e-12
        )

    def test_matches_r_positive_skew(self):
        """R: skewness(c(1,1,1,1,1,2,2,3,10), type=2) = 2.6862825854169157."""
        result = describe([1, 1, 1, 1, 1, 2, 2, 3, 10], backend='cpu')
        np.testing.assert_allclose(
            result.skewness[0], 2.6862825854169157, rtol=1e-12
        )

    def test_multicolumn(self):
        """Skewness computed per column."""
        data = np.array([
            [1, 2],
            [2, 4],
            [3, 5],
            [4, 4],
            [5, 5],
        ], dtype=np.float64)
        result = describe(data, backend='cpu')
        assert result.skewness.shape == (2,)
        np.testing.assert_allclose(result.skewness[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(result.skewness[1], -1.3608276348795434, rtol=1e-12)

    def test_nan_everything_propagates(self):
        """NaN propagates with use='everything'."""
        result = describe([1, 2, np.nan, 4, 5], use='everything', backend='cpu')
        assert np.isnan(result.skewness[0])

    def test_nan_complete_obs(self):
        """NaN rows removed with use='complete.obs'."""
        # After NaN removal: [1, 2, 4, 5] which is still symmetric around 3
        result = describe([1, 2, np.nan, 4, 5], use='complete.obs', backend='cpu')
        assert not np.isnan(result.skewness[0])

    def test_constant_nan(self):
        """Constant column has zero variance => skewness = NaN."""
        result = describe([5, 5, 5, 5, 5], backend='cpu')
        assert np.isnan(result.skewness[0])

    def test_too_few_observations(self):
        """n < 3 returns NaN (formula requires n >= 3)."""
        result = describe([1, 2], backend='cpu')
        assert np.isnan(result.skewness[0])


class TestKurtosis:
    """Test bias-adjusted excess kurtosis (e1071 type 2)."""

    def test_matches_r_uniform_like(self):
        """R: kurtosis(1:5, type=2) = -1.2000000000000004."""
        result = describe([1, 2, 3, 4, 5], backend='cpu')
        np.testing.assert_allclose(
            result.kurtosis[0], -1.2000000000000004, rtol=1e-12
        )

    def test_matches_r_leptokurtic(self):
        """R: kurtosis(c(2,4,5,4,5), type=2) = 2.0."""
        result = describe([2, 4, 5, 4, 5], backend='cpu')
        np.testing.assert_allclose(result.kurtosis[0], 2.0, rtol=1e-12)

    def test_matches_r_heavy_tail(self):
        """R: kurtosis(c(1,1,1,1,1,2,2,3,10), type=2) = 7.5125798985362451."""
        result = describe([1, 1, 1, 1, 1, 2, 2, 3, 10], backend='cpu')
        np.testing.assert_allclose(
            result.kurtosis[0], 7.5125798985362451, rtol=1e-12
        )

    def test_multicolumn(self):
        """Kurtosis computed per column."""
        data = np.array([
            [1, 2],
            [2, 4],
            [3, 5],
            [4, 4],
            [5, 5],
        ], dtype=np.float64)
        result = describe(data, backend='cpu')
        assert result.kurtosis.shape == (2,)
        np.testing.assert_allclose(result.kurtosis[0], -1.2000000000000004, rtol=1e-12)
        np.testing.assert_allclose(result.kurtosis[1], 2.0, rtol=1e-12)

    def test_nan_everything_propagates(self):
        """NaN propagates with use='everything'."""
        result = describe([1, 2, np.nan, 4, 5], use='everything', backend='cpu')
        assert np.isnan(result.kurtosis[0])

    def test_constant_nan(self):
        """Constant column has zero variance => kurtosis = NaN."""
        result = describe([5, 5, 5, 5, 5], backend='cpu')
        assert np.isnan(result.kurtosis[0])

    def test_too_few_observations(self):
        """n < 4 returns NaN (formula requires n >= 4)."""
        result = describe([1, 2, 3], backend='cpu')
        assert np.isnan(result.kurtosis[0])

    def test_normal_near_zero(self):
        """Normal data should have excess kurtosis near 0."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(10000)
        result = describe(data, backend='cpu')
        assert abs(result.kurtosis[0]) < 0.3  # Should be close to 0
