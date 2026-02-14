"""
Tests for quantile computation (all 9 R types) and summary().

All expected values are computed directly from R 4.x using:
    quantile(x, probs, type=t)
    summary(x)
"""

from __future__ import annotations

import numpy as np
import pytest

from pystatistics.descriptive import quantile, summary, describe
from pystatistics.descriptive._quantile_types import r_quantile
from pystatistics.core.exceptions import ValidationError


# ---------------------------------------------------------------------------
# Expected values from R (verified against R 4.5.2)
# ---------------------------------------------------------------------------

# x = 1:5, probs = c(0, 0.25, 0.5, 0.75, 1)
R_QUANTILES_1TO5 = {
    1: [1, 2, 3, 4, 5],
    2: [1, 2, 3, 4, 5],
    3: [1, 1, 2, 4, 5],
    4: [1, 1.25, 2.5, 3.75, 5],
    5: [1, 1.75, 3, 4.25, 5],
    6: [1, 1.5, 3, 4.5, 5],
    7: [1, 2, 3, 4, 5],
    8: [1, 5 / 3, 3, 13 / 3, 5],
    9: [1, 1.6875, 3, 4.3125, 5],
}

# x = sort(c(2.1, 5.3, 8.7, 1.4, 9.2, 3.6, 7.8, 4.5, 6.9, 0.3))
# probs = c(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)
R_QUANTILES_10ELEM = {
    1: [0.3, 0.3, 2.1, 4.5, 7.8, 8.7, 9.2],
    2: [0.3, 0.85, 2.1, 4.9, 7.8, 8.95, 9.2],
    3: [0.3, 0.3, 1.4, 4.5, 7.8, 8.7, 9.2],
    4: [0.3, 0.3, 1.75, 4.5, 7.35, 8.7, 9.2],
    5: [0.3, 0.85, 2.1, 4.9, 7.8, 8.95, 9.2],
    6: [0.3, 0.41, 1.925, 4.9, 8.025, 9.15, 9.2],
    7: [0.3, 1.29, 2.475, 4.9, 7.575, 8.75, 9.2],
    8: [0.3, 0.7033333333333331, 2.0416666666666665, 4.9, 7.875, 9.0166666666666657, 9.2],
    9: [0.3, 0.74, 2.05625, 4.9, 7.85625, 9.0, 9.2],
}

# x = c(10, 20), probs = c(0, 0.25, 0.5, 0.75, 1)
R_QUANTILES_N2 = {
    1: [10, 10, 10, 20, 20],
    2: [10, 10, 15, 20, 20],
    3: [10, 10, 10, 20, 20],
    4: [10, 10, 10, 15, 20],
    5: [10, 10, 15, 20, 20],
    6: [10, 10, 15, 20, 20],
    7: [10, 12.5, 15, 17.5, 20],
    8: [10, 10, 15, 20, 20],
    9: [10, 10, 15, 20, 20],
}

# x = c(1,1,1,2,2,3), probs = c(0, 0.25, 0.5, 0.75, 1)
R_QUANTILES_TIES = {
    1: [1, 1, 1, 2, 3],
    2: [1, 1, 1.5, 2, 3],
    3: [1, 1, 1, 2, 3],
    4: [1, 1, 1, 2, 3],
    5: [1, 1, 1.5, 2, 3],
    6: [1, 1, 1.5, 2.25, 3],
    7: [1, 1, 1.5, 2, 3],
    8: [1, 1, 1.5, 2.0833333333333330, 3],
    9: [1, 1, 1.5, 2.0625, 3],
}


# ---------------------------------------------------------------------------
# Low-level tests: r_quantile() function
# ---------------------------------------------------------------------------

class TestRQuantileBasic:
    """Tests for the low-level r_quantile function."""

    @pytest.mark.parametrize("qtype", range(1, 10))
    def test_type_x_1to5(self, qtype):
        """All 9 types match R for x=1:5."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = r_quantile(x, probs, qtype)
        expected = np.array(R_QUANTILES_1TO5[qtype], dtype=np.float64)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    @pytest.mark.parametrize("qtype", range(1, 10))
    def test_type_10elem(self, qtype):
        """All 9 types match R for a 10-element unsorted dataset."""
        x = np.sort(np.array([2.1, 5.3, 8.7, 1.4, 9.2, 3.6, 7.8, 4.5, 6.9, 0.3]))
        probs = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        result = r_quantile(x, probs, qtype)
        expected = np.array(R_QUANTILES_10ELEM[qtype], dtype=np.float64)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    @pytest.mark.parametrize("qtype", range(1, 10))
    def test_type_n2(self, qtype):
        """All 9 types match R for n=2 edge case."""
        x = np.array([10.0, 20.0])
        probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = r_quantile(x, probs, qtype)
        expected = np.array(R_QUANTILES_N2[qtype], dtype=np.float64)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    @pytest.mark.parametrize("qtype", range(1, 10))
    def test_type_ties(self, qtype):
        """All 9 types match R for data with ties."""
        x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
        probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = r_quantile(x, probs, qtype)
        expected = np.array(R_QUANTILES_TIES[qtype], dtype=np.float64)
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestRQuantileEdgeCases:
    """Edge cases for r_quantile."""

    def test_single_element(self):
        """n=1: all quantiles equal the single value."""
        x = np.array([42.0])
        probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        for qtype in range(1, 10):
            result = r_quantile(x, probs, qtype)
            np.testing.assert_array_equal(result, np.full(5, 42.0))

    def test_empty_array(self):
        """n=0: returns all NaN."""
        x = np.array([], dtype=np.float64)
        probs = np.array([0.0, 0.5, 1.0])
        result = r_quantile(x, probs, 7)
        assert np.all(np.isnan(result))

    def test_single_prob(self):
        """Single probability value."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = r_quantile(x, np.array([0.5]), 7)
        np.testing.assert_allclose(result, [3.0])

    def test_extreme_probs(self):
        """probs=0 gives min, probs=1 gives max."""
        x = np.array([10.0, 20.0, 30.0])
        for qtype in range(1, 10):
            result = r_quantile(x, np.array([0.0, 1.0]), qtype)
            assert result[0] == 10.0, f"Type {qtype}: min"
            assert result[1] == 30.0, f"Type {qtype}: max"

    def test_invalid_type(self):
        """Type outside 1-9 raises ValidationError."""
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValidationError):
            r_quantile(x, np.array([0.5]), 0)
        with pytest.raises(ValidationError):
            r_quantile(x, np.array([0.5]), 10)

    def test_constant_data(self):
        """All same values: quantile is always that value."""
        x = np.array([7.0, 7.0, 7.0, 7.0, 7.0])
        probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        for qtype in range(1, 10):
            result = r_quantile(x, probs, qtype)
            np.testing.assert_array_equal(result, np.full(5, 7.0))


# ---------------------------------------------------------------------------
# High-level tests: quantile() solver function
# ---------------------------------------------------------------------------

class TestQuantileSolver:
    """Tests for the quantile() solver entry point."""

    def test_default_probs(self):
        """Default probs are (0, 0.25, 0.5, 0.75, 1.0)."""
        result = quantile([1, 2, 3, 4, 5])
        assert result.quantile_probs is not None
        np.testing.assert_array_equal(
            result.quantile_probs, [0.0, 0.25, 0.5, 0.75, 1.0]
        )

    def test_default_type_is_7(self):
        """Default type is 7 (R default)."""
        result = quantile([1, 2, 3, 4, 5])
        assert result.quantile_type == 7

    def test_custom_probs(self):
        """Custom probabilities are used."""
        result = quantile([1, 2, 3, 4, 5], probs=[0.1, 0.9])
        assert result.quantiles is not None
        assert result.quantiles.shape == (2, 1)

    @pytest.mark.parametrize("qtype", range(1, 10))
    def test_type_matches_r(self, qtype):
        """All 9 types through quantile() match R."""
        result = quantile([1, 2, 3, 4, 5], type=qtype)
        assert result.quantiles is not None
        expected = np.array(R_QUANTILES_1TO5[qtype], dtype=np.float64)
        np.testing.assert_allclose(result.quantiles[:, 0], expected, rtol=1e-12)

    def test_2d_data(self):
        """Quantiles computed per column for 2D data."""
        data = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]],
                        dtype=np.float64)
        result = quantile(data, type=7)
        assert result.quantiles is not None
        assert result.quantiles.shape == (5, 2)
        # Column 0: same as 1:5
        np.testing.assert_allclose(result.quantiles[:, 0], [1, 2, 3, 4, 5])
        # Column 1: 10x scale
        np.testing.assert_allclose(result.quantiles[:, 1], [10, 20, 30, 40, 50])

    def test_invalid_type_raises(self):
        """Type outside 1-9 raises ValidationError."""
        with pytest.raises(ValidationError):
            quantile([1, 2, 3], type=0)
        with pytest.raises(ValidationError):
            quantile([1, 2, 3], type=10)

    def test_nan_everything_propagates(self):
        """use='everything' propagates NaN."""
        data = [1, 2, np.nan, 4, 5]
        result = quantile(data, use='everything')
        assert result.quantiles is not None
        assert np.all(np.isnan(result.quantiles[:, 0]))

    def test_nan_complete_obs(self):
        """use='complete.obs' drops NaN rows."""
        data = [1, 2, np.nan, 4, 5]
        result = quantile(data, use='complete.obs', type=7)
        assert result.quantiles is not None
        # After removing NaN: [1, 2, 4, 5], type 7
        # R: quantile(c(1,2,4,5), type=7)
        # 0%=1, 25%=1.75, 50%=3, 75%=4.25, 100%=5
        expected = np.array([1.0, 1.75, 3.0, 4.25, 5.0])
        np.testing.assert_allclose(result.quantiles[:, 0], expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------

class TestSummary:
    """Tests for the summary() solver function."""

    def test_basic_summary(self):
        """Six-number summary matches R summary()."""
        # R: summary(1:5)
        # Min. 1st Qu. Median Mean 3rd Qu. Max.
        # 1.0    2.0    3.0   3.0    4.0   5.0
        result = summary([1, 2, 3, 4, 5])
        assert result.summary_table is not None
        table = result.summary_table[:, 0]
        np.testing.assert_allclose(table[0], 1.0)   # Min
        np.testing.assert_allclose(table[1], 2.0)   # Q1
        np.testing.assert_allclose(table[2], 3.0)   # Median
        np.testing.assert_allclose(table[3], 3.0)   # Mean
        np.testing.assert_allclose(table[4], 4.0)   # Q3
        np.testing.assert_allclose(table[5], 5.0)   # Max

    def test_summary_2d(self):
        """Summary computed per column."""
        data = np.array([[1, 100], [2, 200], [3, 300]], dtype=np.float64)
        result = summary(data)
        assert result.summary_table is not None
        assert result.summary_table.shape == (6, 2)
        # Column 0: Min=1, Mean=2, Max=3
        np.testing.assert_allclose(result.summary_table[0, 0], 1.0)
        np.testing.assert_allclose(result.summary_table[3, 0], 2.0)
        np.testing.assert_allclose(result.summary_table[5, 0], 3.0)
        # Column 1: Min=100, Mean=200, Max=300
        np.testing.assert_allclose(result.summary_table[0, 1], 100.0)
        np.testing.assert_allclose(result.summary_table[3, 1], 200.0)
        np.testing.assert_allclose(result.summary_table[5, 1], 300.0)

    def test_summary_nan_everything(self):
        """Summary with NaN and use='everything' propagates NaN."""
        data = [1, np.nan, 3, 4, 5]
        result = summary(data, use='everything')
        assert result.summary_table is not None
        assert np.all(np.isnan(result.summary_table[:, 0]))

    def test_summary_nan_complete_obs(self):
        """Summary with NaN and use='complete.obs' works."""
        data = [1, np.nan, 3, 4, 5]
        result = summary(data, use='complete.obs')
        assert result.summary_table is not None
        table = result.summary_table[:, 0]
        np.testing.assert_allclose(table[0], 1.0)    # Min
        np.testing.assert_allclose(table[5], 5.0)    # Max
        np.testing.assert_allclose(table[3], 3.25)   # Mean of [1,3,4,5]

    def test_summary_text_output(self):
        """summary() method returns formatted text."""
        result = summary([1, 2, 3, 4, 5])
        text = result.summary()
        assert "Min." in text
        assert "Median" in text
        assert "Max." in text


# ---------------------------------------------------------------------------
# describe() includes quantiles and summary
# ---------------------------------------------------------------------------

class TestDescribeQuantiles:
    """Verify describe() includes quantiles and summary."""

    def test_describe_includes_quantiles(self):
        """describe() populates quantiles."""
        result = describe([1, 2, 3, 4, 5])
        assert result.quantiles is not None
        assert result.quantile_type == 7

    def test_describe_includes_summary(self):
        """describe() populates summary_table."""
        result = describe([1, 2, 3, 4, 5])
        assert result.summary_table is not None
        assert result.summary_table.shape == (6, 1)

    def test_describe_custom_quantile_type(self):
        """describe() respects quantile_type parameter."""
        result = describe([1, 2, 3, 4, 5], quantile_type=1)
        assert result.quantile_type == 1
        expected = np.array(R_QUANTILES_1TO5[1], dtype=np.float64)
        np.testing.assert_allclose(result.quantiles[:, 0], expected, rtol=1e-12)
