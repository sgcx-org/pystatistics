"""
Tests for time series decomposition: classical decompose() and STL.

Validates component identities, seasonal properties, and edge cases.
Uses deterministic data for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from pystatistics.core.exceptions import ValidationError
from pystatistics.timeseries import decompose, stl, DecompositionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def additive_series():
    """Synthetic additive series: linear trend + sine seasonal + small noise."""
    rng = np.random.default_rng(42)
    n = 120
    period = 12
    t = np.arange(n, dtype=float)
    trend = 50.0 + 0.5 * t
    seasonal_pattern = 10.0 * np.sin(2.0 * np.pi * t / period)
    noise = rng.normal(0, 0.5, n)
    return trend + seasonal_pattern + noise, period, trend, seasonal_pattern


@pytest.fixture
def multiplicative_series():
    """Synthetic multiplicative series: exponential trend * seasonal * noise."""
    rng = np.random.default_rng(99)
    n = 96
    period = 4
    t = np.arange(n, dtype=float)
    trend = 100.0 * np.exp(0.005 * t)
    seasonal_pattern = 1.0 + 0.2 * np.sin(2.0 * np.pi * t / period)
    noise = 1.0 + rng.normal(0, 0.01, n)
    return trend * seasonal_pattern * noise, period


@pytest.fixture
def clean_additive():
    """Perfectly clean additive series (no noise) for exact recovery."""
    n = 60
    period = 12
    t = np.arange(n, dtype=float)
    trend = 100.0 + 2.0 * t
    raw_seasonal = np.array(
        [3.0, -1.0, 2.0, -4.0, 5.0, -2.0, 1.0, -3.0, 4.0, -5.0, 2.0, -2.0]
    )
    # Center it so sum = 0
    raw_seasonal = raw_seasonal - np.mean(raw_seasonal)
    seasonal = np.tile(raw_seasonal, n // period)
    return trend + seasonal, period, trend, seasonal


# ---------------------------------------------------------------------------
# Classical decompose() tests
# ---------------------------------------------------------------------------

class TestDecomposeAdditive:
    """Tests for additive classical decomposition."""

    def test_identity_holds(self, additive_series):
        """trend + seasonal + residual = observed (where trend is not NaN)."""
        y, period, _, _ = additive_series
        result = decompose(y, period, type="additive")
        valid = ~np.isnan(result.trend)
        reconstructed = result.trend[valid] + result.seasonal[valid] + result.residual[valid]
        np.testing.assert_allclose(reconstructed, result.observed[valid], atol=1e-10)

    def test_seasonal_repeats(self, additive_series):
        """Seasonal component repeats with period m."""
        y, period, _, _ = additive_series
        result = decompose(y, period, type="additive")
        first_cycle = result.seasonal[:period]
        for start in range(period, len(y), period):
            end = min(start + period, len(y))
            np.testing.assert_array_equal(
                result.seasonal[start:end], first_cycle[: end - start]
            )

    def test_seasonal_sums_to_zero(self, additive_series):
        """Additive seasonal component sums to approximately zero over one period."""
        y, period, _, _ = additive_series
        result = decompose(y, period, type="additive")
        cycle_sum = np.sum(result.seasonal[:period])
        assert abs(cycle_sum) < 1e-10

    def test_nan_at_edges(self, additive_series):
        """Trend has NaN at first and last period/2 values for even period."""
        y, period, _, _ = additive_series
        result = decompose(y, period, type="additive")
        k = period // 2
        assert np.all(np.isnan(result.trend[:k]))
        assert np.all(np.isnan(result.trend[-k:]))

    def test_trend_is_smooth(self, additive_series):
        """Trend component has low high-frequency content."""
        y, period, _, _ = additive_series
        result = decompose(y, period, type="additive")
        valid = ~np.isnan(result.trend)
        trend_valid = result.trend[valid]
        # Second differences should be small (smooth)
        d2 = np.diff(trend_valid, n=2)
        assert np.std(d2) < np.std(y) * 0.1

    def test_known_recovery(self, clean_additive):
        """Known constant-slope trend + fixed seasonal recovered correctly."""
        y, period, true_trend, true_seasonal = clean_additive
        result = decompose(y, period, type="additive")
        valid = ~np.isnan(result.trend)
        # Trend should closely match the linear trend in the interior
        np.testing.assert_allclose(
            result.trend[valid], true_trend[valid], atol=0.5
        )
        # Seasonal should closely match
        np.testing.assert_allclose(
            result.seasonal[:period], true_seasonal[:period], atol=0.5
        )


class TestDecomposeMultiplicative:
    """Tests for multiplicative classical decomposition."""

    def test_identity_holds(self, multiplicative_series):
        """trend * seasonal * residual = observed (where trend is not NaN)."""
        y, period = multiplicative_series
        result = decompose(y, period, type="multiplicative")
        valid = ~np.isnan(result.trend)
        reconstructed = result.trend[valid] * result.seasonal[valid] * result.residual[valid]
        np.testing.assert_allclose(reconstructed, result.observed[valid], rtol=1e-10)

    def test_seasonal_averages_to_one(self, multiplicative_series):
        """Multiplicative seasonal component averages to 1 over one period."""
        y, period = multiplicative_series
        result = decompose(y, period, type="multiplicative")
        cycle_mean = np.mean(result.seasonal[:period])
        np.testing.assert_allclose(cycle_mean, 1.0, atol=1e-10)

    def test_nan_at_edges(self, multiplicative_series):
        """Trend has NaN at edges for even period."""
        y, period = multiplicative_series
        result = decompose(y, period, type="multiplicative")
        k = period // 2
        assert np.all(np.isnan(result.trend[:k]))
        assert np.all(np.isnan(result.trend[-k:]))


class TestDecomposeValidation:
    """Input validation tests for decompose()."""

    def test_period_less_than_2(self):
        y = np.arange(20, dtype=float)
        with pytest.raises(ValidationError, match="period.*>= 2"):
            decompose(y, period=1)

    def test_series_too_short(self):
        y = np.arange(5, dtype=float)
        with pytest.raises(ValidationError, match="length.*>= 2 \\* period"):
            decompose(y, period=4)

    def test_non_positive_multiplicative(self):
        y = np.array([1.0, 2.0, -1.0, 3.0, 1.0, 2.0, -1.0, 3.0])
        with pytest.raises(ValidationError, match="all values > 0"):
            decompose(y, period=4, type="multiplicative")

    def test_invalid_type(self):
        y = np.ones(20)
        with pytest.raises(ValidationError, match="type"):
            decompose(y, period=4, type="invalid")

    def test_nan_input(self):
        y = np.ones(20)
        y[5] = np.nan
        with pytest.raises(ValidationError, match="non-finite"):
            decompose(y, period=4)


class TestDecomposeEdgeCases:
    """Edge cases for classical decomposition."""

    def test_exact_multiple_of_period(self):
        """Series length exactly divisible by period."""
        rng = np.random.default_rng(7)
        n = 48  # 48 / 12 = 4
        period = 12
        y = rng.normal(100, 10, n)
        result = decompose(y, period)
        assert len(result.seasonal) == n
        assert result.period == period

    def test_not_multiple_of_period(self):
        """Series length not divisible by period."""
        rng = np.random.default_rng(8)
        n = 50  # 50 / 12 != integer
        period = 12
        y = rng.normal(100, 10, n)
        result = decompose(y, period)
        assert len(result.seasonal) == n

    def test_minimum_viable_length(self):
        """Minimum viable series length = 2 * period."""
        rng = np.random.default_rng(9)
        period = 4
        n = 2 * period
        y = rng.normal(100, 5, n)
        result = decompose(y, period)
        assert result.method == "classical"
        assert len(result.observed) == n

    def test_odd_period(self):
        """Odd period uses simple centered MA (no 2xm)."""
        rng = np.random.default_rng(10)
        period = 7
        n = 56
        y = rng.normal(50, 5, n)
        result = decompose(y, period)
        k = period // 2  # 3
        assert np.all(np.isnan(result.trend[:k]))
        assert np.all(np.isnan(result.trend[-k:]))
        # Interior should not be NaN
        assert not np.any(np.isnan(result.trend[k: -k]))

    def test_result_is_frozen(self):
        """DecompositionResult is a frozen dataclass."""
        y = np.arange(24, dtype=float) + 1.0
        result = decompose(y, period=4)
        with pytest.raises(AttributeError):
            result.period = 10  # type: ignore[misc]

    def test_summary_returns_string(self):
        """summary() produces a non-empty string."""
        rng = np.random.default_rng(11)
        y = rng.normal(100, 5, 48)
        result = decompose(y, period=12)
        s = result.summary()
        assert isinstance(s, str)
        assert "classical" in s


# ---------------------------------------------------------------------------
# STL tests
# ---------------------------------------------------------------------------

class TestSTL:
    """Tests for STL decomposition."""

    def test_identity_holds(self, additive_series):
        """trend + seasonal + residual = observed."""
        y, period, _, _ = additive_series
        result = stl(y, period)
        reconstructed = result.trend + result.seasonal + result.residual
        np.testing.assert_allclose(reconstructed, result.observed, atol=1e-10)

    def test_seasonal_approximately_repeats(self, additive_series):
        """Seasonal component approximately repeats with period m.

        STL allows seasonal to evolve, so we compare interior cycles
        (excluding the first and last which may have edge effects).
        """
        y, period, _, _ = additive_series
        result = stl(y, period)
        n_cycles = len(y) // period
        # Compare consecutive interior cycles -- they should be similar
        for c in range(1, n_cycles - 2):
            c1 = result.seasonal[c * period: (c + 1) * period]
            c2 = result.seasonal[(c + 1) * period: (c + 2) * period]
            np.testing.assert_allclose(c1, c2, atol=3.0)

    def test_default_windows(self, additive_series):
        """STL works with default window parameters."""
        y, period, _, _ = additive_series
        result = stl(y, period)
        assert result.method == "stl"
        assert result.type == "additive"
        assert result.period == period

    def test_robust_handles_outliers(self):
        """robust=True reduces outlier influence on trend."""
        rng = np.random.default_rng(55)
        n = 120
        period = 12
        t = np.arange(n, dtype=float)
        trend = 50.0 + 0.3 * t
        seasonal = 5.0 * np.sin(2 * np.pi * t / period)
        y_clean = trend + seasonal + rng.normal(0, 0.5, n)

        # Add outliers
        y_outlier = y_clean.copy()
        y_outlier[30] += 50.0
        y_outlier[60] += 50.0
        y_outlier[90] += 50.0

        result_nonrobust = stl(y_outlier, period, robust=False)
        result_robust = stl(y_outlier, period, robust=True)

        # Robust trend should be closer to the clean trend at outlier locations
        clean_result = stl(y_clean, period)
        err_nonrobust = np.abs(result_nonrobust.trend[30] - clean_result.trend[30])
        err_robust = np.abs(result_robust.trend[30] - clean_result.trend[30])
        assert err_robust < err_nonrobust

    def test_seasonal_window_affects_result(self, additive_series):
        """Different seasonal_window values produce different decompositions."""
        y, period, _, _ = additive_series
        result_narrow = stl(y, period, seasonal_window=7)
        result_wide = stl(y, period, seasonal_window=21)
        # Different windows should produce different seasonal components
        assert not np.allclose(result_narrow.seasonal, result_wide.seasonal)
        # But both should still satisfy the identity
        recon_narrow = result_narrow.trend + result_narrow.seasonal + result_narrow.residual
        recon_wide = result_wide.trend + result_wide.seasonal + result_wide.residual
        np.testing.assert_allclose(recon_narrow, result_narrow.observed, atol=1e-10)
        np.testing.assert_allclose(recon_wide, result_wide.observed, atol=1e-10)


class TestSTLValidation:
    """Validation tests for stl()."""

    def test_seasonal_window_even(self):
        y = np.arange(24, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="seasonal_window.*odd"):
            stl(y, period=4, seasonal_window=8)

    def test_seasonal_window_too_small(self):
        y = np.arange(24, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="seasonal_window.*>= 7"):
            stl(y, period=4, seasonal_window=5)

    def test_trend_window_even(self):
        y = np.arange(24, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="trend_window.*odd"):
            stl(y, period=4, trend_window=8)

    def test_n_inner_zero(self):
        y = np.arange(24, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="n_inner.*>= 1"):
            stl(y, period=4, n_inner=0)

    def test_period_too_small(self):
        y = np.arange(20, dtype=float)
        with pytest.raises(ValidationError, match="period.*>= 2"):
            stl(y, period=1)

    def test_series_too_short(self):
        y = np.arange(5, dtype=float)
        with pytest.raises(ValidationError, match="length.*>= 2 \\* period"):
            stl(y, period=4)


class TestSTLEdgeCases:
    """Edge cases for STL."""

    def test_minimum_viable_length(self):
        """Minimum viable series length = 2 * period."""
        rng = np.random.default_rng(20)
        period = 4
        n = 2 * period
        y = rng.normal(100, 5, n)
        result = stl(y, period)
        assert len(result.observed) == n

    def test_result_is_frozen(self):
        """DecompositionResult from stl is frozen."""
        y = np.arange(24, dtype=float) + 1.0
        result = stl(y, period=4)
        with pytest.raises(AttributeError):
            result.method = "other"  # type: ignore[misc]

    def test_summary_returns_string(self):
        """summary() produces a non-empty string."""
        rng = np.random.default_rng(21)
        y = rng.normal(100, 5, 48)
        result = stl(y, period=12)
        s = result.summary()
        assert isinstance(s, str)
        assert "stl" in s
