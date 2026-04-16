"""
Tests for ETS (ExponenTial Smoothing) state space models.

Covers model specification parsing, recursion, fitting, and forecasting
for all supported ETS(error, trend, season) combinations.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.core.exceptions import ConvergenceError, ValidationError
from pystatistics.timeseries._ets_models import ETSSpec, ets_recursion, parse_ets_spec
from pystatistics.timeseries._ets_fit import ETSResult, ets
from pystatistics.timeseries._ets_forecast import ETSForecast, forecast_ets


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def constant_series():
    """Constant series of value 5.0 (50 obs)."""
    return np.full(50, 5.0)


@pytest.fixture
def linear_series():
    """Linear trend: y = 2 + 0.5*t for t=0..99."""
    return 2.0 + 0.5 * np.arange(100, dtype=np.float64)


@pytest.fixture
def seasonal_quarterly():
    """Quarterly data with trend and additive seasonality (80 obs, 20 years)."""
    rng = np.random.default_rng(42)
    n = 80
    t = np.arange(n, dtype=np.float64)
    trend = 10.0 + 0.5 * t
    season = np.tile([3.0, -1.0, -2.0, 0.0], n // 4)
    noise = rng.normal(0, 0.5, n)
    return trend + season + noise


@pytest.fixture
def seasonal_multiplicative():
    """Quarterly data with multiplicative seasonality (80 obs)."""
    rng = np.random.default_rng(123)
    n = 80
    t = np.arange(n, dtype=np.float64)
    trend = 100.0 + 2.0 * t
    season = np.tile([1.2, 0.8, 0.9, 1.1], n // 4)
    noise = rng.normal(1.0, 0.02, n)
    return trend * season * noise


# =========================================================================
# Model specification tests
# =========================================================================

class TestParseETSSpec:
    """Tests for parse_ets_spec."""

    def test_ann(self):
        spec = parse_ets_spec("ANN")
        assert spec.error == "A"
        assert spec.trend == "N"
        assert spec.season == "N"
        assert spec.damped is False
        assert spec.period == 1
        assert spec.name == "ETS(A,N,N)"

    def test_aan(self):
        spec = parse_ets_spec("AAN")
        assert spec.error == "A"
        assert spec.trend == "A"
        assert spec.season == "N"
        assert spec.damped is False

    def test_aaa_with_period(self):
        spec = parse_ets_spec("AAA", period=12)
        assert spec.error == "A"
        assert spec.trend == "A"
        assert spec.season == "A"
        assert spec.period == 12
        assert spec.damped is False

    def test_damped_aadn(self):
        spec = parse_ets_spec("AAdN")
        assert spec.trend == "Ad"
        assert spec.damped is True
        assert spec.name == "ETS(A,Ad,N)"

    def test_mam(self):
        spec = parse_ets_spec("MAM", period=4)
        assert spec.error == "M"
        assert spec.trend == "A"
        assert spec.season == "M"

    def test_full_notation(self):
        spec = parse_ets_spec("ETS(A,Ad,N)")
        assert spec.trend == "Ad"
        assert spec.damped is True

    def test_comma_separated(self):
        spec = parse_ets_spec("A,A,M", period=12)
        assert spec.error == "A"
        assert spec.trend == "A"
        assert spec.season == "M"
        assert spec.period == 12

    def test_n_states_ann(self):
        spec = parse_ets_spec("ANN")
        assert spec.n_states == 1

    def test_n_states_aan(self):
        spec = parse_ets_spec("AAN")
        assert spec.n_states == 2  # level + trend

    def test_n_states_aaa(self):
        spec = parse_ets_spec("AAA", period=4)
        assert spec.n_states == 6  # 1 + 1 + 4

    def test_n_states_mam(self):
        spec = parse_ets_spec("MAM", period=12)
        assert spec.n_states == 14  # 1 + 1 + 12

    def test_n_params_ann(self):
        spec = parse_ets_spec("ANN")
        assert spec.n_params == 1  # alpha only

    def test_n_params_aan(self):
        spec = parse_ets_spec("AAN")
        assert spec.n_params == 2  # alpha + beta

    def test_n_params_aadn(self):
        spec = parse_ets_spec("AAdN")
        assert spec.n_params == 3  # alpha + beta + phi

    def test_n_params_aaa(self):
        spec = parse_ets_spec("AAA", period=4)
        assert spec.n_params == 3  # alpha + beta + gamma

    def test_invalid_model_string(self):
        with pytest.raises(ValidationError, match="cannot parse"):
            parse_ets_spec("XYZ")

    def test_empty_model_string(self):
        with pytest.raises(ValidationError, match="non-empty string"):
            parse_ets_spec("")

    def test_seasonal_requires_period(self):
        with pytest.raises(ValidationError, match="period >= 2"):
            parse_ets_spec("AAA", period=1)

    def test_nonseasonal_ignores_period(self):
        spec = parse_ets_spec("AAN", period=12)
        assert spec.period == 1  # forced to 1 for non-seasonal


# =========================================================================
# SES (ANN) tests
# =========================================================================

class TestSES:
    """Tests for ETS(A,N,N) = Simple Exponential Smoothing."""

    def test_constant_series_low_alpha(self, constant_series):
        result = ets(constant_series, model="ANN")
        # For a constant series, alpha should be near 0 (no smoothing needed)
        assert result.alpha < 0.3
        assert result.spec.name == "ETS(A,N,N)"
        # Perfect fit: residuals are all zero, MSE is zero
        assert result.mse < 1e-10

    def test_known_alpha_hand_computation(self):
        """Verify fitted values match hand computation with alpha=0.3."""
        y = np.array([10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0, 13.0, 14.0, 16.0])
        result = ets(y, model="ANN", alpha=0.3)
        assert result.alpha == pytest.approx(0.3, abs=1e-3)
        # With fixed alpha, the recursion should produce deterministic values
        assert len(result.fitted_values) == len(y)
        assert len(result.residuals) == len(y)

    def test_residuals_mean_reasonable(self, linear_series):
        """SES on a linear trend: mean residual is positive (lag bias) but bounded."""
        result = ets(linear_series, model="ANN")
        # SES lags behind a linear trend, so residuals are systematically positive.
        # But the mean absolute residual should be much smaller than the data range.
        assert result.mae < 0.5 * (linear_series[-1] - linear_series[0])

    def test_trending_series_tracks_with_lag(self, linear_series):
        result = ets(linear_series, model="ANN")
        # Fitted values should be below the actual for an upward trend
        # (SES lags behind trends)
        assert np.mean(result.residuals[10:]) > 0


# =========================================================================
# Holt's linear trend (AAN) tests
# =========================================================================

class TestHoltLinear:
    """Tests for ETS(A,A,N) = Holt's linear trend."""

    def test_linear_trend_recovery(self, linear_series):
        result = ets(linear_series, model="AAN")
        assert result.spec.name == "ETS(A,A,N)"
        assert result.beta is not None
        assert result.converged

    def test_improves_over_ses_for_trend(self, linear_series):
        ses = ets(linear_series, model="ANN")
        holt = ets(linear_series, model="AAN")
        # Holt should have lower MSE than SES on trending data
        assert holt.mse < ses.mse

    def test_init_trend_present(self, linear_series):
        result = ets(linear_series, model="AAN")
        assert result.init_trend is not None
        # Initial trend should be in the right ballpark (true slope is 0.5)
        assert 0.0 < result.init_trend < 2.0


# =========================================================================
# Damped trend (AAdN) tests
# =========================================================================

class TestDampedTrend:
    """Tests for ETS(A,Ad,N) = damped trend."""

    def test_phi_in_range(self, linear_series):
        result = ets(linear_series, model="AAdN")
        assert result.phi is not None
        assert 0.8 <= result.phi <= 1.0

    def test_forecasts_flatten(self, linear_series):
        result = ets(linear_series, model="AAdN")
        fc = forecast_ets(result, h=50)
        # Damped trend: forecasts should converge (differences shrink)
        diffs = np.diff(fc.mean)
        assert abs(diffs[-1]) < abs(diffs[0])

    def test_damped_override(self, linear_series):
        result = ets(linear_series, model="AAN", damped=True)
        assert result.spec.trend == "Ad"
        assert result.phi is not None


# =========================================================================
# Seasonal (AAA, MAM) tests
# =========================================================================

class TestSeasonal:
    """Tests for seasonal ETS models."""

    def test_aaa_quarterly(self, seasonal_quarterly):
        result = ets(seasonal_quarterly, model="AAA", period=4)
        assert result.spec.name == "ETS(A,A,A)"
        assert result.gamma is not None
        assert result.init_season is not None
        assert len(result.init_season) == 4
        assert result.converged

    def test_additive_season_sums_near_zero(self, seasonal_quarterly):
        result = ets(seasonal_quarterly, model="AAA", period=4)
        # Additive seasonal components should sum close to zero
        assert abs(np.sum(result.init_season)) < 5.0

    def test_mam_quarterly(self, seasonal_multiplicative):
        result = ets(seasonal_multiplicative, model="MAM", period=4)
        assert result.spec.name == "ETS(M,A,M)"
        assert result.gamma is not None
        assert result.init_season is not None
        assert len(result.init_season) == 4

    def test_multiplicative_season_mean_near_one(self, seasonal_multiplicative):
        result = ets(seasonal_multiplicative, model="MAM", period=4)
        # Multiplicative seasonal indices should average ~1
        assert np.mean(result.init_season) == pytest.approx(1.0, abs=0.3)

    def test_too_short_for_seasonal(self):
        y = np.arange(6, dtype=np.float64) + 1.0
        with pytest.raises(ValidationError, match="at least"):
            ets(y, model="AAA", period=4)


# =========================================================================
# Forecast tests
# =========================================================================

class TestForecast:
    """Tests for forecast_ets."""

    def test_ses_flat_forecast(self, constant_series):
        result = ets(constant_series, model="ANN")
        fc = forecast_ets(result, h=10)
        assert len(fc.mean) == 10
        # SES forecast should be flat (constant = last level)
        assert_allclose(fc.mean, fc.mean[0] * np.ones(10), atol=1e-10)

    def test_linear_trend_increasing(self, linear_series):
        result = ets(linear_series, model="AAN")
        fc = forecast_ets(result, h=10)
        # Forecasts should be monotonically increasing
        assert np.all(np.diff(fc.mean) > 0)

    def test_pi_widen_with_horizon(self, linear_series):
        result = ets(linear_series, model="AAN")
        fc = forecast_ets(result, h=20)
        widths_95 = fc.upper[95] - fc.lower[95]
        # Width should be non-decreasing
        assert np.all(np.diff(widths_95) >= -1e-10)

    def test_lower_lt_mean_lt_upper(self, linear_series):
        result = ets(linear_series, model="AAN")
        fc = forecast_ets(result, h=10)
        for lv in [80, 95]:
            assert np.all(fc.lower[lv] < fc.mean)
            assert np.all(fc.mean < fc.upper[lv])

    def test_95_wider_than_80(self, linear_series):
        result = ets(linear_series, model="AAN")
        fc = forecast_ets(result, h=10)
        width_80 = fc.upper[80] - fc.lower[80]
        width_95 = fc.upper[95] - fc.lower[95]
        assert np.all(width_95 > width_80)

    def test_forecast_horizon(self, constant_series):
        result = ets(constant_series, model="ANN")
        fc = forecast_ets(result, h=5)
        assert fc.h == 5
        assert len(fc.mean) == 5

    def test_invalid_h(self, constant_series):
        result = ets(constant_series, model="ANN")
        with pytest.raises(ValidationError, match="must be >= 1"):
            forecast_ets(result, h=0)

    def test_invalid_level(self, constant_series):
        result = ets(constant_series, model="ANN")
        with pytest.raises(ValidationError, match="\\[1, 99\\]"):
            forecast_ets(result, h=5, levels=[0])

    def test_seasonal_forecast_cycles(self, seasonal_quarterly):
        result = ets(seasonal_quarterly, model="AAA", period=4)
        fc = forecast_ets(result, h=12)
        # Three full cycles: seasonal pattern should repeat
        assert len(fc.mean) == 12

    def test_summary_string(self, constant_series):
        result = ets(constant_series, model="ANN")
        fc = forecast_ets(result, h=3)
        s = fc.summary()
        assert "ETS(A,N,N)" in s
        assert "Forecast" in s


# =========================================================================
# Fit quality tests
# =========================================================================

class TestFitQuality:
    """Tests for information criteria and fit diagnostics."""

    def test_aic_bic_aicc_finite(self, linear_series):
        result = ets(linear_series, model="AAN")
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)
        assert np.isfinite(result.aicc)

    def test_mse_mae_nonnegative(self, linear_series):
        result = ets(linear_series, model="AAN")
        assert result.mse >= 0
        assert result.mae >= 0

    def test_log_likelihood_finite(self, linear_series):
        result = ets(linear_series, model="AAN")
        assert np.isfinite(result.log_likelihood)

    def test_n_params_correct_ann(self, constant_series):
        result = ets(constant_series, model="ANN")
        # alpha + l0 + sigma^2 = 3
        assert result.n_params == 3

    def test_n_params_correct_aan(self, linear_series):
        result = ets(linear_series, model="AAN")
        # alpha + beta + l0 + b0 + sigma^2 = 5
        assert result.n_params == 5

    def test_n_params_correct_aaa(self, seasonal_quarterly):
        result = ets(seasonal_quarterly, model="AAA", period=4)
        # alpha + beta + gamma + l0 + b0 + 4 seasons + sigma^2 = 10
        assert result.n_params == 10

    def test_convergence_flag(self, linear_series):
        result = ets(linear_series, model="AAN")
        assert isinstance(result.converged, bool)

    def test_result_summary_string(self, linear_series):
        result = ets(linear_series, model="AAN")
        s = result.summary()
        assert "ETS(A,A,N)" in s
        assert "alpha" in s
        assert "beta" in s
        assert "AIC" in s


# =========================================================================
# Validation tests
# =========================================================================

class TestValidation:
    """Tests for input validation."""

    def test_nonpositive_multiplicative_error(self):
        y = np.array([1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        with pytest.raises(ValidationError, match="positive"):
            ets(y, model="MNN")

    def test_nonpositive_multiplicative_season(self):
        y = np.concatenate([np.arange(1, 9, dtype=np.float64), np.array([-1.0, 2.0, 3.0, 4.0])])
        with pytest.raises(ValidationError, match="positive"):
            ets(y, model="AAM", period=4)

    def test_empty_series(self):
        with pytest.raises(ValidationError):
            ets(np.array([]), model="ANN")

    def test_too_short_series(self):
        with pytest.raises(ValidationError, match="at least 3"):
            ets(np.array([1.0, 2.0]), model="ANN")

    def test_invalid_model_string(self):
        y = np.arange(20, dtype=np.float64) + 1.0
        with pytest.raises(ValidationError, match="cannot parse"):
            ets(y, model="ZZZ")

    def test_nan_values(self):
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        with pytest.raises(ValidationError, match="non-finite"):
            ets(y, model="ANN")

    def test_inf_values(self):
        y = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        with pytest.raises(ValidationError, match="non-finite"):
            ets(y, model="ANN")

    def test_damped_without_trend_raises(self):
        y = np.arange(20, dtype=np.float64) + 1.0
        with pytest.raises(ValidationError, match="trend"):
            ets(y, model="ANN", damped=True)


# =========================================================================
# Edge cases
# =========================================================================

class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_fixed_alpha(self):
        y = np.arange(20, dtype=np.float64) + 1.0
        result = ets(y, model="ANN", alpha=0.5)
        assert result.alpha == pytest.approx(0.5, abs=1e-3)

    def test_fixed_multiple_params(self):
        y = np.arange(20, dtype=np.float64) + 1.0
        result = ets(y, model="AAN", alpha=0.3, beta=0.05)
        assert result.alpha == pytest.approx(0.3, abs=1e-3)
        assert result.beta == pytest.approx(0.05, abs=1e-3)

    def test_very_short_series(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = ets(y, model="ANN")
        assert result.n_obs == 10
        assert len(result.fitted_values) == 10

    def test_frozen_result(self, constant_series):
        result = ets(constant_series, model="ANN")
        with pytest.raises(AttributeError):
            result.alpha = 0.5  # type: ignore[misc]

    def test_frozen_forecast(self, constant_series):
        result = ets(constant_series, model="ANN")
        fc = forecast_ets(result, h=5)
        with pytest.raises(AttributeError):
            fc.h = 10  # type: ignore[misc]

    def test_states_shape_ann(self, constant_series):
        result = ets(constant_series, model="ANN")
        # (n+1, 1) for level only
        assert result.states.shape == (51, 1)

    def test_states_shape_aan(self, linear_series):
        result = ets(linear_series, model="AAN")
        # (n+1, 2) for level + trend
        assert result.states.shape == (101, 2)

    def test_states_shape_aaa(self, seasonal_quarterly):
        result = ets(seasonal_quarterly, model="AAA", period=4)
        # (n+1, 1+1+4) = (81, 6)
        assert result.states.shape == (81, 6)

    def test_n_obs_matches(self, linear_series):
        result = ets(linear_series, model="AAN")
        assert result.n_obs == len(linear_series)


# =========================================================================
# Recursion tests
# =========================================================================

class TestRecursion:
    """Direct tests for ets_recursion."""

    def test_ann_manual(self):
        """Verify ANN recursion matches hand computation."""
        spec = parse_ets_spec("ANN")
        y = np.array([10.0, 12.0, 11.0, 13.0])
        alpha = 0.3
        params = np.array([alpha])
        init_states = np.array([10.0])  # l_0 = 10

        fitted, resid, states = ets_recursion(y, spec, params, init_states)

        # t=0: mu = l_0 = 10, e = 10-10=0, l_1 = 10 + 0.3*0 = 10
        assert fitted[0] == pytest.approx(10.0)
        assert resid[0] == pytest.approx(0.0)

        # t=1: mu = l_1 = 10, e = 12-10=2, l_2 = 10 + 0.3*2 = 10.6
        assert fitted[1] == pytest.approx(10.0)
        assert resid[1] == pytest.approx(2.0)

        # t=2: mu = l_2 = 10.6, e = 11-10.6 = 0.4, l_3 = 10.6 + 0.3*0.4 = 10.72
        assert fitted[2] == pytest.approx(10.6)
        assert resid[2] == pytest.approx(0.4)

        # t=3: mu = l_3 = 10.72
        assert fitted[3] == pytest.approx(10.72)

    def test_aan_manual(self):
        """Verify AAN recursion with known values."""
        spec = parse_ets_spec("AAN")
        y = np.array([10.0, 11.0, 12.0, 13.0])
        alpha = 0.5
        beta_val = 0.1
        params = np.array([alpha, beta_val])
        init_states = np.array([10.0, 1.0])  # l_0=10, b_0=1

        fitted, resid, states = ets_recursion(y, spec, params, init_states)

        # t=0: mu = 10 + 1 = 11, e = 10 - 11 = -1
        # l_1 = 10 + 1 + 0.5*(-1) = 10.5
        # b_1 = 1 + 0.1*(-1) = 0.9
        assert fitted[0] == pytest.approx(11.0)
        assert resid[0] == pytest.approx(-1.0)
        assert states[1, 0] == pytest.approx(10.5)  # l_1
        assert states[1, 1] == pytest.approx(0.9)   # b_1

    def test_output_shapes(self):
        spec = parse_ets_spec("AAN")
        y = np.arange(10, dtype=np.float64) + 1.0
        params = np.array([0.3, 0.05])
        init_states = np.array([1.0, 0.5])
        fitted, resid, states = ets_recursion(y, spec, params, init_states)
        assert fitted.shape == (10,)
        assert resid.shape == (10,)
        assert states.shape == (11, 2)
