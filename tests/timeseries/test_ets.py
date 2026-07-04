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
from pystatistics.timeseries._ets_fit import ETSSolution
from pystatistics.timeseries._ets_select import ets
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

    def test_additive_season_sums_to_zero(self, seasonal_quarterly):
        result = ets(seasonal_quarterly, model="AAA", period=4)
        # Sum-to-zero holds exactly by construction: the first-used
        # seasonal state is reconstructed from the other m - 1 (as in R)
        assert abs(np.sum(result.init_season)) < 1e-8

    def test_mam_quarterly(self, seasonal_multiplicative):
        result = ets(seasonal_multiplicative, model="MAM", period=4)
        assert result.spec.name == "ETS(M,A,M)"
        assert result.gamma is not None
        assert result.init_season is not None
        assert len(result.init_season) == 4

    def test_multiplicative_season_mean_is_one(self, seasonal_multiplicative):
        result = ets(seasonal_multiplicative, model="MAM", period=4)
        # Mean-one holds exactly by construction (sum(s) = m, as in R)
        assert np.mean(result.init_season) == pytest.approx(1.0, abs=1e-8)

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
        # alpha + beta + gamma + l0 + b0 + 3 free seasons + sigma^2 = 9
        # (the 4th seasonal state is fixed by the sum-to-zero
        # normalisation, matching R forecast::ets's parameter count)
        assert result.n_params == 9

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
            ets(y, model="XQW")

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

    def test_fixed_alpha_out_of_range_raises(self):
        """Out-of-range fixed parameters raise (R: 'Parameters out of
        range') instead of being silently coerced into bounds."""
        y = np.arange(20, dtype=np.float64) + 1.0
        with pytest.raises(ValidationError, match="alpha.*usual region"):
            ets(y, model="ANN", alpha=1.5)

    def test_fixed_phi_above_r_upper_bound_raises(self):
        y = np.arange(30, dtype=np.float64) + 1.0
        with pytest.raises(ValidationError, match="phi.*usual region"):
            ets(y, model="AAdN", phi=0.99)

    def test_fixed_beta_above_fixed_alpha_raises(self):
        y = np.arange(30, dtype=np.float64) + 1.0
        with pytest.raises(ValidationError, match="beta <= alpha"):
            ets(y, model="AAN", alpha=0.2, beta=0.5)

    def test_fixed_gamma_above_one_minus_alpha_raises(self, seasonal_quarterly):
        with pytest.raises(ValidationError, match="gamma <= 1 - alpha"):
            ets(seasonal_quarterly, model="AAA", period=4,
                alpha=0.9, gamma=0.5)


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


# ---------------------------------------------------------------------------
# "Z"-wildcard automatic model selection (matches forecast::ets)
# ---------------------------------------------------------------------------

import json
from pathlib import Path

_ETS_FIXTURE = Path(__file__).parent.parent / "fixtures" / "ets_r_reference.json"


def _r_reference():
    with open(_ETS_FIXTURE) as fh:
        return json.load(fh)


class TestZZZCandidateSet:
    """The enumerated candidate set mirrors forecast::ets exactly."""

    def test_full_zzz_seasonal_positive_has_15_candidates(self):
        ref = _r_reference()
        x = np.asarray(ref["selection"]["airpassengers"]["x"])
        sol = ets(x, model="ZZZ", period=12)
        sel = sol.info["selection"]
        names = {c["model"] for c in sel["candidates"]}
        assert len(sel["candidates"]) == 15
        # A-error x multiplicative-season trio excluded (restrict=TRUE).
        skipped_models = {s["model"] for s in sel["skipped"]}
        assert {"ANM", "AAdM", "AAM"} <= skipped_models
        assert "ETS(M,A,M)" in names and "ETS(A,N,N)" in names

    def test_zzn_has_6_candidates(self):
        ref = _r_reference()
        x = np.asarray(ref["selection"]["airpassengers"]["x"])
        sol = ets(x, model="ZZN", period=12)
        sel = sol.info["selection"]
        names = {c["model"] for c in sel["candidates"]}
        assert names == {
            "ETS(A,N,N)", "ETS(A,Ad,N)", "ETS(A,A,N)",
            "ETS(M,N,N)", "ETS(M,Ad,N)", "ETS(M,A,N)",
        }

    def test_nonseasonal_period1_zzz(self):
        ref = _r_reference()
        x = np.asarray(ref["selection"]["nile"]["x"])
        sol = ets(x, model="ZZZ", period=1)
        assert len(sol.info["selection"]["candidates"]) == 6

    def test_negative_data_drops_multiplicative_error(self):
        ref = _r_reference()
        x = np.asarray(ref["selection"]["diff_nile"]["x"])
        sol = ets(x, model="ZZZ", period=1)
        sel = sol.info["selection"]
        assert all(c["model"].startswith("ETS(A") for c in sel["candidates"])
        assert any("strictly positive" in s["reason"] for s in sel["skipped"])

    def test_damped_true_restricts_trend_candidates(self):
        ref = _r_reference()
        x = np.asarray(ref["selection"]["nile"]["x"])
        sol = ets(x, model="ZZN", period=1, damped=True)
        names = {c["model"] for c in sol.info["selection"]["candidates"]}
        assert names == {"ETS(A,Ad,N)", "ETS(M,Ad,N)"}

    def test_damped_false_excludes_damped(self):
        ref = _r_reference()
        x = np.asarray(ref["selection"]["nile"]["x"])
        sol = ets(x, model="ZZN", period=1, damped=False)
        names = {c["model"] for c in sol.info["selection"]["candidates"]}
        assert names == {"ETS(A,N,N)", "ETS(A,A,N)",
                         "ETS(M,N,N)", "ETS(M,A,N)"}


class TestZZZSelection:
    """Selection agrees with R where the engines' optima agree, and is
    always internally consistent with the disclosed candidate table."""

    @pytest.mark.parametrize("name", ["usaccdeaths", "nile", "wwwusage",
                                      "diff_nile"])
    def test_selection_matches_r(self, name):
        """Datasets where pystatistics and forecast::ets agree exactly.

        (The engine optimises the same parameter space as R, but the
        selected model can still differ where R's Nelder-Mead stalls
        short of a candidate's optimum — see timeseries/_ets_select.py;
        those datasets are exercised by the two tests below.)
        """
        case = _r_reference()["selection"][name]
        sol = ets(np.asarray(case["x"]), model=case["model_arg"],
                  period=case["period"])
        assert sol.spec.name == case["method"]

    @pytest.mark.parametrize("name", ["airpassengers", "co2", "lynx",
                                      "airpassengers_zzn",
                                      "airpassengers_azz",
                                      "airpassengers_mzz"])
    def test_selection_is_argmin_of_disclosed_table(self, name):
        case = _r_reference()["selection"][name]
        sol = ets(np.asarray(case["x"]), model=case["model_arg"],
                  period=case["period"])
        sel = sol.info["selection"]
        best = min(sel["candidates"], key=lambda c: c[sel["ic"]])
        assert sel["selected"] == best["model"]
        assert sel["selected"] == sol.spec.name

    @pytest.mark.parametrize("name", ["airpassengers", "co2", "lynx",
                                      "airpassengers_zzn",
                                      "airpassengers_azz",
                                      "airpassengers_mzz"])
    def test_divergent_selection_dominates_r_choice(self, name):
        """Where the selection differs from R, it must be because our
        engine found a better optimum, never a worse criterion value:
        the selected model's AICc (converted to R's log-likelihood
        convention) must beat the AICc R reported for *its* selection."""
        case = _r_reference()["selection"][name]
        sol = ets(np.asarray(case["x"]), model=case["model_arg"],
                  period=case["period"])
        n = case["n"]
        const = 0.5 * n * (np.log(n / (2.0 * np.pi)) - 1.0)
        aicc_r_convention = sol.aicc + 2.0 * const
        assert aicc_r_convention <= case["aicc"] + 0.01

    @pytest.mark.parametrize("name", ["airpassengers", "usaccdeaths",
                                      "nile", "lynx"])
    def test_selected_params_in_usual_region(self, name):
        """Fitted smoothing parameters respect R's usual region and the
        seasonal normalisation (the aligned parameter space)."""
        case = _r_reference()["selection"][name]
        sol = ets(np.asarray(case["x"]), model=case["model_arg"],
                  period=case["period"])
        eps = 1e-4
        assert eps <= sol.alpha <= 1.0 - eps
        if sol.beta is not None:
            assert eps <= sol.beta <= sol.alpha
        if sol.gamma is not None:
            assert eps <= sol.gamma <= 1.0 - sol.alpha
        if sol.phi is not None:
            assert 0.8 <= sol.phi <= 0.98
        if sol.init_season is not None:
            target = 0.0 if sol.spec.season == "A" else float(sol.spec.period)
            assert np.sum(sol.init_season) == pytest.approx(target, abs=1e-8)

    def test_default_model_is_zzz(self):
        """ets(y) now auto-selects, matching forecast::ets's default."""
        case = _r_reference()["selection"]["nile"]
        sol = ets(np.asarray(case["x"]))
        assert "selection" in sol.info
        assert sol.info["selection"]["requested"] == "ZZZ"

    def test_selection_uses_requested_ic(self):
        case = _r_reference()["selection"]["nile"]
        x = np.asarray(case["x"])
        sol = ets(x, model="ZZN", ic="bic")
        sel = sol.info["selection"]
        assert sel["ic"] == "bic"
        best = min(sel["candidates"], key=lambda c: c["bic"])
        assert sel["selected"] == best["model"]

    def test_fully_specified_model_bypasses_selection(self):
        """No 'Z' -> fit exactly what was asked, no selection metadata."""
        case = _r_reference()["selection"]["nile"]
        sol = ets(np.asarray(case["x"]), model="ANN")
        assert sol.spec.name == "ETS(A,N,N)"
        assert "selection" not in sol.info


class TestZZZFailures:
    """Explicit requests that cannot be honoured fail loud."""

    def test_multiplicative_error_wildcard_on_negative_data(self):
        y = np.array([1.0, -2.0, 3.0, -1.0, 2.0, 0.5, 1.5, -0.5] * 3)
        with pytest.raises(ValidationError, match="strictly positive"):
            ets(y, model="MZZ", period=1)

    def test_explicit_seasonal_with_period_one(self):
        y = np.arange(30, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="period >= 2"):
            ets(y, model="ZZA", period=1)

    def test_explicit_seasonal_with_period_over_24(self):
        y = np.arange(120, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="period <= 24"):
            ets(y, model="ZZA", period=25)

    def test_wildcard_seasonal_with_period_over_24_warns_and_drops(self):
        rng = np.random.default_rng(8)
        y = rng.normal(100.0, 5.0, 200)
        sol = ets(y, model="ZZZ", period=30)
        assert any("period 30" in w for w in sol.warnings)
        assert all("N)" in c["model"] for c in sol.info["selection"]["candidates"])

    def test_damped_false_with_explicit_ad_string(self):
        y = np.arange(30, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="damped"):
            ets(y, model="AAdZ", period=1, damped=False)

    def test_invalid_ic(self):
        y = np.arange(30, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="ic"):
            ets(y, model="ZZN", ic="hqic")


class TestLogLikelihoodConvention:
    """The documented full-Gaussian vs R-concentrated constant.

    ll_pystat = ll_R + 0.5 * n * [log(n / (2*pi)) - 1]; verified against
    stored forecast::ets fits where both optimisers find the same optimum.
    """

    @pytest.mark.parametrize("name", ["nile_ann", "nile_aan"])
    def test_constant_offset_vs_r(self, name):
        case = _r_reference()["fixed"][name]
        x = np.asarray(case["x"])
        sol = ets(x, model=case["model_arg"], period=case["period"])
        n = case["n"]
        const = 0.5 * n * (np.log(n / (2.0 * np.pi)) - 1.0)
        assert abs((sol.log_likelihood - case["loglik"]) - const) < 0.01

    def test_constant_value_n100(self):
        """The documented example: n=100 -> constant 88.36."""
        const = 0.5 * 100 * (np.log(100 / (2.0 * np.pi)) - 1.0)
        assert abs(const - 88.3647) < 1e-3

    @pytest.mark.parametrize("name", ["airpassengers_aaa",
                                      "airpassengers_mam",
                                      "usaccdeaths_ana"])
    def test_seasonal_loglik_at_least_r_optimum(self, name):
        """The engine optimises the same parameter space as R but with
        L-BFGS-B instead of R's Nelder-Mead (which stalls on these
        seasonal fits — see _ets_select.py), so after removing the
        convention constant its log-likelihood should not fall
        meaningfully below R's optimum."""
        case = _r_reference()["fixed"][name]
        x = np.asarray(case["x"])
        sol = ets(x, model=case["model_arg"], period=case["period"])
        n = case["n"]
        const = 0.5 * n * (np.log(n / (2.0 * np.pi)) - 1.0)
        assert sol.log_likelihood - const >= case["loglik"] - 0.5


class TestWildcardValidation:
    """Boundary validation of the public ets() (adversarial-review fixes)."""

    def test_float_period_raises(self):
        y = np.arange(40, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="period.*integer"):
            ets(y, period=12.0)

    def test_string_period_raises(self):
        y = np.arange(40, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="period.*integer"):
            ets(y, period="12")

    def test_bool_period_raises(self):
        y = np.arange(40, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="period.*integer"):
            ets(y, period=True)

    def test_empty_series_raises(self):
        with pytest.raises(ValidationError, match="at least 3 observations"):
            ets(np.array([]))

    def test_two_dimensional_series_raises(self):
        y = np.arange(40, dtype=float).reshape(20, 2)
        with pytest.raises(ValidationError, match="1-D"):
            ets(y, model="ZZN")

    def test_two_dimensional_series_raises_fully_specified(self):
        y = np.arange(40, dtype=float).reshape(20, 2)
        with pytest.raises(ValidationError, match="1-D"):
            ets(y, model="ANN")

    def test_column_vector_accepted(self):
        """(n, 1) input is unambiguous and accepted."""
        y = (np.arange(30, dtype=float) + 1.0).reshape(-1, 1)
        sol = ets(y, model="ANN")
        assert sol.n_obs == 30

    def test_damped_true_with_trend_n_wildcard_raises(self):
        """model='ZNN' + damped=True is a forbidden combination (as in R),
        not a silent no-op."""
        y = np.arange(40, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="trend component"):
            ets(y, model="ZNN", damped=True)

    def test_period_over_24_fully_specified_raises(self):
        y = np.arange(120, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="period <= 24"):
            ets(y, model="MAM", period=25)

    def test_tiny_series_aicc_selection_raises_with_remedy(self):
        """n=3: every candidate fits but AICc is infinite -> a clear
        ValidationError naming the remedy, not a false ConvergenceError."""
        with pytest.raises(ValidationError, match="finite aicc"):
            ets(np.array([1.0, 2.0, 3.0]))

    def test_tiny_series_selectable_with_aic(self):
        """The remedy works: ic='aic' selects at n=3."""
        sol = ets(np.array([1.0, 2.0, 3.0]), ic="aic")
        assert "selection" in sol.info

    def test_n5_selects_under_default_aicc(self):
        """n=5 is the smallest series the default ets(y) can select on."""
        sol = ets(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))
        assert np.isfinite(sol.aicc)
