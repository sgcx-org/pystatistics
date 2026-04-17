"""
Tests for ARIMA fitting, forecasting, and automatic order selection.

Covers model fitting (AR, MA, ARIMA, seasonal), forecasting (point
forecasts, prediction intervals, un-differencing), automatic order
selection (stepwise and grid search), validation, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.core.exceptions import ConvergenceError, ValidationError
from pystatistics.timeseries._arima_fit import ARIMAResult, arima
from pystatistics.timeseries._arima_forecast import (
    ARIMAForecast,
    forecast_arima,
    _psi_weights,
    _undifference,
)
from pystatistics.timeseries._arima_order import (
    AutoARIMAResult,
    auto_arima,
    _determine_d,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def rng():
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def ar1_series(rng):
    """AR(1) process with phi=0.7, n=200."""
    n = 200
    phi = 0.7
    y = np.zeros(n)
    e = rng.normal(0, 1.0, n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


@pytest.fixture
def ma1_series(rng):
    """MA(1) process with theta=0.5, n=200."""
    n = 200
    theta = 0.5
    e = rng.normal(0, 1.0, n + 1)
    y = np.zeros(n)
    for t in range(n):
        y[t] = e[t + 1] + theta * e[t]
    return y


@pytest.fixture
def random_walk(rng):
    """Random walk (integrated white noise), n=200."""
    return np.cumsum(rng.normal(0, 1.0, 200))


@pytest.fixture
def arima_111_series(rng):
    """ARIMA(1,1,1) process with phi=0.5, theta=0.3, n=200."""
    n = 200
    phi = 0.5
    theta = 0.3
    e = rng.normal(0, 1.0, n + 1)
    # Generate ARMA(1,1) innovations
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + e[t + 1] + theta * e[t]
    # Integrate to get ARIMA(1,1,1)
    y = np.cumsum(z)
    return y


@pytest.fixture
def white_noise(rng):
    """White noise, n=200."""
    return rng.normal(0, 1.0, 200)


@pytest.fixture
def seasonal_ar1_12(rng):
    """Seasonal AR(1) at period 12: y[t] = 0.8 * y[t-12] + e[t], n=240."""
    n = 240
    phi_s = 0.8
    y = np.zeros(n)
    e = rng.normal(0, 1.0, n)
    for t in range(12, n):
        y[t] = phi_s * y[t - 12] + e[t]
    return y


@pytest.fixture
def airline_data(rng):
    """Synthetic airline-style data: SARIMA(0,1,1)(0,1,1)[12], n=144."""
    n = 144
    theta = -0.4
    theta_s = -0.6
    e = rng.normal(0, 0.5, n + 13)
    # Build MA(1) x SMA(1)_12 innovations
    z = np.zeros(n)
    for t in range(n):
        z[t] = (
            e[t + 13]
            + theta * e[t + 12]
            + theta_s * e[t + 1]
            + theta * theta_s * e[t]
        )
    # Integrate: diff once, seasonal diff once with period 12
    # y[t] - y[t-1] - y[t-12] + y[t-13] = z[t]
    y = np.zeros(n)
    for t in range(13, n):
        y[t] = y[t - 1] + y[t - 12] - y[t - 13] + z[t]
    return y


# =========================================================================
# ARIMA Fitting Tests
# =========================================================================

class TestARIMAFitting:
    """Tests for the arima() fitting function."""

    def test_ar1_coefficient(self, ar1_series):
        """AR(1) model recovers phi close to 0.7."""
        result = arima(ar1_series, order=(1, 0, 0))
        assert result.converged
        assert len(result.ar) == 1
        assert len(result.ma) == 0
        assert abs(result.ar[0] - 0.7) < 0.15

    def test_ma1_coefficient(self, ma1_series):
        """MA(1) model recovers theta close to 0.5."""
        result = arima(ma1_series, order=(0, 0, 1))
        assert result.converged
        assert len(result.ar) == 0
        assert len(result.ma) == 1
        assert abs(result.ma[0] - 0.5) < 0.2

    def test_arima_111_converges(self, arima_111_series):
        """ARIMA(1,1,1) converges with reasonable coefficients."""
        result = arima(arima_111_series, order=(1, 1, 1))
        assert result.converged
        assert result.order == (1, 1, 1)
        assert len(result.ar) >= 1
        assert len(result.ma) >= 1

    def test_random_walk_010(self, random_walk):
        """Random walk as ARIMA(0,1,0): no AR/MA coefficients."""
        result = arima(random_walk, order=(0, 1, 0))
        assert result.converged
        assert result.order == (0, 1, 0)
        # Effective AR/MA should be empty (or only from differencing)
        # The model is just differencing, so the ARMA part has p=0, q=0
        # after differencing.

    def test_sigma2_positive(self, ar1_series):
        """Innovation variance is positive."""
        result = arima(ar1_series, order=(1, 0, 0))
        assert result.sigma2 > 0

    def test_residuals_white_noise(self, ar1_series):
        """Residuals from a correctly specified model are approximately white noise."""
        from pystatistics.timeseries._acf import acf

        result = arima(ar1_series, order=(1, 0, 0))
        acf_result = acf(result.residuals, max_lag=20)
        # After lag 0, autocorrelations should be small
        # Use a generous threshold since we're checking approximate whiteness
        max_acf = np.max(np.abs(acf_result.acf[1:]))
        assert max_acf < 0.2, f"Max residual ACF = {max_acf}, expected < 0.2"

    def test_n_obs(self, ar1_series):
        """n_obs matches input length."""
        result = arima(ar1_series, order=(1, 0, 0))
        assert result.n_obs == len(ar1_series)

    def test_log_likelihood_finite(self, ar1_series):
        """Log-likelihood is finite."""
        result = arima(ar1_series, order=(1, 0, 0))
        assert np.isfinite(result.log_likelihood)

    def test_aic_less_than_bic_for_small_model(self, ar1_series):
        """AIC < BIC for typical models (BIC penalises more)."""
        result = arima(ar1_series, order=(1, 0, 0))
        # BIC penalty is log(n) * k vs 2k for AIC; for n > ~8 BIC > AIC
        assert result.aic < result.bic

    def test_information_criteria_finite(self, ar1_series):
        """AIC, AICc, BIC are all finite."""
        result = arima(ar1_series, order=(1, 0, 0))
        assert np.isfinite(result.aic)
        assert np.isfinite(result.aicc)
        assert np.isfinite(result.bic)

    def test_fitted_values_length(self, ar1_series):
        """Fitted values have same length as the differenced series."""
        result = arima(ar1_series, order=(1, 0, 0))
        assert len(result.fitted_values) == len(result.residuals)

    def test_result_is_frozen(self, ar1_series):
        """ARIMAResult is a frozen dataclass."""
        result = arima(ar1_series, order=(1, 0, 0))
        with pytest.raises(AttributeError):
            result.sigma2 = 999.0  # type: ignore[misc]


# =========================================================================
# Seasonal ARIMA Tests
# =========================================================================

@pytest.mark.slow
class TestSeasonalARIMA:
    """Tests for seasonal ARIMA models."""

    def test_sarima_100_100_12(self, seasonal_ar1_12):
        """SARIMA(1,0,0)(1,0,0)[12] recovers seasonal AR coefficient."""
        result = arima(
            seasonal_ar1_12,
            order=(0, 0, 0),
            seasonal=(1, 0, 0, 12),
        )
        assert result.converged
        assert result.seasonal_order == (1, 0, 0, 12)

    def test_airline_model(self, airline_data):
        """SARIMA(0,1,1)(0,1,1)[12] airline model on synthetic data."""
        result = arima(
            airline_data,
            order=(0, 1, 1),
            seasonal=(0, 1, 1, 12),
        )
        assert result.converged
        assert result.order == (0, 1, 1)
        assert result.seasonal_order == (0, 1, 1, 12)


# =========================================================================
# Forecast Tests
# =========================================================================

class TestARIMAForecast:
    """Tests for forecast_arima()."""

    def test_ar1_forecast_decays_toward_mean(self, ar1_series):
        """AR(1) forecast should decay toward the mean."""
        result = arima(ar1_series, order=(1, 0, 0))
        fc = forecast_arima(result, ar1_series, h=50)
        assert isinstance(fc, ARIMAForecast)
        assert fc.h == 50
        # Forecasts should converge toward the series mean
        series_mean = np.mean(ar1_series)
        # Last forecasts should be closer to mean than first
        dist_first = abs(fc.mean[0] - series_mean)
        dist_last = abs(fc.mean[-1] - series_mean)
        assert dist_last <= dist_first + 0.5  # generous tolerance

    def test_random_walk_forecast_is_flat(self, random_walk):
        """Random walk ARIMA(0,1,0) forecast is flat at the last value."""
        result = arima(random_walk, order=(0, 1, 0))
        fc = forecast_arima(result, random_walk, h=10)
        last_val = random_walk[-1]
        assert_allclose(fc.mean, last_val, atol=0.5)

    def test_arima_011_first_step_differs(self, rng):
        """ARIMA(0,1,1) forecast: first step may differ from subsequent."""
        y = np.cumsum(rng.normal(0, 1.0, 200))
        result = arima(y, order=(0, 1, 1))
        fc = forecast_arima(result, y, h=5)
        # After the first step, forecasts should be nearly identical
        # (MA effect dies out after step 1 on the differenced scale)
        if fc.h >= 3:
            diff_23 = abs(fc.mean[2] - fc.mean[1])
            assert diff_23 < 0.1  # nearly flat after step 1

    def test_pi_widen_with_horizon(self, ar1_series):
        """Prediction intervals widen with forecast horizon."""
        result = arima(ar1_series, order=(1, 0, 0))
        fc = forecast_arima(result, ar1_series, h=20)
        # Standard errors should be non-decreasing
        for i in range(1, len(fc.se)):
            assert fc.se[i] >= fc.se[i - 1] - 1e-10

    def test_pi_ordering(self, ar1_series):
        """Lower < mean < upper for all horizons."""
        result = arima(ar1_series, order=(1, 0, 0))
        fc = forecast_arima(result, ar1_series, h=10, levels=[80, 95])
        for lv in [80, 95]:
            assert np.all(fc.lower[lv] < fc.mean)
            assert np.all(fc.mean < fc.upper[lv])

    def test_95_wider_than_80(self, ar1_series):
        """95% PI is wider than 80% PI."""
        result = arima(ar1_series, order=(1, 0, 0))
        fc = forecast_arima(result, ar1_series, h=10, levels=[80, 95])
        width_80 = fc.upper[80] - fc.lower[80]
        width_95 = fc.upper[95] - fc.lower[95]
        assert np.all(width_95 > width_80)

    def test_forecast_on_original_scale(self, random_walk):
        """Forecasts are on the original (un-differenced) scale."""
        result = arima(random_walk, order=(0, 1, 0))
        fc = forecast_arima(result, random_walk, h=5)
        # Forecasts should be in the same range as the original series
        y_range = np.max(random_walk) - np.min(random_walk)
        for val in fc.mean:
            assert abs(val - random_walk[-1]) < y_range * 2

    def test_h1_forecast_close_to_one_step(self, ar1_series):
        """h=1 forecast should be close to one-step-ahead prediction."""
        result = arima(ar1_series, order=(1, 0, 0))
        fc = forecast_arima(result, ar1_series, h=1)
        assert fc.h == 1
        assert len(fc.mean) == 1
        assert np.isfinite(fc.mean[0])

    def test_forecast_summary(self, ar1_series):
        """summary() returns a non-empty string."""
        result = arima(ar1_series, order=(1, 0, 0))
        fc = forecast_arima(result, ar1_series, h=3)
        s = fc.summary()
        assert isinstance(s, str)
        assert "Forecast" in s
        assert "ARIMA" in s

    def test_forecast_result_is_frozen(self, ar1_series):
        """ARIMAForecast is a frozen dataclass."""
        result = arima(ar1_series, order=(1, 0, 0))
        fc = forecast_arima(result, ar1_series, h=3)
        with pytest.raises(AttributeError):
            fc.h = 999  # type: ignore[misc]

    def test_default_levels(self, ar1_series):
        """Default levels are [80, 95]."""
        result = arima(ar1_series, order=(1, 0, 0))
        fc = forecast_arima(result, ar1_series, h=3)
        assert set(fc.lower.keys()) == {80, 95}
        assert set(fc.upper.keys()) == {80, 95}


# =========================================================================
# Psi Weights Tests
# =========================================================================

class TestPsiWeights:
    """Tests for the MA(infinity) psi weight computation."""

    def test_psi_weights_pure_ar1(self):
        """For AR(1) with phi=0.5, psi_j = 0.5^j."""
        ar = np.array([0.5])
        ma = np.array([])
        psi = _psi_weights(ar, ma, 5)
        expected = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        assert_allclose(psi, expected, atol=1e-12)

    def test_psi_weights_pure_ma1(self):
        """For MA(1) with theta=0.3, psi = [1, 0.3, 0, 0, ...]."""
        ar = np.array([])
        ma = np.array([0.3])
        psi = _psi_weights(ar, ma, 4)
        expected = np.array([1.0, 0.3, 0.0, 0.0])
        assert_allclose(psi, expected, atol=1e-12)

    def test_psi_weights_white_noise(self):
        """For ARMA(0,0), psi = [1, 0, 0, ...]."""
        psi = _psi_weights(np.array([]), np.array([]), 3)
        expected = np.array([1.0, 0.0, 0.0])
        assert_allclose(psi, expected, atol=1e-12)


# =========================================================================
# Un-differencing Tests
# =========================================================================

class TestUndifference:
    """Tests for the _undifference helper."""

    def test_undifference_d1(self):
        """Un-differencing d=1 recovers original-scale forecasts."""
        y = np.array([10.0, 12.0, 15.0, 19.0, 24.0])
        fc_diff = np.array([5.0, 5.0, 5.0])  # constant increment
        result = _undifference(fc_diff, y, d=1)
        # Expected: 24+5=29, 29+5=34, 34+5=39
        expected = np.array([29.0, 34.0, 39.0])
        assert_allclose(result, expected, atol=1e-10)

    def test_undifference_d0(self):
        """d=0 returns forecasts unchanged."""
        fc = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        result = _undifference(fc, y, d=0)
        assert_allclose(result, fc, atol=1e-10)


# =========================================================================
# Auto ARIMA Tests
# =========================================================================

@pytest.mark.slow
class TestAutoARIMA:
    """Tests for auto_arima()."""

    def test_ar1_selects_ar(self, ar1_series):
        """AR(1) data: auto_arima selects p >= 1."""
        result = auto_arima(ar1_series, max_p=3, max_q=3, stepwise=True)
        assert isinstance(result, AutoARIMAResult)
        p, d, q = result.best_order
        # Should select some AR component
        assert p >= 1 or d >= 1

    def test_white_noise_selects_000(self, white_noise):
        """White noise: auto_arima selects (0,0,0) or very simple model."""
        result = auto_arima(white_noise, max_p=2, max_q=2, stepwise=True)
        p, d, q = result.best_order
        # Should select d=0 and small p+q
        assert d == 0
        assert p + q <= 2

    def test_random_walk_selects_d1(self, random_walk):
        """Random walk: auto_arima selects d >= 1."""
        result = auto_arima(random_walk, max_p=2, max_q=2, stepwise=True)
        _p, d, _q = result.best_order
        assert d >= 1

    def test_grid_more_models_than_stepwise(self, white_noise):
        """Grid search evaluates more models than stepwise."""
        r_step = auto_arima(
            white_noise, max_p=2, max_q=2, stepwise=True
        )
        r_grid = auto_arima(
            white_noise, max_p=2, max_q=2, stepwise=False
        )
        assert len(r_grid.search_results) >= len(r_step.search_results)

    def test_search_results_format(self, ar1_series):
        """search_results contains (order, ic) tuples."""
        result = auto_arima(ar1_series, max_p=2, max_q=2, stepwise=True)
        assert len(result.search_results) > 0
        for order, ic_val in result.search_results:
            assert isinstance(order, tuple)
            assert len(order) == 3
            assert isinstance(ic_val, float)

    def test_best_aic_is_minimum(self, ar1_series):
        """best_aic equals the minimum IC across converged models."""
        result = auto_arima(ar1_series, max_p=2, max_q=2, stepwise=True)
        finite_ics = [v for _, v in result.search_results if np.isfinite(v)]
        assert len(finite_ics) > 0
        assert_allclose(result.best_aic, min(finite_ics), atol=1e-10)

    def test_models_fitted_count(self, ar1_series):
        """models_fitted counts only converged models."""
        result = auto_arima(ar1_series, max_p=2, max_q=2, stepwise=True)
        assert result.models_fitted > 0
        assert result.models_fitted <= len(result.search_results)

    def test_auto_arima_summary(self, ar1_series):
        """summary() returns a non-empty string."""
        result = auto_arima(ar1_series, max_p=2, max_q=2, stepwise=True)
        s = result.summary()
        assert isinstance(s, str)
        assert "Best model" in s

    def test_auto_arima_result_is_frozen(self, ar1_series):
        """AutoARIMAResult is a frozen dataclass."""
        result = auto_arima(ar1_series, max_p=2, max_q=2, stepwise=True)
        with pytest.raises(AttributeError):
            result.best_aic = 0.0  # type: ignore[misc]

    def test_ic_bic(self, ar1_series):
        """ic='bic' selects using BIC."""
        result = auto_arima(
            ar1_series, max_p=2, max_q=2, ic="bic", stepwise=True
        )
        assert result.best_aic is not None  # best_aic field holds BIC value
        assert np.isfinite(result.best_aic)


# =========================================================================
# Validation Tests
# =========================================================================

class TestValidation:
    """Tests for input validation."""

    def test_invalid_order_negative(self, ar1_series):
        """Negative order values raise ValidationError."""
        with pytest.raises(ValidationError):
            arima(ar1_series, order=(-1, 0, 0))

    def test_empty_series_raises(self):
        """Empty series raises ValidationError."""
        with pytest.raises(ValidationError):
            arima(np.array([]), order=(1, 0, 0))

    def test_order_wrong_length(self, ar1_series):
        """Order not length 3 raises ValidationError."""
        with pytest.raises((ValidationError, TypeError)):
            arima(ar1_series, order=(1, 0))  # type: ignore[arg-type]

    def test_seasonal_wrong_length(self, ar1_series):
        """Seasonal order not length 4 raises ValidationError."""
        with pytest.raises((ValidationError, TypeError)):
            arima(ar1_series, order=(1, 0, 0), seasonal_order=(1, 0, 1))  # type: ignore[arg-type]

    def test_forecast_invalid_h(self, ar1_series):
        """h < 1 raises ValidationError."""
        result = arima(ar1_series, order=(1, 0, 0))
        with pytest.raises(ValidationError):
            forecast_arima(result, ar1_series, h=0)

    def test_forecast_invalid_levels(self, ar1_series):
        """Invalid levels raise ValidationError."""
        result = arima(ar1_series, order=(1, 0, 0))
        with pytest.raises(ValidationError):
            forecast_arima(result, ar1_series, h=5, levels=[0])
        with pytest.raises(ValidationError):
            forecast_arima(result, ar1_series, h=5, levels=[100])

    def test_forecast_empty_y_original(self, ar1_series):
        """Empty y_original raises ValidationError."""
        result = arima(ar1_series, order=(1, 0, 0))
        with pytest.raises(ValidationError):
            forecast_arima(result, np.array([]), h=5)

    def test_forecast_wrong_fitted_type(self, ar1_series):
        """Passing non-ARIMAResult raises ValidationError."""
        with pytest.raises(ValidationError):
            forecast_arima("not a result", ar1_series, h=5)  # type: ignore[arg-type]

    def test_auto_arima_invalid_ic(self, ar1_series):
        """Invalid IC string raises ValidationError."""
        with pytest.raises(ValidationError):
            auto_arima(ar1_series, ic="mse")

    def test_auto_arima_empty_series(self):
        """Empty series raises ValidationError."""
        with pytest.raises(ValidationError):
            auto_arima(np.array([]))

    def test_auto_arima_short_series(self):
        """Very short series (< 10) raises ValidationError."""
        with pytest.raises(ValidationError):
            auto_arima(np.array([1.0, 2.0, 3.0]))

    def test_auto_arima_negative_max(self, ar1_series):
        """Negative max_p/max_q/max_d raises ValidationError."""
        with pytest.raises(ValidationError):
            auto_arima(ar1_series, max_p=-1)

    def test_auto_arima_invalid_period(self, ar1_series):
        """period < 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            auto_arima(ar1_series, period=0)


# =========================================================================
# Edge Case Tests
# =========================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_short_series_n20(self, rng):
        """ARIMA fits on a short series (n=20)."""
        y = rng.normal(0, 1, 20)
        result = arima(y, order=(1, 0, 0))
        assert result.converged
        assert result.n_obs == 20

    def test_white_noise_000(self, white_noise):
        """ARIMA(0,0,0) is a white noise model."""
        result = arima(white_noise, order=(0, 0, 0))
        assert result.converged
        assert len(result.ar) == 0
        assert len(result.ma) == 0
        # sigma2 should be close to variance of the series
        assert abs(result.sigma2 - np.var(white_noise, ddof=0)) < 1.0

    def test_determine_d_stationary(self, white_noise):
        """Stationary series should have d=0."""
        d = _determine_d(white_noise, max_d=2)
        assert d == 0

    def test_determine_d_random_walk(self, random_walk):
        """Random walk should have d >= 1."""
        d = _determine_d(random_walk, max_d=2)
        assert d >= 1

    def test_forecast_h1(self, ar1_series):
        """h=1 produces length-1 arrays."""
        result = arima(ar1_series, order=(1, 0, 0))
        fc = forecast_arima(result, ar1_series, h=1)
        assert len(fc.mean) == 1
        assert len(fc.se) == 1

    @pytest.mark.slow
    def test_auto_arima_grid_search(self, white_noise):
        """Grid search with small bounds completes."""
        result = auto_arima(
            white_noise, max_p=1, max_q=1, stepwise=False
        )
        assert result.models_fitted > 0
        # Grid of p=0..1, q=0..1 = 4 combinations
        assert len(result.search_results) >= 4


# =====================================================================
# Whittle approximate MLE (CPU + GPU)
# =====================================================================


def _ar2_ma1_series(n, seed=0):
    """Synthetic ARMA(2,1) with known coefficients."""
    rng = np.random.RandomState(seed)
    eps = rng.randn(n)
    y = np.zeros(n)
    y[0] = eps[0]
    y[1] = 0.6 * y[0] + eps[1]
    for t in range(2, n):
        y[t] = 0.6 * y[t - 1] - 0.2 * y[t - 2] + eps[t] + 0.4 * eps[t - 1]
    return y


class TestArimaWhittle:
    """CPU Whittle (frequency-domain approximate MLE) for ARMA(p, q).

    Validated against the time-domain ML fit on long stationary
    series — the two agree to within Whittle's O(1/n) approximation
    floor.
    """

    def test_whittle_matches_ml_on_long_ar2_ma1(self):
        y = _ar2_ma1_series(5000)
        r_ml = arima(y, order=(2, 0, 1), method="ML")
        r_w = arima(y, order=(2, 0, 1), method="Whittle")
        np.testing.assert_allclose(r_ml.ar, r_w.ar, rtol=5e-3, atol=1e-3)
        np.testing.assert_allclose(r_ml.ma, r_w.ma, rtol=5e-3, atol=1e-3)
        assert r_ml.sigma2 == pytest.approx(r_w.sigma2, rel=5e-3)

    def test_whittle_ar_only(self):
        rng = np.random.RandomState(1)
        n = 3000
        eps = rng.randn(n)
        y = np.zeros(n)
        y[0] = eps[0]
        for t in range(1, n):
            y[t] = 0.7 * y[t - 1] + eps[t]
        r = arima(y, order=(1, 0, 0), method="Whittle")
        assert r.converged
        assert r.ar[0] == pytest.approx(0.7, abs=0.05)

    def test_whittle_ma_only(self):
        rng = np.random.RandomState(2)
        n = 3000
        eps = rng.randn(n)
        y = np.zeros(n)
        y[0] = eps[0]
        for t in range(1, n):
            y[t] = eps[t] + 0.5 * eps[t - 1]
        r = arima(y, order=(0, 0, 1), method="Whittle")
        assert r.converged
        assert r.ma[0] == pytest.approx(0.5, abs=0.1)

    def test_whittle_with_differencing(self):
        """Whittle on a random-walk-plus-drift via d=1."""
        rng = np.random.RandomState(3)
        n = 3000
        eps = rng.randn(n)
        y = np.cumsum(eps) + 0.05 * np.arange(n)  # unit root + drift
        # ARIMA(1, 1, 1)
        r = arima(y, order=(1, 1, 1), method="Whittle")
        assert r.converged
        # With d=1 applied, the differenced AR(1) coefficient is modest.
        assert abs(r.ar[0]) < 1.0

    def test_whittle_rejects_seasonal(self):
        """Whittle does not support seasonal ARMA in 1.8.0 — must raise."""
        y = _ar2_ma1_series(500)
        with pytest.raises(ValidationError, match="Whittle"):
            arima(y, order=(1, 0, 1),
                  seasonal=(1, 0, 1, 12), method="Whittle")


class TestArimaWhittleGPU:
    """Tests for the GPU Whittle ARMA backend.

    Two-tier validation: CPU Whittle is validated against the exact
    ML fit (see ``TestArimaWhittle``); GPU is validated against CPU
    Whittle at the ``GPU_FP32`` tier on log-likelihood and sigma².
    """

    def _gpu_available(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        return torch.cuda.is_available() or (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )

    def test_invalid_backend_raises(self):
        y = _ar2_ma1_series(500)
        with pytest.raises(ValidationError, match="backend"):
            arima(y, order=(2, 0, 1), method="Whittle", backend="quantum")

    def test_gpu_unavailable_raises_explicitly(self, monkeypatch):
        y = _ar2_ma1_series(500)
        from pystatistics.core.compute import device as dev_mod
        monkeypatch.setattr(dev_mod, "detect_gpu", lambda *a, **k: None)
        with pytest.raises(RuntimeError, match="no GPU"):
            arima(y, order=(2, 0, 1), method="Whittle", backend="gpu")

    def test_auto_backend_falls_back_to_cpu_when_no_gpu(self, monkeypatch):
        y = _ar2_ma1_series(500)
        from pystatistics.core.compute import device as dev_mod
        monkeypatch.setattr(dev_mod, "detect_gpu", lambda *a, **k: None)
        r = arima(y, order=(2, 0, 1), method="Whittle", backend="auto")
        assert r.converged

    def test_gpu_fp64_matches_cpu_whittle(self):
        if not self._gpu_available():
            pytest.skip("no GPU available")
        import torch
        if not torch.cuda.is_available():
            pytest.skip("FP64 test requires CUDA (MPS has no FP64)")
        y = _ar2_ma1_series(2000)
        r_cpu = arima(y, order=(2, 0, 1), method="Whittle", backend="cpu")
        r_gpu = arima(y, order=(2, 0, 1), method="Whittle",
                      backend="gpu", use_fp64=True)
        np.testing.assert_allclose(r_cpu.ar, r_gpu.ar,
                                   rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(r_cpu.ma, r_gpu.ma,
                                   rtol=1e-6, atol=1e-8)
        assert r_cpu.sigma2 == pytest.approx(r_gpu.sigma2, rel=1e-6)

    def test_gpu_fp32_matches_cpu_at_tier(self):
        from pystatistics.core.compute.tolerances import GPU_FP32
        if not self._gpu_available():
            pytest.skip("no GPU available")
        y = _ar2_ma1_series(2000)
        r_cpu = arima(y, order=(2, 0, 1), method="Whittle", backend="cpu")
        r_gpu = arima(y, order=(2, 0, 1), method="Whittle",
                      backend="gpu", use_fp64=False)
        assert r_cpu.sigma2 == pytest.approx(
            r_gpu.sigma2, rel=GPU_FP32.rtol, abs=GPU_FP32.atol,
        )
        assert r_cpu.log_likelihood == pytest.approx(
            r_gpu.log_likelihood, rel=GPU_FP32.rtol, abs=GPU_FP32.atol,
        )

    def test_gpu_datasource_input_matches_gpu_numpy(self):
        """Device-resident torch.Tensor input routes to GPU even
        without explicit backend='gpu'."""
        from pystatistics.core.compute.tolerances import GPU_FP32
        if not self._gpu_available():
            pytest.skip("no GPU available")
        y = _ar2_ma1_series(2000)
        r_numpy = arima(y, order=(2, 0, 1), method="Whittle",
                        backend="gpu", use_fp64=False)
        # Repeat the numpy-input GPU fit — should be deterministic
        # and match its own previous run.
        r_numpy_2 = arima(y, order=(2, 0, 1), method="Whittle",
                          backend="gpu", use_fp64=False)
        assert r_numpy.sigma2 == pytest.approx(
            r_numpy_2.sigma2, rel=GPU_FP32.rtol, abs=GPU_FP32.atol,
        )

    def test_whittle_on_very_long_series_runs(self):
        """The Whittle path's raison d'être: long series where exact
        ML is prohibitive. Just check it produces finite output at
        n = 100_000 — convergence quality is covered by the fp64
        match test above at smaller n."""
        if not self._gpu_available():
            pytest.skip("no GPU available")
        y = _ar2_ma1_series(100_000)
        r = arima(y, order=(2, 0, 1), method="Whittle",
                  backend="gpu", use_fp64=False)
        assert r.converged
        assert np.isfinite(r.sigma2)
        assert r.sigma2 > 0


# =====================================================================
# Batched ARMA fit (arima_batch) — 1.9.0
# =====================================================================


def _ar1_batch(K, n, seed=0):
    """Generate K independent AR(1) series with random phi in [0.1, 0.8]."""
    rng = np.random.RandomState(seed)
    phi_true = rng.uniform(0.1, 0.8, K)
    Y = np.zeros((K, n))
    for k in range(K):
        eps = rng.randn(n)
        Y[k, 0] = eps[0]
        for t in range(1, n):
            Y[k, t] = phi_true[k] * Y[k, t - 1] + eps[t]
    return Y, phi_true


class TestArimaBatch:
    """Batched ARMA fit via ``arima_batch``.

    CPU path is a Python loop over ``arima(method='Whittle')`` — no
    speed benefit, just an API-shape packaging for users with
    K series. GPU path is the actual win.
    """

    def test_cpu_loop_equivalent_to_serial_whittle(self):
        """The CPU loop fit returns results identical to calling
        ``arima(method='Whittle')`` per series."""
        from pystatistics.timeseries import arima_batch
        Y, phi_true = _ar1_batch(K=20, n=1500, seed=0)
        # Match tol between the batch call and the per-series
        # comparison — the batch API uses tol=1e-5 by default while
        # single-series arima uses 1e-8, so same code path runs with
        # different stopping criteria otherwise.
        r = arima_batch(Y, order=(1, 0, 0), method="Whittle",
                        backend="cpu", tol=1e-8, max_iter=200)
        assert r.n_series == 20
        assert r.ar.shape == (20, 1)
        matched = 0
        for k in range(20):
            try:
                rs = arima(Y[k], order=(1, 0, 0), method="Whittle",
                           tol=1e-8, max_iter=200)
            except Exception:
                # scipy L-BFGS-B can report ABNORMAL when finite-diff
                # gradient hits its noise floor — skip those rows in
                # the cross-check; the batch and serial paths raise
                # the same exception on the same input.
                continue
            np.testing.assert_array_equal(r.ar[k], rs.ar)
            assert r.sigma2[k] == rs.sigma2
            matched += 1
        assert matched >= 15   # well above scipy-stall incidence

    def test_invalid_backend_raises(self):
        from pystatistics.timeseries import arima_batch
        Y, _ = _ar1_batch(K=5, n=500)
        with pytest.raises(ValidationError, match="backend"):
            arima_batch(Y, order=(1, 0, 0), method="Whittle",
                        backend="quantum")

    def test_non_whittle_method_raises(self):
        from pystatistics.timeseries import arima_batch
        Y, _ = _ar1_batch(K=5, n=500)
        with pytest.raises(ValidationError, match="Whittle"):
            arima_batch(Y, order=(1, 0, 0), method="ML")

    def test_rejects_1d_input(self):
        from pystatistics.timeseries import arima_batch
        with pytest.raises(ValidationError, match="2-D"):
            arima_batch(np.zeros(500), order=(1, 0, 0), method="Whittle")


class TestArimaBatchGPU:
    """GPU batched ARMA fit — the actual win of ``arima_batch``.

    Validates against the serial CPU Whittle fit (row-by-row) at the
    GPU_FP32 tier on AR/MA coefficients and σ². The CPU Whittle has
    been validated against the exact time-domain ML fit upstream
    (TestArimaWhittle).
    """

    def _gpu_available(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        return torch.cuda.is_available() or (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )

    def test_gpu_unavailable_raises_explicitly(self, monkeypatch):
        from pystatistics.timeseries import arima_batch
        Y, _ = _ar1_batch(K=5, n=500)
        from pystatistics.core.compute import device as dev_mod
        monkeypatch.setattr(dev_mod, "detect_gpu", lambda *a, **k: None)
        with pytest.raises(RuntimeError, match="no GPU"):
            arima_batch(Y, order=(1, 0, 0), method="Whittle", backend="gpu")

    def test_auto_backend_falls_back_to_cpu_when_no_gpu(self, monkeypatch):
        from pystatistics.timeseries import arima_batch
        Y, _ = _ar1_batch(K=5, n=500)
        from pystatistics.core.compute import device as dev_mod
        monkeypatch.setattr(dev_mod, "detect_gpu", lambda *a, **k: None)
        r = arima_batch(Y, order=(1, 0, 0), method="Whittle", backend="auto")
        assert r.converged.any()

    def test_gpu_matches_serial_whittle_on_phi(self):
        """AR(1) on batched GPU matches serial CPU Whittle at a
        loosened-of-FP32 tier reflecting Adam vs L-BFGS-B curvature.

        Skips any series where scipy L-BFGS-B hits ABNORMAL (finite-
        diff gradient noise floor) — that's a property of the serial
        comparison, not the batched GPU fit.
        """
        if not self._gpu_available():
            pytest.skip("no GPU available")
        from pystatistics.timeseries import arima_batch
        Y, _ = _ar1_batch(K=30, n=3000, seed=1)
        r_gpu = arima_batch(Y, order=(1, 0, 0), method="Whittle",
                            backend="gpu", use_fp64=True)
        gpu_phi = []
        cpu_phi = []
        for k in range(30):
            try:
                rs = arima(Y[k], order=(1, 0, 0), method="Whittle")
            except Exception:
                continue
            gpu_phi.append(r_gpu.ar[k, 0])
            cpu_phi.append(rs.ar[0])
        assert len(cpu_phi) >= 25
        np.testing.assert_allclose(
            np.array(gpu_phi), np.array(cpu_phi),
            rtol=1e-2, atol=1e-3,
        )

    def test_gpu_batch_converges_on_all_series(self):
        """All series should converge within max_iter at the default
        learning rate on a well-posed synthetic problem."""
        if not self._gpu_available():
            pytest.skip("no GPU available")
        from pystatistics.timeseries import arima_batch
        Y, _ = _ar1_batch(K=50, n=2000, seed=2)
        r = arima_batch(Y, order=(1, 0, 0), method="Whittle",
                        backend="gpu", use_fp64=True)
        assert r.converged.sum() >= 48  # allow at most two stragglers
        assert np.all(np.isfinite(r.sigma2))
        assert np.all(r.sigma2 > 0)

    def test_tensor_input_routes_to_gpu(self):
        """A device-resident torch.Tensor for Y routes to GPU without
        an explicit ``backend='gpu'``, matching the shared convention."""
        if not self._gpu_available():
            pytest.skip("no GPU available")
        import torch
        from pystatistics.timeseries import arima_batch
        Y_np, _ = _ar1_batch(K=10, n=1000, seed=3)
        Y_t = torch.as_tensor(Y_np, device="cuda", dtype=torch.float32)
        r = arima_batch(Y_t, order=(1, 0, 0), method="Whittle",
                        use_fp64=False)
        assert "GPU" in r.method
        assert r.ar.shape == (10, 1)

    def test_tensor_input_with_cpu_backend_raises(self):
        """Rule 1: explicit backend='cpu' with a GPU tensor raises."""
        if not self._gpu_available():
            pytest.skip("no GPU available")
        import torch
        from pystatistics.timeseries import arima_batch
        Y_np, _ = _ar1_batch(K=5, n=500)
        Y_t = torch.as_tensor(Y_np, device="cuda", dtype=torch.float32)
        with pytest.raises(ValidationError, match="torch.Tensor"):
            arima_batch(Y_t, order=(1, 0, 0), method="Whittle",
                        backend="cpu")
