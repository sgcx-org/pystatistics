"""
Time series analysis module.

Phase 7A: ACF/PACF, differencing, and stationarity tests.
Phase 7B: Exponential smoothing (ETS) models.
Phase 7C: ARIMA models, forecasting, and automatic order selection.
Phase 7D: Time series decomposition (classical and STL).

Public API:
    acf(x)          - Autocorrelation function (matches R stats::acf)
    pacf(x)         - Partial autocorrelation function (matches R stats::pacf)
    diff(x)         - Difference a time series (matches R base::diff)
    ndiffs(x)       - Estimate differences for stationarity (matches R forecast::ndiffs)
    adf_test(x)     - Augmented Dickey-Fuller unit root test (matches R tseries::adf.test)
    kpss_test(x)    - KPSS stationarity test (matches R tseries::kpss.test)
    ets(y)          - Fit an ETS state space model (matches R forecast::ets)
    forecast_ets(f) - Forecast from a fitted ETS model
    arima(y)        - Fit an ARIMA model (matches R stats::arima)
    forecast_arima(f, y) - Forecast from a fitted ARIMA model
    auto_arima(y)   - Automatic ARIMA order selection (matches R forecast::auto.arima)
    decompose(x)    - Classical time series decomposition (matches R stats::decompose)
    stl(x)          - STL decomposition (matches R stats::stl)
"""

from pystatistics.timeseries._acf import acf, pacf
from pystatistics.timeseries._differencing import diff, ndiffs
from pystatistics.timeseries._stationarity import adf_test, kpss_test
from pystatistics.timeseries._common import ACFResult, StationarityResult
from pystatistics.timeseries._ets_fit import ets, ETSResult
from pystatistics.timeseries._ets_forecast import forecast_ets, ETSForecast
from pystatistics.timeseries._ets_models import ETSSpec
from pystatistics.timeseries._arima_fit import arima, ARIMAResult
from pystatistics.timeseries._arima_forecast import forecast_arima, ARIMAForecast
from pystatistics.timeseries._arima_order import auto_arima, AutoARIMAResult
from pystatistics.timeseries._decomposition import decompose, stl, DecompositionResult

__all__ = [
    "acf",
    "pacf",
    "diff",
    "ndiffs",
    "adf_test",
    "kpss_test",
    "ACFResult",
    "StationarityResult",
    "ets",
    "ETSResult",
    "forecast_ets",
    "ETSForecast",
    "ETSSpec",
    "arima",
    "ARIMAResult",
    "forecast_arima",
    "ARIMAForecast",
    "auto_arima",
    "AutoARIMAResult",
    "decompose",
    "stl",
    "DecompositionResult",
]
