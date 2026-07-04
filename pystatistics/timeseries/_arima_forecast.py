"""
Forecasting from fitted ARIMA models.

Generates point forecasts and prediction intervals on the original
(un-differenced) scale.  Matches the behaviour of R's
``predict.Arima()`` / ``forecast::forecast.Arima()``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats as sp_stats

from pystatistics.core.exceptions import ValidationError
from pystatistics.timeseries._arima_factored import (
    _multiply_ma_polynomials,
    _multiply_polynomials,
)
from pystatistics.timeseries._arima_kalman import kalman_arma_forecast
from pystatistics.timeseries._differencing import diff


# ---------------------------------------------------------------------------
# Forecast result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ARIMAForecast:
    """Forecast from a fitted ARIMA model.

    Attributes
    ----------
    mean : NDArray
        Point forecasts on the **original** (un-differenced) scale,
        length *h*.
    se : NDArray
        Standard errors of forecasts, length *h*.
    lower : dict[int, NDArray]
        Lower prediction-interval bounds keyed by level (e.g. 80, 95).
    upper : dict[int, NDArray]
        Upper prediction-interval bounds keyed by level.
    h : int
        Forecast horizon.
    order : tuple[int, int, int]
        The (p, d, q) order of the model.
    """

    mean: NDArray
    se: NDArray
    lower: dict[int, NDArray]
    upper: dict[int, NDArray]
    h: int
    order: tuple[int, int, int]

    def summary(self) -> str:
        """Return a human-readable forecast summary.

        Returns
        -------
        str
            Multi-line table of point forecasts and intervals.
        """
        levels = sorted(self.lower.keys())
        header_parts = ["  h", "    Forecast", "        SE"]
        for lv in levels:
            header_parts.append(f"  Lo {lv}")
            header_parts.append(f"  Hi {lv}")
        header = "".join(header_parts)

        lines = [
            f"Forecasts from ARIMA{self.order}",
            "",
            header,
            "  " + "-" * (len(header) - 2),
        ]
        for i in range(self.h):
            parts = [
                f"  {i + 1:>3}",
                f"  {self.mean[i]:>10.4f}",
                f"  {self.se[i]:>10.4f}",
            ]
            for lv in levels:
                parts.append(f"  {self.lower[lv][i]:>8.4f}")
                parts.append(f"  {self.upper[lv][i]:>8.4f}")
            lines.append("".join(parts))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MA(infinity) representation (psi weights)
# ---------------------------------------------------------------------------

def _psi_weights(ar: NDArray, ma: NDArray, h: int) -> NDArray:
    r"""Compute MA(:math:`\infty`) representation coefficients.

    The psi weights satisfy:

    .. math::

        \psi_0 = 1, \quad
        \psi_j = \theta_j + \sum_{i=1}^{\min(j,p)} \phi_i \, \psi_{j-i}

    where :math:`\theta_j = 0` for *j > q*.

    Parameters
    ----------
    ar : NDArray
        Effective AR coefficients (length *p*).
    ma : NDArray
        Effective MA coefficients (length *q*).
    h : int
        Number of psi weights to compute (indices 1 .. h-1 are used
        for the forecast-error variance).

    Returns
    -------
    NDArray
        Psi weights :math:`\psi_0, \psi_1, \ldots, \psi_{h-1}`.
    """
    p = len(ar)
    q = len(ma)
    psi = np.zeros(h, dtype=np.float64)
    psi[0] = 1.0
    for j in range(1, h):
        val = ma[j - 1] if j <= q else 0.0  # theta_j: ma is 0-indexed, so ma[j-1] = θ_j
        for i in range(1, min(j, p) + 1):
            val += ar[i - 1] * psi[j - i]  # ar is 0-indexed: ar[i-1] = phi_i
        psi[j] = val
    return psi


# ---------------------------------------------------------------------------
# Un-differencing
# ---------------------------------------------------------------------------

def _undifference(
    forecasts_diff: NDArray,
    y_original: NDArray,
    d: int,
    seasonal_d: int = 0,
    period: int = 1,
) -> NDArray:
    """Reverse differencing to obtain forecasts on the original scale.

    Parameters
    ----------
    forecasts_diff : NDArray
        Forecasts on the differenced scale, length *h*.
    y_original : NDArray
        The original (un-differenced) series.
    d : int
        Non-seasonal differencing order.
    seasonal_d : int
        Seasonal differencing order.
    period : int
        Seasonal period.

    Returns
    -------
    NDArray
        Forecasts on the original scale, length *h*.
    """
    h = len(forecasts_diff)
    fc = forecasts_diff.copy()

    # Integration must proceed from the MOST-differenced scale outward:
    # forecasts live on (1-B)^d (1-B^m)^D x, so the first pass
    # integrates against the tail of the (d, D-1)-differenced series,
    # the next against (d, D-2), ..., and the LAST pass against the
    # un-differenced series itself. The previous implementation walked
    # this ladder in reverse (first cumsum seeded with the tail of the
    # raw series), which was invisible for d + D <= 1 but produced
    # divergent point forecasts for d >= 2 or D >= 2 — ARIMA(1,2,1)
    # means were off by four orders of magnitude at h=12 while the SEs
    # were fine (RIGOR R18 follow-up).

    # Reverse seasonal differencing first (D times). The reference for
    # integration round j (j = D-1 .. 0) is the original series
    # differenced d times regularly and j times seasonally.
    if seasonal_d > 0 and period > 1:
        y_after_nonseasonal = y_original.copy()
        for _ in range(d):
            y_after_nonseasonal = diff(y_after_nonseasonal, differences=1, lag=1)

        for j in range(seasonal_d - 1, -1, -1):
            ref = y_after_nonseasonal.copy()
            for _ in range(j):
                ref = diff(ref, differences=1, lag=period)
            # Undo one round of seasonal differencing:
            # z[t] = z_diff[t] + z[t - period].
            tail = ref[-period:]
            extended = np.concatenate([tail, fc])
            for i in range(h):
                extended[period + i] = extended[period + i] + extended[i]
            fc = extended[period:]

    # Reverse non-seasonal differencing (d times): round k (k = d-1
    # .. 0) integrates against the tail of the k-times-differenced
    # original series.
    for k in range(d - 1, -1, -1):
        ref = y_original.copy()
        for _ in range(k):
            ref = diff(ref, differences=1, lag=1)
        fc = np.cumsum(np.concatenate([[ref[-1]], fc]))[1:]

    return fc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forecast_arima(
    fitted: 'ARIMASolution',  # noqa: F821  forward reference
    y_original: ArrayLike,
    *,
    h: int = 10,
    levels: list[int] | None = None,
) -> ARIMAForecast:
    """Generate forecasts from a fitted ARIMA model.

    Matches R's ``predict.Arima()`` / ``forecast::forecast.Arima()``.

    Parameters
    ----------
    fitted : ARIMASolution
        A fitted ARIMA model (from :func:`arima`).
    y_original : ArrayLike
        The **original** (un-differenced) time series that was passed
        to :func:`arima`.  Needed to reverse the differencing.
    h : int
        Forecast horizon (number of steps ahead).  Default 10.
    levels : list of int, optional
        Prediction-interval levels in percent (default ``[80, 95]``).

    Returns
    -------
    ARIMAForecast
        Point forecasts and prediction intervals on the original scale.

    Raises
    ------
    ValidationError
        If *h* < 1 or *levels* are invalid.
    """
    from pystatistics.timeseries._arima_fit import ARIMASolution  # noqa: F811

    if not isinstance(fitted, ARIMASolution):
        raise ValidationError(
            f"fitted: expected ARIMASolution, got {type(fitted).__name__}"
        )

    y_orig = np.asarray(y_original, dtype=np.float64).ravel()
    if y_orig.size == 0:
        raise ValidationError("y_original: must be non-empty")

    if h < 1:
        raise ValidationError(f"h: must be >= 1, got {h}")

    if levels is None:
        levels = [80, 95]
    for lv in levels:
        if lv < 1 or lv > 99:
            raise ValidationError(f"levels: each must be in [1, 99], got {lv}")

    p, d, q = fitted.order

    # Seasonal parameters
    seasonal_d = 0
    period = 1
    if fitted.seasonal_order is not None:
        _P, seasonal_d, _Q, period = fitted.seasonal_order

    # Effective ARMA polynomials. For seasonal models fitted.ar/ma hold
    # the factored NON-seasonal coefficients only; forecasting (and the
    # psi weights below) need the multiplied-out polynomials — dropping
    # the seasonal factors put airline-model forecasts ~5 units off R's
    # predict() (RIGOR R18 follow-up). For non-seasonal fits the
    # seasonal arrays are empty and these are identity operations.
    ar_eff = _multiply_polynomials(fitted.ar, fitted.seasonal_ar, period)
    ma_eff = _multiply_ma_polynomials(fitted.ma, fitted.seasonal_ma, period)

    # Differenced series
    y_diff = y_orig.copy()
    if seasonal_d > 0 and period > 1:
        for _ in range(seasonal_d):
            y_diff = diff(y_diff, differences=1, lag=period)
    for _ in range(d):
        y_diff = diff(y_diff, differences=1, lag=1)

    # Mean on the differenced scale
    mean_val = fitted.mean if fitted.mean is not None else 0.0

    # Point forecasts on the differenced scale, from the exact Kalman
    # filtered state — the same forecast origin R's predict.Arima uses
    # (KalmanForecast). A CSS-residual recursion (the previous
    # implementation) still carries conditioning error at the end of the
    # sample when an MA root is near the unit circle: on AirPassengers
    # (2,1,1)(0,1,0)[12] (ma1 = -0.98) it was ~1.4 off R's forecasts.
    # err_cov is the h x h forecast-error covariance under sigma2 = 1.
    fc_z, err_cov = kalman_arma_forecast(y_diff - mean_val, ar_eff, ma_eff, h)
    fc_diff = fc_z + mean_val

    # Forecast standard errors on the ORIGINAL scale. Un-differencing is
    # linear: the original-scale forecast error at horizon k is
    #     sum_{j=1..k} c_{k-j} e_j,
    # where e_j are the differenced-scale forecast errors and the c_i
    # are the coefficients of the integration operator
    # 1 / ((1-B)^d (1-B^m)^D). The variance therefore needs the full
    # error covariance (the e_j are serially correlated), not just a
    # psi-weight cumsum — the previous implementation ignored the
    # differencing entirely, reporting a flat se = sigma at every
    # horizon for a random walk instead of sigma*sqrt(h). This matches
    # R's integrated-state-space variance (predict.Arima) exactly.
    delta = np.array([1.0])
    for _ in range(d):
        delta = np.convolve(delta, np.array([1.0, -1.0]))
    for _ in range(seasonal_d):
        sdiff = np.zeros(period + 1)
        sdiff[0] = 1.0
        sdiff[-1] = -1.0
        delta = np.convolve(delta, sdiff)
    # c_i = psi weights of the pure integration operator (AR side only).
    c = _psi_weights(-delta[1:], np.array([]), h)

    var = np.empty(h, dtype=np.float64)
    for k in range(h):
        w = c[k::-1]  # w[j] = c_{k-j} for j = 0..k
        var[k] = w @ err_cov[: k + 1, : k + 1] @ w
    se = np.sqrt(var * fitted.sigma2)

    # Un-difference to original scale
    fc_orig = _undifference(fc_diff, y_orig, d, seasonal_d, period)

    # Prediction intervals
    lower: dict[int, NDArray] = {}
    upper: dict[int, NDArray] = {}
    for lv in levels:
        z = sp_stats.norm.ppf(0.5 + lv / 200.0)
        lower[lv] = fc_orig - z * se
        upper[lv] = fc_orig + z * se

    return ARIMAForecast(
        mean=fc_orig,
        se=se,
        lower=lower,
        upper=upper,
        h=h,
        order=fitted.order,
    )
