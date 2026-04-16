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
# Point forecasts on the differenced scale
# ---------------------------------------------------------------------------

def _forecast_differenced(
    ar: NDArray,
    ma: NDArray,
    mean: float,
    residuals: NDArray,
    y_diff: NDArray,
    h: int,
) -> NDArray:
    """Compute h-step-ahead point forecasts on the differenced scale.

    Uses the recursive ARMA forecast:

    .. math::

        \\hat y_{n+k} = \\mu + \\sum_{i=1}^{p} \\phi_i (y^*_{n+k-i} - \\mu)
                      + \\sum_{j=1}^{q} \\theta_j e_{n+k-j}

    where future residuals are set to zero and future y values are
    replaced by their forecasts.

    Parameters
    ----------
    ar : NDArray
        Effective AR coefficients.
    ma : NDArray
        Effective MA coefficients.
    mean : float
        Mean of the differenced series (0 if ``include_mean=False``).
    residuals : NDArray
        Residuals on the differenced scale.
    y_diff : NDArray
        The differenced series.
    h : int
        Forecast horizon.

    Returns
    -------
    NDArray
        Point forecasts of length *h* on the differenced scale.
    """
    p = len(ar)
    q = len(ma)
    n = len(y_diff)

    forecasts = np.empty(h, dtype=np.float64)
    for k in range(1, h + 1):
        val = mean
        # AR part
        for i in range(1, p + 1):
            idx = n + k - i
            if idx < n:
                # Use actual observation (de-meaned)
                val += ar[i - 1] * (y_diff[idx] - mean)
            else:
                # Use previous forecast (de-meaned)
                val += ar[i - 1] * (forecasts[idx - n] - mean)
        # MA part
        for j in range(1, q + 1):
            idx = n + k - j
            if idx < n:
                val += ma[j - 1] * residuals[idx]
            # else: future residual = 0
        forecasts[k - 1] = val
    return forecasts


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

    # Reverse seasonal differencing first (D times)
    if seasonal_d > 0 and period > 1:
        # We need to undo D rounds of seasonal differencing.
        # For each round, z[t] = z_diff[t] + z[t - period].
        # We need access to the series *before* the last round of
        # non-seasonal differencing was undone, so we undo seasonal
        # differencing first, then non-seasonal.
        #
        # The original series after non-seasonal differencing (d times)
        # but before seasonal differencing:
        y_after_nonseasonal = y_original.copy()
        for _ in range(d):
            y_after_nonseasonal = diff(y_after_nonseasonal, differences=1, lag=1)

        for _dd in range(seasonal_d):
            # y_after_nonseasonal is the series before this round of
            # seasonal differencing.  We need the last ``period`` values.
            tail = y_after_nonseasonal[-(period):]  # noqa: E275
            extended = np.concatenate([tail, fc])
            for i in range(h):
                extended[period + i] = extended[period + i] + extended[i]
            fc = extended[period:]

            # For additional rounds, apply one seasonal diff to the
            # reference series.
            y_after_nonseasonal = diff(
                y_after_nonseasonal, differences=1, lag=period
            )

    # Reverse non-seasonal differencing (d times)
    for _dd in range(d):
        last_val = y_original[-(d - _dd) :][0] if d > 1 else y_original[-1]
        # More robust: use the series differenced (_dd) times for the
        # appropriate tail value.
        y_temp = y_original.copy()
        for _k in range(_dd):
            y_temp = diff(y_temp, differences=1, lag=1)
        last_val = y_temp[-1]
        fc = np.cumsum(np.concatenate([[last_val], fc]))[1:]

    return fc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forecast_arima(
    fitted: 'ARIMAResult',  # noqa: F821  forward reference
    y_original: ArrayLike,
    *,
    h: int = 10,
    levels: list[int] | None = None,
) -> ARIMAForecast:
    """Generate forecasts from a fitted ARIMA model.

    Matches R's ``predict.Arima()`` / ``forecast::forecast.Arima()``.

    Parameters
    ----------
    fitted : ARIMAResult
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
    from pystatistics.timeseries._arima_fit import ARIMAResult  # noqa: F811

    if not isinstance(fitted, ARIMAResult):
        raise ValidationError(
            f"fitted: expected ARIMAResult, got {type(fitted).__name__}"
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

    # Differenced series
    y_diff = y_orig.copy()
    if seasonal_d > 0 and period > 1:
        for _ in range(seasonal_d):
            y_diff = diff(y_diff, differences=1, lag=period)
    for _ in range(d):
        y_diff = diff(y_diff, differences=1, lag=1)

    # Mean on the differenced scale
    mean_val = fitted.mean if fitted.mean is not None else 0.0

    # Point forecasts on the differenced scale
    fc_diff = _forecast_differenced(
        fitted.ar, fitted.ma, mean_val, fitted.residuals, y_diff, h
    )

    # Psi weights for forecast-error variance
    psi = _psi_weights(fitted.ar, fitted.ma, h)

    # Forecast standard errors
    cumvar = np.cumsum(psi ** 2) * fitted.sigma2
    se = np.sqrt(cumvar)

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
