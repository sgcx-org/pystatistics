"""
Forecasting from fitted ETS models.

Generates point forecasts and prediction intervals for all ETS model types.
Analytical variance formulas are used for the common additive-error cases;
a ``sigma * sqrt(h)`` approximation is used as a fallback.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from pystatistics.core.exceptions import ValidationError
from pystatistics.timeseries._ets_fit import ETSSolution
from pystatistics.timeseries._ets_models import ETSSpec
from pystatistics.timeseries._forecast_common import _normalize_conf_levels


@dataclass(frozen=True)
class ETSForecast:
    """
    Forecast from a fitted ETS model.

    Attributes
    ----------
    mean : NDArray
        Point forecasts of length *n_ahead*.
    lower : dict[float, NDArray]
        Lower prediction-interval bounds keyed by confidence level as a
        fraction (e.g. ``{0.8: ..., 0.95: ...}``).
    upper : dict[float, NDArray]
        Upper prediction-interval bounds keyed by confidence level.
    n_ahead : int
        Forecast horizon.
    model : ETSSpec
        Model specification used.
    fitted : ETSSolution
        The underlying fitted model.
    """

    mean: NDArray
    lower: dict[float, NDArray]
    upper: dict[float, NDArray]
    n_ahead: int
    model: ETSSpec
    fitted: ETSSolution

    def summary(self) -> str:
        """
        Return a human-readable forecast summary.

        Returns
        -------
        str
            Multi-line table of forecasts and intervals.
        """
        levels = sorted(self.lower.keys())
        header_parts = ["  h", "    Forecast"]
        for lv in levels:
            pct = round(lv * 100)
            header_parts.append(f"  Lo {pct}")
            header_parts.append(f"  Hi {pct}")
        header = "".join(header_parts)

        lines = [
            f"Forecasts from {self.model.name}",
            "",
            header,
            "  " + "-" * (len(header) - 2),
        ]
        for i in range(self.n_ahead):
            parts = [f"  {i + 1:>3}", f"  {self.mean[i]:>10.4f}"]
            for lv in levels:
                parts.append(f"  {self.lower[lv][i]:>8.4f}")
                parts.append(f"  {self.upper[lv][i]:>8.4f}")
            lines.append("".join(parts))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Point forecast
# ---------------------------------------------------------------------------

def _point_forecast(
    spec: ETSSpec,
    level: float,
    trend: float | None,
    season: NDArray | None,
    phi: float | None,
    h: int,
) -> NDArray:
    """
    Compute h-step-ahead point forecasts from final states.

    Parameters
    ----------
    spec : ETSSpec
        Model specification.
    level : float
        Final level state.
    trend : float or None
        Final trend state.
    season : NDArray or None
        Final seasonal state vector of length ``period``.
    phi : float or None
        Damping parameter.
    h : int
        Forecast horizon.

    Returns
    -------
    NDArray
        Point forecasts of length *h*.
    """
    m = spec.period
    phi_val = phi if phi is not None else 1.0

    forecasts = np.empty(h, dtype=np.float64)

    for j in range(1, h + 1):
        # Trend contribution
        if spec.trend == "N":
            trend_comp = 0.0
        elif spec.trend == "A":
            trend_comp = j * trend
        else:  # Ad
            # phi + phi^2 + ... + phi^j
            if abs(phi_val - 1.0) < 1e-12:
                trend_comp = j * trend
            else:
                trend_comp = phi_val * (1.0 - phi_val ** j) / (1.0 - phi_val) * trend

        base = level + trend_comp

        # Seasonal contribution
        if spec.season == "N":
            forecasts[j - 1] = base
        elif spec.season == "A":
            s_idx = (j - 1) % m
            forecasts[j - 1] = base + season[s_idx]
        else:  # M
            s_idx = (j - 1) % m
            forecasts[j - 1] = base * season[s_idx]

    return forecasts


# ---------------------------------------------------------------------------
# Prediction interval variance
# ---------------------------------------------------------------------------

def _forecast_variance(
    spec: ETSSpec,
    sigma2: float,
    alpha: float,
    beta: float | None,
    gamma: float | None,
    phi: float | None,
    h: int,
) -> NDArray:
    """
    Compute forecast error variance for each horizon 1..h.

    Uses analytical formulas for additive-error models where available,
    and a ``sigma^2 * h`` approximation otherwise.

    Parameters
    ----------
    spec : ETSSpec
        Model specification.
    sigma2 : float
        Estimated error variance.
    alpha, beta, gamma, phi : float or None
        Smoothing parameters.
    h : int
        Forecast horizon.

    Returns
    -------
    NDArray
        Variance at each horizon, shape ``(h,)``.
    """
    if spec.error == "A" and spec.season == "N":
        return _variance_additive_nonseasonal(
            spec, sigma2, alpha, beta, phi, h
        )
    # Fallback: sigma^2 * sum of c_j^2 approximated as sigma^2 * j
    return sigma2 * np.arange(1, h + 1, dtype=np.float64)


def _variance_additive_nonseasonal(
    spec: ETSSpec,
    sigma2: float,
    alpha: float,
    beta: float | None,
    phi: float | None,
    h: int,
) -> NDArray:
    """
    Analytical variance for additive-error, non-seasonal models.

    ETS(A,N,N):
        Var(h) = sigma^2 * (1 + (h-1)*alpha^2)

    ETS(A,A,N):
        Var(h) = sigma^2 * (1 + (h-1)*(alpha^2 + alpha*beta + beta^2/6*(2h-1)))

    ETS(A,Ad,N):
        Uses cumulative phi sums.
    """
    var = np.empty(h, dtype=np.float64)
    phi_val = phi if phi is not None else 1.0

    if spec.trend == "N":
        # ETS(A,N,N)
        for j in range(1, h + 1):
            var[j - 1] = sigma2 * (1.0 + (j - 1) * alpha ** 2)
    elif spec.trend == "A":
        # ETS(A,A,N)
        b = beta if beta is not None else 0.0
        for j in range(1, h + 1):
            var[j - 1] = sigma2 * (
                1.0
                + (j - 1) * (alpha ** 2 + alpha * b + b ** 2 * (2 * j - 1) / 6.0)
            )
    else:
        # ETS(A,Ad,N) — cumulative approach
        b = beta if beta is not None else 0.0
        for j in range(1, h + 1):
            # c_i = alpha + beta * sum_{k=1}^{i} phi^k  for i=1..j-1, c_0=1
            cum = 1.0
            for i in range(1, j):
                if abs(phi_val - 1.0) < 1e-12:
                    phi_sum = float(i)
                else:
                    phi_sum = phi_val * (1.0 - phi_val ** i) / (1.0 - phi_val)
                c_i = alpha + b * phi_sum
                cum += c_i ** 2
            var[j - 1] = sigma2 * cum

    return var


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forecast_ets(
    fitted: ETSSolution,
    n_ahead: int = 10,
    *,
    conf_level: float | Sequence[float] = (0.80, 0.95),
) -> ETSForecast:
    """
    Generate forecasts from a fitted ETS model.

    Parameters
    ----------
    fitted : ETSSolution
        A fitted ETS model (from :func:`ets`).
    n_ahead : int
        Forecast horizon (number of steps ahead).
    conf_level : float or sequence of float
        Prediction-interval confidence level(s) as fractions in ``(0, 1)``
        (default ``(0.80, 0.95)``). A single float requests one interval;
        a sequence requests several. Whole-percent values (e.g. ``95``)
        are rejected.

    Returns
    -------
    ETSForecast
        Point forecasts and prediction intervals.

    Raises
    ------
    ValidationError
        If *n_ahead* < 1 or *conf_level* is invalid.
    """
    levels = _normalize_conf_levels(conf_level)

    if n_ahead < 1:
        raise ValidationError(f"n_ahead: must be >= 1, got {n_ahead}")

    spec = fitted.spec
    n = fitted.n_obs

    # Extract final states from last row
    final_states = fitted.states[n]
    idx = 0
    level = float(final_states[idx]); idx += 1

    trend: float | None = None
    if spec.trend in ("A", "Ad"):
        trend = float(final_states[idx]); idx += 1

    season: NDArray | None = None
    if spec.season in ("A", "M"):
        season = final_states[idx : idx + spec.period].copy()

    # Point forecasts
    mean = _point_forecast(spec, level, trend, season, fitted.phi, n_ahead)

    # Variance and intervals
    sigma2 = fitted.mse
    var = _forecast_variance(
        spec, sigma2, fitted.alpha, fitted.beta, fitted.gamma, fitted.phi, n_ahead
    )
    sd = np.sqrt(np.maximum(var, 0.0))

    lower: dict[float, NDArray] = {}
    upper: dict[float, NDArray] = {}
    for lv in levels:
        z = sp_stats.norm.ppf(0.5 + lv / 2.0)
        lower[lv] = mean - z * sd
        upper[lv] = mean + z * sd

    return ETSForecast(
        mean=mean,
        lower=lower,
        upper=upper,
        n_ahead=n_ahead,
        model=spec,
        fitted=fitted,
    )
