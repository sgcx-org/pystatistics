"""
Seasonal-Trend decomposition using Loess (STL), matching R ``stats::stl``.

Clean-room implementation of Cleveland, Cleveland, McRae & Terpenning
(1990), "STL: A Seasonal-Trend Decomposition Procedure Based on Loess",
J. Official Statistics 6(1), 3-73 — the same procedure R's ``stats::stl``
wraps.  The algorithm was studied from the paper and from R's sources;
this code is written fresh (no transliteration of R's GPL Fortran).

Structure per the paper / R:

* **Inner loop** (``n_inner`` passes): detrend; smooth each cycle-subseries
  by loess (span ``seasonal_window``, degree ``seasonal_degree``), extended
  one period at each end; low-pass filter the extended result (moving
  averages of length *period*, *period*, 3, then a loess of span
  ``lowpass_window`` / degree ``lowpass_degree``); seasonal = smoothed
  subseries minus low-pass; deseasonalise; trend = loess of the
  deseasonalised series (span ``trend_window``, degree ``trend_degree``).
* **Outer loop** (``n_outer`` passes): bisquare robustness weights from the
  remainder (``w = (1 - (r/(6*MAD))^2)^2`` with 0.001/0.999 clamps) feed the
  seasonal and trend loess of the next round.
* ``seasonal_window="periodic"`` uses span ``10*n + 1`` with degree 0 and
  afterwards replaces the seasonal by its cycle-position means (exactly
  periodic), as R's driver does.

Numerical parity with R ``stats::stl`` is exact to floating-point noise
(validated near machine precision on the R reference fixtures) because the
loess evaluation grid (the ``*_jump`` strides with linear interpolation),
the endpoint rules, and the running-sum moving averages all replicate the
reference algorithm.

Documented divergences from R (interface only, both fail-loud by design):

* R has no default ``s.window`` (it must be supplied); here
  ``seasonal_window`` defaults to ``"periodic"``, the standard published
  choice.
* R silently rounds spans up to the next odd integer >= 3;  PyStatistics
  raises ``ValidationError`` instead of silently changing the question
  (window parameters must be odd integers >= 3).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.result import Result
from pystatistics.timeseries._decomposition import (
    DecompositionParams,
    DecompositionSolution,
    _validate_series,
)
from pystatistics.timeseries._loess import loess_smooth, loess_subseries_smooth
from pystatistics.timeseries._stl_robust import _robustness_weights


# ---------------------------------------------------------------------------
# Building blocks (input contracts enforced by stl() below)
# ---------------------------------------------------------------------------

def _next_odd(value: float) -> int:
    """Round to the nearest integer, then bump even values up by one."""
    v = int(round(value))
    return v + 1 if v % 2 == 0 else v


def _moving_average(x: NDArray, width: int) -> NDArray:
    """
    Running mean of *width* consecutive values, length ``len(x)-width+1``.

    Uses the same sequential running-sum update order as the reference
    implementation — ``(v - x[out]) + x[in]``, then divide — so the
    floating-point result matches R exactly.  (Vectorised cumulative sums
    round differently; the sequential loop runs on plain Python floats,
    which are the same IEEE doubles.)
    """
    n = len(x)
    values = x.tolist()
    out = [0.0] * (n - width + 1)
    v = 0.0
    for i in range(width):
        v += values[i]
    out[0] = v / width
    for j in range(1, n - width + 1):
        v = v - values[j - 1] + values[j + width - 1]
        out[j] = v / width
    return np.asarray(out, dtype=np.float64)


def _low_pass(ext: NDArray, period: int) -> NDArray:
    """
    Moving-average cascade of the STL low-pass filter.

    Two running means of length *period* followed by one of length 3;
    input length ``n + 2*period`` reduces to ``n``.  (The trailing loess of
    span ``lowpass_window`` is applied by the caller.)
    """
    stage = _moving_average(ext, period)
    stage = _moving_average(stage, period)
    return _moving_average(stage, 3)


def _cycle_subseries(
    detrended: NDArray,
    period: int,
    span: int,
    degree: int,
    jump: int,
    rob_weights: NDArray | None,
) -> NDArray:
    """
    Smooth every cycle-subseries and extend one period at each end.

    Each of the *period* subseries (positions ``pos, pos+period, ...``) is
    loess-smoothed at its own time scale, then extrapolated to positions
    0 and ``k+1``, giving an interleaved result of length
    ``n + 2*period``.  Subseries of equal length share their evaluation
    geometry, so each length-group is smoothed in one vectorised batch
    (at most two groups: when *period* does not divide *n*, the first
    ``n % period`` subseries are one element longer).
    """
    n = len(detrended)
    ext = np.empty(n + 2 * period, dtype=np.float64)
    remainder = n % period
    groups = [range(0, remainder), range(remainder, period)] if remainder \
        else [range(period)]
    for group in groups:
        positions = list(group)
        if not positions:
            continue
        sub_y = np.stack([detrended[pos::period] for pos in positions])
        sub_w = (np.stack([rob_weights[pos::period] for pos in positions])
                 if rob_weights is not None else None)
        smoothed, head, tail = loess_subseries_smooth(
            sub_y, span, degree, jump, sub_weights=sub_w
        )
        for row, pos in enumerate(positions):
            ext[pos::period] = np.concatenate(
                ([head[row]], smoothed[row], [tail[row]])
            )
    return ext


def _inner_loop(
    y: NDArray,
    period: int,
    trend: NDArray,
    rob_weights: NDArray | None,
    cfg: dict,
) -> tuple[NDArray, NDArray]:
    """
    Run ``cfg['n_inner']`` passes of the STL inner loop.

    Returns the updated ``(seasonal, trend)``.  ``rob_weights`` (from the
    outer loop) weight the seasonal and trend loess but never the low-pass
    loess, exactly as in the reference algorithm.
    """
    n = len(y)
    seasonal = np.zeros(n, dtype=np.float64)
    for _ in range(cfg["n_inner"]):
        detrended = y - trend
        ext = _cycle_subseries(
            detrended, period, cfg["seasonal_window"], cfg["seasonal_degree"],
            cfg["seasonal_jump"], rob_weights,
        )
        low = loess_smooth(
            _low_pass(ext, period), cfg["lowpass_window"],
            cfg["lowpass_degree"], cfg["lowpass_jump"], weights=None,
        )
        seasonal = ext[period: period + n] - low
        trend = loess_smooth(
            y - seasonal, cfg["trend_window"], cfg["trend_degree"],
            cfg["trend_jump"], weights=rob_weights,
        )
    return seasonal, trend


# ---------------------------------------------------------------------------
# Parameter validation / defaults
# ---------------------------------------------------------------------------

def _check_window(name: str, value: object) -> int:
    """Windows must be odd integers >= 3 (R silently rounds; we fail loud)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValidationError(
            f"{name}: expected an odd integer >= 3, got {value!r}"
        )
    v = int(value)
    if v < 3:
        raise ValidationError(f"{name}: must be >= 3, got {v}")
    if v % 2 == 0:
        raise ValidationError(
            f"{name}: must be odd, got {v} "
            f"(R rounds even spans up silently; pass {v + 1} explicitly)"
        )
    return v


def _check_degree(name: str, value: object) -> int:
    """Loess degrees are 0 or 1."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) \
            or int(value) not in (0, 1):
        raise ValidationError(f"{name}: must be 0 or 1, got {value!r}")
    return int(value)


def _check_jump(name: str, value: object) -> int:
    """Jump strides are integers >= 1."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValidationError(
            f"{name}: expected a positive integer, got {value!r}"
        )
    v = int(value)
    if v < 1:
        raise ValidationError(f"{name}: must be >= 1, got {v}")
    return v


def _resolve_config(
    n: int,
    period: int,
    seasonal_window: int | str,
    seasonal_degree: int,
    trend_window: int | None,
    trend_degree: int,
    lowpass_window: int | None,
    lowpass_degree: int | None,
    seasonal_jump: int | None,
    trend_jump: int | None,
    lowpass_jump: int | None,
    robust: bool,
    n_inner: int | None,
    n_outer: int | None,
) -> tuple[dict, bool]:
    """Validate every STL parameter and fill R's defaults. See stl()."""
    periodic = False
    if isinstance(seasonal_window, str):
        if seasonal_window != "periodic":
            raise ValidationError(
                f"seasonal_window: string value must be 'periodic', "
                f"got {seasonal_window!r}"
            )
        periodic = True
        if seasonal_degree != 0:
            raise ValidationError(
                "seasonal_degree: seasonal_window='periodic' implies "
                f"seasonal_degree=0, got {seasonal_degree}"
            )
        s_window = 10 * n + 1
        s_degree = 0
    else:
        s_window = _check_window("seasonal_window", seasonal_window)
        s_degree = _check_degree("seasonal_degree", seasonal_degree)

    t_degree = _check_degree("trend_degree", trend_degree)
    if trend_window is None:
        t_window = _next_odd(
            math.ceil(1.5 * period / (1.0 - 1.5 / s_window))
        )
    else:
        t_window = _check_window("trend_window", trend_window)

    if lowpass_window is None:
        l_window = _next_odd(period)
    else:
        l_window = _check_window("lowpass_window", lowpass_window)
    if lowpass_degree is None:
        l_degree = t_degree
    else:
        l_degree = _check_degree("lowpass_degree", lowpass_degree)

    s_jump = (math.ceil(s_window / 10) if seasonal_jump is None
              else _check_jump("seasonal_jump", seasonal_jump))
    t_jump = (math.ceil(t_window / 10) if trend_jump is None
              else _check_jump("trend_jump", trend_jump))
    l_jump = (math.ceil(l_window / 10) if lowpass_jump is None
              else _check_jump("lowpass_jump", lowpass_jump))

    if n_inner is None:
        inner = 1 if robust else 2
    else:
        if isinstance(n_inner, bool) or not isinstance(n_inner, (int, np.integer)) \
                or int(n_inner) < 1:
            raise ValidationError(f"n_inner: must be an integer >= 1, got {n_inner!r}")
        inner = int(n_inner)
    if n_outer is None:
        outer = 15 if robust else 0
    else:
        if isinstance(n_outer, bool) or not isinstance(n_outer, (int, np.integer)) \
                or int(n_outer) < 0:
            raise ValidationError(f"n_outer: must be an integer >= 0, got {n_outer!r}")
        outer = int(n_outer)

    cfg = {
        "seasonal_window": s_window, "seasonal_degree": s_degree,
        "seasonal_jump": s_jump,
        "trend_window": t_window, "trend_degree": t_degree,
        "trend_jump": t_jump,
        "lowpass_window": l_window, "lowpass_degree": l_degree,
        "lowpass_jump": l_jump,
        "n_inner": inner, "n_outer": outer,
    }
    return cfg, periodic


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stl(
    x: ArrayLike,
    period: int,
    *,
    seasonal_window: int | str = "periodic",
    seasonal_degree: int = 0,
    trend_window: int | None = None,
    trend_degree: int = 1,
    lowpass_window: int | None = None,
    lowpass_degree: int | None = None,
    seasonal_jump: int | None = None,
    trend_jump: int | None = None,
    lowpass_jump: int | None = None,
    robust: bool = False,
    n_inner: int | None = None,
    n_outer: int | None = None,
) -> DecompositionSolution:
    """
    Seasonal-Trend decomposition using Loess (STL).

    Matches R's ``stats::stl`` (Cleveland et al., 1990): identical
    parameters produce identical seasonal/trend/remainder components up to
    floating-point noise.  Two interface divergences, both deliberate:
    ``seasonal_window`` defaults to ``"periodic"`` (R requires it
    explicitly), and invalid spans raise instead of being silently rounded
    up as R does.

    Parameters
    ----------
    x : ArrayLike
        Time series (1-D, finite). Length must exceed ``2 * period``.
    period : int
        Seasonal period (>= 2), e.g. 12 for monthly data.
    seasonal_window : int or "periodic"
        Loess span for cycle-subseries smoothing (odd, >= 3), or
        ``"periodic"`` for an exactly periodic seasonal (equivalent to a
        span of ``10*n + 1`` with degree 0, followed by cycle-position
        averaging). Default ``"periodic"``. R: ``s.window``.
    seasonal_degree : int
        Loess degree (0 or 1) for the seasonal smoother. Default 0,
        matching R. Must be 0 when ``seasonal_window="periodic"``.
    trend_window : int or None
        Loess span for the trend (odd, >= 3). Default
        ``nextodd(ceil(1.5*period / (1 - 1.5/seasonal_window)))``, as in R.
    trend_degree : int
        Loess degree (0 or 1) for the trend. Default 1, matching R.
    lowpass_window : int or None
        Loess span of the low-pass filter. Default ``nextodd(period)``.
    lowpass_degree : int or None
        Loess degree of the low-pass filter. Default: ``trend_degree``.
    seasonal_jump, trend_jump, lowpass_jump : int or None
        Evaluation strides: each loess is evaluated every ``jump``-th point
        and linearly interpolated between, exactly as in R. Defaults
        ``ceil(window/10)`` (R's defaults). Pass 1 to evaluate the loess at
        every point with no interpolation (R's own interpolation is a speed
        shortcut, Cleveland et al. 1990, sec. 3.4; with the default strides
        the output matches R's bit-for-bit up to float noise).
    robust : bool
        Enable outer-loop robustness iterations (bisquare weights on the
        remainder). Changes the ``n_inner``/``n_outer`` defaults to R's
        (1, 15) from (2, 0).
    n_inner : int or None
        Inner-loop passes (>= 1). Default: 1 if *robust* else 2. R: ``inner``.
    n_outer : int or None
        Robustness iterations (>= 0). Default: 15 if *robust* else 0.
        R: ``outer``.

    Returns
    -------
    DecompositionSolution
        With ``seasonal + trend + residual == observed`` exactly.
        ``info`` records the resolved windows/degrees/jumps, the iteration
        counts, and the final robustness weights.

    Raises
    ------
    ValidationError
        On invalid input, spans that are even or < 3, degrees outside
        {0, 1}, or a series with fewer than two full periods.
    """
    arr = _validate_series(x)
    if not isinstance(period, (int, np.integer)) or isinstance(period, bool):
        raise ValidationError(
            f"period: expected integer, got {type(period).__name__}"
        )
    if period < 2:
        raise ValidationError(f"period: must be >= 2, got {period}")
    n = len(arr)
    if n <= 2 * period:
        raise ValidationError(
            f"x: STL requires more than two full periods "
            f"(length {n} <= 2 * period = {2 * period})"
        )

    cfg, periodic = _resolve_config(
        n, period, seasonal_window, seasonal_degree,
        trend_window, trend_degree, lowpass_window, lowpass_degree,
        seasonal_jump, trend_jump, lowpass_jump, robust, n_inner, n_outer,
    )

    trend = np.zeros(n, dtype=np.float64)
    rob_weights: NDArray | None = None
    for iteration in range(cfg["n_outer"] + 1):
        seasonal, trend = _inner_loop(arr, period, trend, rob_weights, cfg)
        if iteration == cfg["n_outer"]:
            break
        rob_weights = _robustness_weights(arr, trend + seasonal)
    weights_out = rob_weights if rob_weights is not None \
        else np.ones(n, dtype=np.float64)

    if periodic:
        for pos in range(period):
            seasonal[pos::period] = np.mean(seasonal[pos::period])
    residual = arr - seasonal - trend

    info = {
        "method": "stl",
        "type": "additive",
        "robust": robust,
        "windows": {
            "seasonal": cfg["seasonal_window"],
            "trend": cfg["trend_window"],
            "lowpass": cfg["lowpass_window"],
        },
        "degrees": {
            "seasonal": cfg["seasonal_degree"],
            "trend": cfg["trend_degree"],
            "lowpass": cfg["lowpass_degree"],
        },
        "jumps": {
            "seasonal": cfg["seasonal_jump"],
            "trend": cfg["trend_jump"],
            "lowpass": cfg["lowpass_jump"],
        },
        "n_inner": cfg["n_inner"],
        "n_outer": cfg["n_outer"],
        "seasonal_window_mode": "periodic" if periodic else "span",
        "robustness_weights": weights_out,
    }
    return DecompositionSolution(
        _result=Result(
            params=DecompositionParams(
                observed=arr,
                trend=trend,
                seasonal=seasonal,
                residual=residual,
                period=int(period),
                type="additive",
                method="stl",
            ),
            info=info,
            timing=None,
            backend_name="cpu",
            warnings=(),
        )
    )
