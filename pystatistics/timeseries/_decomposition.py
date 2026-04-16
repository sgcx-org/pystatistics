"""
Time series decomposition: classical and STL.

Provides decompose() matching R's stats::decompose() and stl() matching
R's stats::stl(). Classical decomposition uses centered moving averages;
STL uses iterative LOESS-based smoothing (Cleveland et al., 1990).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.validation import check_array, check_1d, check_finite


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecompositionResult:
    """
    Result from time series decomposition.

    Attributes
    ----------
    observed : NDArray
        Original time series.
    trend : NDArray
        Trend component. May contain NaN at edges for classical decomposition.
    seasonal : NDArray
        Seasonal component (repeats with the given period).
    residual : NDArray
        Remainder after removing trend and seasonal.
    period : int
        Seasonal period used.
    type : str
        ``'additive'`` or ``'multiplicative'``.
    method : str
        ``'classical'`` or ``'stl'``.
    """

    observed: NDArray
    trend: NDArray
    seasonal: NDArray
    residual: NDArray
    period: int
    type: str
    method: str

    def summary(self) -> str:
        """
        Return a human-readable summary of the decomposition.

        Returns
        -------
        str
            Multi-line summary string.
        """
        n = len(self.observed)
        n_trend_nan = int(np.sum(np.isnan(self.trend)))
        lines = [
            f"Time Series Decomposition ({self.method})",
            f"  Type:           {self.type}",
            f"  Period:         {self.period}",
            f"  Observations:   {n}",
            f"  Trend NaNs:     {n_trend_nan}",
        ]
        valid = ~np.isnan(self.residual)
        if np.any(valid):
            res = self.residual[valid]
            lines.append(f"  Residual std:   {float(np.std(res)):.6f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_series(x: ArrayLike, name: str = "x") -> NDArray:
    """
    Validate and convert input to a 1-D float array.

    Parameters
    ----------
    x : ArrayLike
        Input time series.
    name : str
        Parameter name for error messages.

    Returns
    -------
    NDArray
        Validated 1-D float array.

    Raises
    ------
    ValidationError
        If input is not 1-D or contains NaN/Inf.
    """
    arr = check_array(x, name)
    if arr.ndim == 0:
        raise ValidationError(f"{name}: expected 1D array, got scalar")
    arr = arr.ravel()
    check_1d(arr, name)
    check_finite(arr, name)
    return arr


def _validate_decompose_inputs(
    x: NDArray, period: int, decomp_type: str,
) -> None:
    """
    Validate inputs common to decompose() and stl().

    Raises
    ------
    ValidationError
        If *period*, *decomp_type*, or series length is invalid.
    """
    if not isinstance(period, (int, np.integer)):
        raise ValidationError(
            f"period: expected integer, got {type(period).__name__}"
        )
    if period < 2:
        raise ValidationError(f"period: must be >= 2, got {period}")
    if decomp_type not in ("additive", "multiplicative"):
        raise ValidationError(
            f"type: must be 'additive' or 'multiplicative', got {decomp_type!r}"
        )
    n = len(x)
    if n < 2 * period:
        raise ValidationError(
            f"x: length ({n}) must be >= 2 * period ({2 * period})"
        )
    if decomp_type == "multiplicative" and np.any(x <= 0):
        raise ValidationError(
            "x: multiplicative decomposition requires all values > 0"
        )


# ---------------------------------------------------------------------------
# Moving average helpers
# ---------------------------------------------------------------------------

def _centered_ma(x: NDArray, m: int) -> NDArray:
    """
    Compute a centered moving average matching R's ``decompose()``.

    For odd *m*: simple centered MA of length *m*.
    For even *m*: 2x *m* MA -- average of two *m*-length MAs offset by 1,
    equivalent to convolving with a symmetric filter of length *m* + 1
    whose endpoints are weighted 1/(2m) and interior weights are 1/m.

    Edges where the filter cannot be centered are set to NaN.

    Parameters
    ----------
    m : int
        Window width (the seasonal period).

    Returns
    -------
    NDArray
        Trend estimate with NaN at edges.
    """
    n = len(x)
    trend = np.full(n, np.nan)

    if m % 2 == 1:
        # Odd period: simple centered MA
        k = m // 2
        kernel = np.ones(m) / m
        conv = np.convolve(x, kernel, mode="valid")
        trend[k: k + len(conv)] = conv
    else:
        # Even period: 2xm MA (R's approach)
        # First pass: m-length MA (non-centered, shifted)
        kernel_m = np.ones(m) / m
        ma1 = np.convolve(x, kernel_m, mode="valid")  # length n - m + 1
        # Second pass: average consecutive pairs -> centered
        ma2 = (ma1[:-1] + ma1[1:]) / 2.0  # length n - m
        k = m // 2
        trend[k: k + len(ma2)] = ma2

    return trend


# ---------------------------------------------------------------------------
# Classical decomposition
# ---------------------------------------------------------------------------

def decompose(
    x: ArrayLike,
    period: int,
    *,
    type: str = "additive",
) -> DecompositionResult:
    """
    Classical time series decomposition.

    Matches R's ``stats::decompose()``.

    Algorithm
    ---------
    1. **Trend** via centered moving average of length *period*.
    2. **De-trend**: subtract (additive) or divide (multiplicative).
    3. **Seasonal**: average de-trended values at each seasonal position,
       then center so they sum to zero (additive) or average to 1
       (multiplicative).
    4. **Residual**: ``x - trend - seasonal`` (additive) or
       ``x / (trend * seasonal)`` (multiplicative).

    Parameters
    ----------
    x : ArrayLike
        Time series (1-D). Length must be >= 2 * *period*.
    period : int
        Seasonal period (>= 2).
    type : str
        ``'additive'`` or ``'multiplicative'``.

    Returns
    -------
    DecompositionResult

    Raises
    ------
    ValidationError
        On invalid inputs.
    """
    arr = _validate_series(x)
    _validate_decompose_inputs(arr, period, type)

    n = len(arr)
    trend = _centered_ma(arr, period)

    # De-trend
    if type == "additive":
        detrended = arr - trend
    else:
        detrended = arr / trend

    # Seasonal averages per position
    seasonal_means = np.zeros(period)
    for pos in range(period):
        indices = np.arange(pos, n, period)
        vals = detrended[indices]
        valid = ~np.isnan(vals)
        if np.any(valid):
            seasonal_means[pos] = np.mean(vals[valid])

    # Center seasonal component
    if type == "additive":
        seasonal_means -= np.mean(seasonal_means)
    else:
        seasonal_means /= np.mean(seasonal_means)

    # Tile seasonal to full length
    seasonal = np.tile(seasonal_means, n // period + 1)[:n]

    # Residual
    if type == "additive":
        residual = arr - trend - seasonal
    else:
        residual = arr / (trend * seasonal)

    return DecompositionResult(
        observed=arr,
        trend=trend,
        seasonal=seasonal,
        residual=residual,
        period=period,
        type=type,
        method="classical",
    )


# ---------------------------------------------------------------------------
# LOESS helper (simplified local regression)
# ---------------------------------------------------------------------------

def _tricube(d: NDArray) -> NDArray:
    """Tricube weight function: (1 - |d|^3)^3 for |d| < 1, else 0."""
    w = np.zeros_like(d)
    mask = np.abs(d) < 1.0
    w[mask] = (1.0 - np.abs(d[mask]) ** 3) ** 3
    return w


def _loess_smooth(
    x_pos: NDArray,
    y_vals: NDArray,
    span: int,
    degree: int = 1,
    weights: NDArray | None = None,
) -> NDArray:
    """
    Simplified LOESS smoothing for STL.

    For each point, fits a weighted local polynomial of the given degree
    using a tricube kernel with *span* nearest neighbours.

    Parameters
    ----------
    x_pos : NDArray
        Positions (typically ``np.arange(len(y_vals))``).
    y_vals : NDArray
        Values to smooth.
    span : int
        Number of nearest neighbours (window width).
    degree : int
        Polynomial degree (0 or 1).
    weights : NDArray or None
        External robustness weights (multiplied with tricube weights).

    Returns
    -------
    NDArray
        Smoothed values at each position in *x_pos*.
    """
    n = len(y_vals)
    span = min(span, n)
    smoothed = np.empty(n)

    for i in range(n):
        # Find span nearest neighbours
        dists = np.abs(x_pos - x_pos[i])
        idx = np.argsort(dists)[:span]
        d_max = dists[idx[-1]]
        if d_max == 0:
            d_max = 1.0

        w = _tricube(dists[idx] / d_max)
        if weights is not None:
            w = w * weights[idx]

        xi = x_pos[idx]
        yi = y_vals[idx]

        if degree == 0 or np.sum(w) == 0:
            total_w = np.sum(w)
            smoothed[i] = np.sum(w * yi) / total_w if total_w > 0 else yi[0]
        else:
            # Weighted least squares: degree-1 polynomial
            W = np.diag(w)
            X = np.column_stack([np.ones(len(xi)), xi - x_pos[i]])
            try:
                XtW = X.T @ W
                beta = np.linalg.solve(XtW @ X, XtW @ yi)
                smoothed[i] = beta[0]
            except np.linalg.LinAlgError:
                smoothed[i] = np.average(yi, weights=w) if np.sum(w) > 0 else yi[0]

    return smoothed


# ---------------------------------------------------------------------------
# Low-pass filter for STL
# ---------------------------------------------------------------------------

def _low_pass_filter(x: NDArray, period: int) -> NDArray:
    """
    Low-pass filter used in STL inner loop.

    Applies three successive moving averages of lengths *period*, *period*,
    and 3, matching the original STL specification.

    Parameters
    ----------
    x : NDArray
        Input series.
    period : int
        Seasonal period.

    Returns
    -------
    NDArray
        Filtered series (same length, NaN-padded edges trimmed via LOESS).
    """
    # MA of length period
    k1 = np.ones(period) / period
    s1 = np.convolve(x, k1, mode="same")
    # MA of length period again
    s2 = np.convolve(s1, k1, mode="same")
    # MA of length 3
    k3 = np.ones(3) / 3.0
    s3 = np.convolve(s2, k3, mode="same")
    return s3


# ---------------------------------------------------------------------------
# STL decomposition
# ---------------------------------------------------------------------------

def _next_odd(v: int) -> int:
    """Return *v* if odd, else *v* + 1."""
    return v if v % 2 == 1 else v + 1


def stl(
    x: ArrayLike,
    period: int,
    *,
    seasonal_window: int | None = None,
    trend_window: int | None = None,
    robust: bool = False,
    n_outer: int = 0,
    n_inner: int = 2,
) -> DecompositionResult:
    """
    Seasonal and Trend decomposition using LOESS (STL).

    Simplified implementation of Cleveland et al. (1990), matching
    R's ``stats::stl()`` in spirit.

    Parameters
    ----------
    x : ArrayLike
        Time series (1-D). Length must be >= 2 * *period*.
    period : int
        Seasonal period (>= 2).
    seasonal_window : int or None
        LOESS window width for seasonal extraction. Must be odd and >= 7.
        Default: next odd >= ``period + 1``, but at least 7.
    trend_window : int or None
        LOESS window width for trend extraction. Must be odd.
        Default: next odd >= ``ceil(1.5 * period / (1 - 1.5 / s_window)) + 1``.
    robust : bool
        If True, sets *n_outer* = 1 if it is 0, enabling robustness weights.
    n_outer : int
        Number of outer (robustness) iterations (>= 0).
    n_inner : int
        Number of inner iterations (>= 1).

    Returns
    -------
    DecompositionResult

    Raises
    ------
    ValidationError
        On invalid inputs.
    """
    arr = _validate_series(x)
    _validate_decompose_inputs(arr, period, "additive")

    n = len(arr)

    # Default seasonal window
    if seasonal_window is None:
        seasonal_window = _next_odd(max(7, period + 1))
    else:
        if seasonal_window < 7:
            raise ValidationError(
                f"seasonal_window: must be >= 7, got {seasonal_window}"
            )
        if seasonal_window % 2 == 0:
            raise ValidationError(
                f"seasonal_window: must be odd, got {seasonal_window}"
            )

    # Default trend window
    if trend_window is None:
        denom = 1.0 - 1.5 / seasonal_window
        trend_window = _next_odd(
            max(3, int(math.ceil(1.5 * period / denom)) + 1)
        )
    else:
        if trend_window % 2 == 0:
            raise ValidationError(
                f"trend_window: must be odd, got {trend_window}"
            )

    if n_inner < 1:
        raise ValidationError(f"n_inner: must be >= 1, got {n_inner}")

    if robust and n_outer == 0:
        n_outer = 1

    # Initialise components
    trend = np.zeros(n)
    seasonal = np.zeros(n)
    rob_weights = np.ones(n)
    x_pos_full = np.arange(n, dtype=float)

    total_outer = max(1, n_outer + 1)

    for _outer in range(total_outer):
        for _inner in range(n_inner):
            # Step 1: Detrending
            detrended = arr - trend

            # Step 2: Cycle-subseries smoothing
            smoothed_seasonal = np.empty(n)
            for pos in range(period):
                indices = np.arange(pos, n, period)
                sub_y = detrended[indices]
                sub_x = np.arange(len(sub_y), dtype=float)
                sub_w = rob_weights[indices]
                sub_smooth = _loess_smooth(
                    sub_x, sub_y, span=seasonal_window, degree=1, weights=sub_w,
                )
                smoothed_seasonal[indices] = sub_smooth

            # Step 3: Low-pass filter on smoothed seasonal
            lp = _low_pass_filter(smoothed_seasonal, period)

            # Step 4: Seasonal = smoothed - low-pass
            seasonal = smoothed_seasonal - lp

            # Step 5: De-seasonalise
            deseasoned = arr - seasonal

            # Step 6: Trend via LOESS on deseasoned series
            trend = _loess_smooth(
                x_pos_full, deseasoned, span=trend_window,
                degree=1, weights=rob_weights,
            )

        # Outer loop: compute robustness weights
        if _outer < total_outer - 1 and n_outer > 0:
            residual = arr - trend - seasonal
            h = 6.0 * np.median(np.abs(residual))
            if h > 0:
                u = np.abs(residual) / h
                rob_weights = _tricube(u)
            else:
                rob_weights = np.ones(n)

    residual = arr - trend - seasonal

    return DecompositionResult(
        observed=arr,
        trend=trend,
        seasonal=seasonal,
        residual=residual,
        period=period,
        type="additive",
        method="stl",
    )
