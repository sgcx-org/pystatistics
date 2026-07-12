"""
Classical time series decomposition and shared decomposition results.

Provides decompose() matching R's stats::decompose() (centered moving
averages) plus the DecompositionParams/DecompositionSolution result
types shared with the STL implementation in ``_stl.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.result import Result, SolutionReprMixin
from pystatistics.core.validation import check_array, check_1d, check_finite


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecompositionParams:
    """Immutable parameter payload for a time series decomposition.

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
    kind : str
        ``'additive'`` or ``'multiplicative'``.
    method : str
        ``'classical'`` or ``'stl'``.
    """

    observed: NDArray
    trend: NDArray
    seasonal: NDArray
    residual: NDArray
    period: int
    kind: str
    method: str


@dataclass
class DecompositionSolution(SolutionReprMixin):
    """
    Result from time series decomposition.

    Wraps a :class:`Result` ``[DecompositionParams]`` envelope; every datum
    is exposed via a read-only ``@property`` so the public attribute
    surface is unchanged from the previous flat dataclass.

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
    kind : str
        ``'additive'`` or ``'multiplicative'``.
    method : str
        ``'classical'`` or ``'stl'``.
    """

    _result: Result[DecompositionParams]

    @property
    def observed(self) -> NDArray:
        return self._result.params.observed

    @property
    def trend(self) -> NDArray:
        return self._result.params.trend

    @property
    def seasonal(self) -> NDArray:
        return self._result.params.seasonal

    @property
    def residual(self) -> NDArray:
        return self._result.params.residual

    @property
    def period(self) -> int:
        return self._result.params.period

    @property
    def kind(self) -> str:
        return self._result.params.kind

    @property
    def method(self) -> str:
        return self._result.params.method

    @property
    def info(self) -> dict:
        return self._result.info

    @property
    def timing(self) -> dict[str, float] | None:
        return self._result.timing

    @property
    def backend_name(self) -> str:
        return self._result.backend_name

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._result.warnings

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
            f"  Type:           {self.kind}",
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
            f"kind: must be 'additive' or 'multiplicative', got {decomp_type!r}"
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
    For even *m*: 2x *m* MA -- average of two *m*-length MAs shifted by 1,
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
    kind: str = "additive",
) -> DecompositionSolution:
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
    kind : str
        ``'additive'`` or ``'multiplicative'``.

    Returns
    -------
    DecompositionSolution

    Raises
    ------
    ValidationError
        On invalid inputs.
    """
    arr = _validate_series(x)
    _validate_decompose_inputs(arr, period, kind)

    n = len(arr)
    trend = _centered_ma(arr, period)

    # De-trend
    if kind == "additive":
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
    if kind == "additive":
        seasonal_means -= np.mean(seasonal_means)
    else:
        seasonal_means /= np.mean(seasonal_means)

    # Tile seasonal to full length
    seasonal = np.tile(seasonal_means, n // period + 1)[:n]

    # Residual
    if kind == "additive":
        residual = arr - trend - seasonal
    else:
        residual = arr / (trend * seasonal)

    return DecompositionSolution(
        _result=Result(
            params=DecompositionParams(
                observed=arr,
                trend=trend,
                seasonal=seasonal,
                residual=residual,
                period=period,
                kind=kind,
                method="classical",
            ),
            info={"method": "classical", "kind": kind},
            timing=None,
            backend_name="cpu",
            warnings=(),
        )
    )
