"""
Autocorrelation and partial autocorrelation functions.

Provides acf() and pacf() matching R's stats::acf() and stats::pacf().
Uses biased (1/n) normalization for ACF (matching R) and Durbin-Levinson
recursion for PACF.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats as sp_stats

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.validation import check_array, check_1d, check_finite
from pystatistics.timeseries._common import ACFResult


def _validate_series(x: ArrayLike, name: str = "x") -> NDArray:
    """
    Validate and convert a time series input to a 1-D float array.

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
        If input contains NaN/Inf or is not 1-D.
    """
    arr = check_array(x, name)
    if arr.ndim == 0:
        raise ValidationError(f"{name}: expected 1D array, got scalar")
    arr = arr.ravel()
    check_1d(arr, name)
    check_finite(arr, name)
    return arr


def _default_max_lag(n: int) -> int:
    """
    Compute the default max lag matching R's acf(): floor(10 * log10(n)).

    Parameters
    ----------
    n : int
        Number of observations.

    Returns
    -------
    int
        Default maximum lag, clamped to [1, n-1].
    """
    lag = int(np.floor(10.0 * np.log10(n)))
    return max(1, min(lag, n - 1))


def _compute_acf(x: NDArray, max_lag: int, demean: bool) -> NDArray:
    """
    Compute sample autocorrelations using biased (1/n) normalization.

    This matches R's stats::acf(type="correlation") exactly.

    Parameters
    ----------
    x : NDArray
        1-D time series array.
    max_lag : int
        Maximum lag to compute.
    demean : bool
        Whether to subtract the mean before computing.

    Returns
    -------
    NDArray
        Autocorrelation values for lags 0, 1, ..., max_lag.
    """
    n = len(x)
    if demean:
        x = x - np.mean(x)

    # Biased autocovariance c(k) = (1/n) * sum_{t=0}^{n-k-1} x[t]*x[t+k]
    acf_vals = np.empty(max_lag + 1)
    c0 = np.dot(x, x) / n
    acf_vals[0] = 1.0

    if c0 == 0.0:
        # Constant series: all autocorrelations are NaN except lag 0
        acf_vals[1:] = np.nan
    else:
        for k in range(1, max_lag + 1):
            ck = np.dot(x[:n - k], x[k:]) / n
            acf_vals[k] = ck / c0

    return acf_vals


def _ci_bounds(n: int, conf_level: float, n_lags: int) -> tuple[NDArray, NDArray]:
    """
    Compute confidence interval bounds under white noise null hypothesis.

    Uses Bartlett's approximation: +/- z_{alpha/2} / sqrt(n).

    Parameters
    ----------
    n : int
        Number of observations.
    conf_level : float
        Confidence level (e.g. 0.95).
    n_lags : int
        Number of lags (determines array length).

    Returns
    -------
    tuple[NDArray, NDArray]
        Upper and lower confidence bounds.
    """
    alpha = 1.0 - conf_level
    z = sp_stats.norm.ppf(1.0 - alpha / 2.0)
    bound = z / np.sqrt(n)
    ci_upper = np.full(n_lags, bound)
    ci_lower = np.full(n_lags, -bound)
    return ci_upper, ci_lower


def acf(
    x: ArrayLike,
    *,
    max_lag: int | None = None,
    conf_level: float = 0.95,
    demean: bool = True,
) -> ACFResult:
    """
    Compute the autocorrelation function.

    Matches R's stats::acf(type="correlation").

    Algorithm:
        c(k) = (1/n) * sum_{t=1}^{n-k} (x_t - x_bar)(x_{t+k} - x_bar)
        acf(k) = c(k) / c(0)

    Confidence intervals use Bartlett's approximation under the white noise
    null hypothesis: +/- z_{alpha/2} / sqrt(n).

    Parameters
    ----------
    x : ArrayLike
        Time series (1-D array).
    max_lag : int or None
        Maximum lag to compute. Default: min(10*log10(n), n-1), matching R.
    conf_level : float
        Confidence level for CI bands. Must be in (0, 1).
    demean : bool
        Whether to subtract the mean before computing. Default True, matches R.

    Returns
    -------
    ACFResult
        Result containing acf values, lags, and confidence bands.

    Raises
    ------
    ValidationError
        If inputs are invalid (NaN, non-1D, bad max_lag, etc.).

    Notes
    -----
    Validated against R stats::acf().
    """
    arr = _validate_series(x)
    n = len(arr)

    if n < 2:
        raise ValidationError(
            f"x: requires at least 2 observations for ACF, got {n}"
        )
    if not (0.0 < conf_level < 1.0):
        raise ValidationError(
            f"conf_level: must be in (0, 1), got {conf_level}"
        )

    if max_lag is None:
        max_lag = _default_max_lag(n)
    else:
        if not isinstance(max_lag, (int, np.integer)):
            raise ValidationError(
                f"max_lag: must be an integer, got {type(max_lag).__name__}"
            )
        max_lag = int(max_lag)
        if max_lag < 0:
            raise ValidationError(f"max_lag: must be >= 0, got {max_lag}")
        if max_lag >= n:
            raise ValidationError(
                f"max_lag: must be < n ({n}), got {max_lag}"
            )

    acf_vals = _compute_acf(arr, max_lag, demean)
    lags = np.arange(max_lag + 1)
    ci_upper, ci_lower = _ci_bounds(n, conf_level, max_lag + 1)

    return ACFResult(
        acf=acf_vals,
        lags=lags,
        n_obs=n,
        conf_level=conf_level,
        ci_upper=ci_upper,
        ci_lower=ci_lower,
        type="correlation",
    )


def pacf(
    x: ArrayLike,
    *,
    max_lag: int | None = None,
    conf_level: float = 0.95,
) -> ACFResult:
    """
    Compute the partial autocorrelation function.

    Matches R's stats::pacf(). Uses the Durbin-Levinson recursion.

    Algorithm:
        phi_{1,1} = acf(1)
        For k = 2, 3, ...:
            phi_{k,k} = (acf(k) - sum_{j=1}^{k-1} phi_{k-1,j} * acf(k-j)) /
                         (1 - sum_{j=1}^{k-1} phi_{k-1,j} * acf(j))
            phi_{k,j} = phi_{k-1,j} - phi_{k,k} * phi_{k-1,k-j}  for j < k

    Note: R's pacf() does NOT include lag 0. Lags start from 1.

    Confidence intervals: +/- z_{alpha/2} / sqrt(n) (same as ACF under
    white noise null hypothesis).

    Parameters
    ----------
    x : ArrayLike
        Time series (1-D array).
    max_lag : int or None
        Maximum lag to compute. Default: min(10*log10(n), n-1), matching R.
    conf_level : float
        Confidence level for CI bands. Must be in (0, 1).

    Returns
    -------
    ACFResult
        Result with type='partial'. Lags start at 1 (no lag 0).

    Raises
    ------
    ValidationError
        If inputs are invalid.

    Notes
    -----
    Validated against R stats::pacf().
    """
    arr = _validate_series(x)
    n = len(arr)

    if n < 3:
        raise ValidationError(
            f"x: requires at least 3 observations for PACF, got {n}"
        )
    if not (0.0 < conf_level < 1.0):
        raise ValidationError(
            f"conf_level: must be in (0, 1), got {conf_level}"
        )

    if max_lag is None:
        max_lag = _default_max_lag(n)
    else:
        if not isinstance(max_lag, (int, np.integer)):
            raise ValidationError(
                f"max_lag: must be an integer, got {type(max_lag).__name__}"
            )
        max_lag = int(max_lag)
        if max_lag < 1:
            raise ValidationError(f"max_lag: must be >= 1, got {max_lag}")
        if max_lag >= n:
            raise ValidationError(
                f"max_lag: must be < n ({n}), got {max_lag}"
            )

    # Compute full ACF first (always demean for PACF)
    acf_vals = _compute_acf(arr, max_lag, demean=True)

    # Durbin-Levinson recursion
    pacf_vals = np.empty(max_lag)
    phi = np.zeros(max_lag)

    # k=1
    phi[0] = acf_vals[1]
    pacf_vals[0] = acf_vals[1]

    for k in range(2, max_lag + 1):
        # Compute phi_{k,k}
        numer = acf_vals[k] - np.dot(phi[:k - 1], acf_vals[k - 1:0:-1])
        denom = 1.0 - np.dot(phi[:k - 1], acf_vals[1:k])

        if abs(denom) < 1e-15:
            # Degenerate case: fill remaining with NaN
            pacf_vals[k - 1:] = np.nan
            break

        phi_kk = numer / denom
        pacf_vals[k - 1] = phi_kk

        # Update phi_{k,j} for j < k
        phi_new = np.zeros(max_lag)
        for j in range(k - 1):
            phi_new[j] = phi[j] - phi_kk * phi[k - 2 - j]
        phi_new[k - 1] = phi_kk
        phi = phi_new

    lags = np.arange(1, max_lag + 1)
    ci_upper, ci_lower = _ci_bounds(n, conf_level, max_lag)

    return ACFResult(
        acf=pacf_vals,
        lags=lags,
        n_obs=n,
        conf_level=conf_level,
        ci_upper=ci_upper,
        ci_lower=ci_lower,
        type="partial",
    )
