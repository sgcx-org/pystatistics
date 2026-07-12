"""
Time series differencing utilities.

Provides diff() matching R's base::diff() and ndiffs() matching
R's forecast::ndiffs() for estimating the number of differences
needed for stationarity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.validation import check_array, check_1d, check_finite


def diff(
    x: ArrayLike,
    differences: int = 1,
    lag: int = 1,
) -> NDArray:
    """
    Difference a time series.

    Matches R's base::diff().

    For differences=1, lag=1: y[t] = x[t] - x[t-1]
    For differences=1, lag=12: y[t] = x[t] - x[t-12]  (seasonal)
    For differences=2: apply differencing twice.

    Parameters
    ----------
    x : ArrayLike
        Time series (1-D array).
    differences : int
        Number of times to difference. Default 1. Must be >= 1.
    lag : int
        Lag for differencing. Default 1. Must be >= 1.

    Returns
    -------
    NDArray
        Differenced series of length n - differences * lag.

    Raises
    ------
    ValidationError
        If inputs are invalid or the series is too short after differencing.
    """
    arr = check_array(x, "x")
    arr = arr.ravel()
    check_1d(arr, "x")
    check_finite(arr, "x")

    if not isinstance(differences, (int, np.integer)) or differences < 1:
        raise ValidationError(
            f"differences: must be a positive integer, got {differences}"
        )
    if not isinstance(lag, (int, np.integer)) or lag < 1:
        raise ValidationError(
            f"lag: must be a positive integer, got {lag}"
        )

    n = len(arr)
    required_length = differences * lag + 1
    if n < required_length:
        raise ValidationError(
            f"x: series of length {n} is too short for "
            f"differences={differences}, lag={lag} "
            f"(requires at least {required_length} observations)"
        )

    result = arr.copy()
    for _ in range(differences):
        result = result[lag:] - result[:-lag]

    return result


def ndiffs(
    x: ArrayLike,
    *,
    test: str = "kpss",
    significance_level: float = 0.05,
    max_d: int = 2,
) -> int:
    """
    Estimate the number of differences needed for stationarity.

    Matches R's forecast::ndiffs(), including its default test. Repeatedly
    differences and tests for stationarity until either the test indicates
    stationarity or max_d is reached.

    Parameters
    ----------
    x : ArrayLike
        Time series.
    test : str
        Stationarity test to use: 'kpss' (default, matching R
        forecast::ndiffs) or 'adf'. The two tests have opposite null
        hypotheses (KPSS: stationary; ADF: unit root) and can recommend
        different d on borderline series.
    significance_level : float
        Significance level for the test.
    max_d : int
        Maximum number of differences to try.

    Returns
    -------
    int
        Recommended number of differences (0, 1, ..., max_d).

    Raises
    ------
    ValidationError
        If inputs are invalid.
    """
    # Import here to avoid circular imports at module level
    from pystatistics.timeseries._stationarity import adf_test, kpss_test

    arr = check_array(x, "x")
    arr = arr.ravel()
    check_1d(arr, "x")
    check_finite(arr, "x")

    valid_tests = ("adf", "kpss")
    if test not in valid_tests:
        raise ValidationError(
            f"test: must be one of {valid_tests}, got '{test}'"
        )
    if not (0.0 < significance_level < 1.0):
        raise ValidationError(
            f"significance_level: must be in (0, 1), got {significance_level}"
        )
    if not isinstance(max_d, (int, np.integer)) or max_d < 0:
        raise ValidationError(
            f"max_d: must be a non-negative integer, got {max_d}"
        )

    current = arr.copy()
    for d in range(max_d + 1):
        if len(current) < 3:
            return d

        if test == "adf":
            # Pinned to the constant-only ("drift") variant that
            # forecast::ndiffs uses for its ADF test — adf_test's own
            # default is 'ct' (tseries::adf.test parity) and must not
            # leak into ndiffs semantics.
            result = adf_test(current, regression="c")
            # ADF: H0 = unit root. Reject H0 (p < significance_level) means stationary.
            if result.p_value < significance_level:
                return d
        else:
            # Pinned to forecast::ndiffs's bandwidth: it calls
            # urca::ur.kpss with use.lag = trunc(3*sqrt(n)/13), NOT the
            # tseries lshort rule that is now kpss_test's default.
            # Verified on WWWusage (n=100): forecast uses 2 lags and
            # returns d=1; the lshort default (4 lags) would flip it
            # to d=0.
            use_lag = int(np.trunc(3.0 * np.sqrt(len(current)) / 13.0))
            result = kpss_test(current, n_lags=use_lag)
            # KPSS: H0 = stationary. Fail to reject (p > significance_level) means stationary.
            if result.p_value > significance_level:
                return d

        # Difference once more
        if d < max_d:
            current = current[1:] - current[:-1]

    return max_d
