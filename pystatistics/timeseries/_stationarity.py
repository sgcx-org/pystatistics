"""
Stationarity tests for time series.

Provides adf_test() matching R's tseries::adf.test() and kpss_test()
matching R's tseries::kpss.test(). Both use embedded critical value
tables and linear interpolation for p-value computation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.validation import check_array, check_1d, check_finite
from pystatistics.timeseries._common import StationarityResult


# ---------------------------------------------------------------------------
# ADF critical value tables (MacKinnon 1996)
# ---------------------------------------------------------------------------
# Tables indexed by sample size. Critical values for tau statistic at
# 1%, 5%, 10% significance levels.
# Source: MacKinnon, J.G. (1996), "Numerical Distribution Functions for
# Unit Root and Cointegration Tests", Journal of Applied Econometrics.
#
# These are the asymptotic (large-n) critical values widely used in
# the ADF literature.

_ADF_CRITICAL_VALUES = {
    "nc": {
        # No constant, no trend
        "1%": -2.58,
        "5%": -1.95,
        "10%": -1.62,
    },
    "c": {
        # Constant, no trend
        "1%": -3.43,
        "5%": -2.86,
        "10%": -2.57,
    },
    "ct": {
        # Constant + trend
        "1%": -3.96,
        "5%": -3.41,
        "10%": -3.12,
    },
}

# Finite-sample adjustment coefficients for ADF critical values.
# cv(n) = cv_inf + c1/n + c2/n^2
# From MacKinnon (1996) Table 1 (tau statistics).
_ADF_CV_ADJUSTMENTS = {
    "nc": {
        "1%": (-2.5658, -1.960, -10.04),
        "5%": (-1.9393, -0.398, 0.0),
        "10%": (-1.6156, -0.181, 0.0),
    },
    "c": {
        "1%": (-3.4336, -5.999, -29.25),
        "5%": (-2.8621, -2.738, -8.36),
        "10%": (-2.5671, -1.438, -4.48),
    },
    "ct": {
        "1%": (-3.9638, -8.353, -47.44),
        "5%": (-3.4126, -4.039, -17.83),
        "10%": (-3.1279, -2.418, -7.58),
    },
}


def _adf_critical_values_for_n(
    regression: str, n: int
) -> dict[str, float]:
    """
    Compute ADF critical values adjusted for finite sample size.

    Uses MacKinnon (1996) response surface regression:
        cv(n) = c_inf + c1/n + c2/n^2

    Parameters
    ----------
    regression : str
        Regression type: 'nc', 'c', or 'ct'.
    n : int
        Number of observations.

    Returns
    -------
    dict[str, float]
        Critical values at 1%, 5%, 10% levels.
    """
    adjustments = _ADF_CV_ADJUSTMENTS[regression]
    result = {}
    for level, (c_inf, c1, c2) in adjustments.items():
        result[level] = c_inf + c1 / n + c2 / (n * n)
    return result


def _adf_pvalue(statistic: float, regression: str, n: int) -> float:
    """
    Compute ADF p-value by interpolation from critical values.

    Uses linear interpolation between the standard critical value levels
    (1%, 5%, 10%). Values beyond the table range are clamped.

    Parameters
    ----------
    statistic : float
        ADF test statistic.
    regression : str
        Regression type.
    n : int
        Sample size.

    Returns
    -------
    float
        Approximate p-value.
    """
    cv = _adf_critical_values_for_n(regression, n)

    # Critical values sorted from most negative (1%) to least negative (10%)
    # p-values: 0.01, 0.05, 0.10
    cv_levels = [0.01, 0.05, 0.10]
    cv_vals = [cv["1%"], cv["5%"], cv["10%"]]

    # If statistic is more extreme than 1% CV
    if statistic <= cv_vals[0]:
        return 0.01

    # If statistic is less extreme than 10% CV
    if statistic >= cv_vals[2]:
        # Extrapolate toward p=1 using the slope between 5% and 10%
        slope = (0.10 - 0.05) / (cv_vals[2] - cv_vals[1])
        p = 0.10 + slope * (statistic - cv_vals[2])
        return min(max(p, 0.10), 0.9999)

    # Linear interpolation between adjacent CV levels
    for i in range(len(cv_vals) - 1):
        if cv_vals[i] <= statistic <= cv_vals[i + 1]:
            frac = (statistic - cv_vals[i]) / (cv_vals[i + 1] - cv_vals[i])
            return cv_levels[i] + frac * (cv_levels[i + 1] - cv_levels[i])

    # Fallback (should not reach here)
    return 0.5


def adf_test(
    x: ArrayLike,
    *,
    n_lags: int | None = None,
    regression: str = "c",
) -> StationarityResult:
    """
    Augmented Dickey-Fuller test for unit root.

    Matches R's tseries::adf.test().

    Tests H0: x has a unit root (non-stationary)
    vs   H1: x is stationary

    The test regression is:
        Delta_x_t = alpha + beta*t + gamma*x_{t-1}
                    + sum_{i=1}^{p} delta_i * Delta_x_{t-i} + eps_t

    The test statistic is the t-statistic for gamma. The ``regression``
    parameter controls which deterministic terms are included:
    - 'nc': no constant, no trend
    - 'c': constant only (default, matches R)
    - 'ct': constant + linear trend

    Parameters
    ----------
    x : ArrayLike
        Time series (1-D array). Must have at least 3 observations.
    n_lags : int or None
        Number of lagged difference terms. Default: floor((n-1)^(1/3)),
        matching R's tseries::adf.test().
    regression : str
        'nc' (none), 'c' (constant), 'ct' (constant + trend).

    Returns
    -------
    StationarityResult
        Test result with statistic, p-value, and critical values.

    Raises
    ------
    ValidationError
        If inputs are invalid.

    Notes
    -----
    Validated against R tseries::adf.test().
    """
    arr = check_array(x, "x")
    arr = arr.ravel()
    check_1d(arr, "x")
    check_finite(arr, "x")

    n = len(arr)
    if n < 3:
        raise ValidationError(
            f"x: requires at least 3 observations for ADF test, got {n}"
        )

    valid_regression = ("nc", "c", "ct")
    if regression not in valid_regression:
        raise ValidationError(
            f"regression: must be one of {valid_regression}, got '{regression}'"
        )

    if n_lags is None:
        n_lags = int(np.floor((n - 1) ** (1.0 / 3.0)))
    else:
        if not isinstance(n_lags, (int, np.integer)) or n_lags < 0:
            raise ValidationError(
                f"n_lags: must be a non-negative integer, got {n_lags}"
            )
        n_lags = int(n_lags)

    # Compute first differences
    dx = arr[1:] - arr[:-1]  # length n-1

    # Build the regression: Delta_x_t on x_{t-1} and lagged differences
    # Effective sample: t = n_lags+1, ..., n-1 (0-indexed in dx)
    # Number of usable observations
    n_eff = len(dx) - n_lags
    if n_eff < 3:
        raise ValidationError(
            f"x: too few observations ({n_eff}) after accounting for "
            f"n_lags={n_lags}. Need at least 3."
        )

    # Dependent variable: Delta_x_t for t = n_lags+1 ... n-1
    y = dx[n_lags:]

    # Build design matrix columns
    columns = []

    # x_{t-1} (the level term whose coefficient gamma we test)
    # For observation dx[t] = arr[t+1] - arr[t], x_{t-1} = arr[t]
    x_lag = arr[n_lags:-1]
    columns.append(x_lag)

    # Lagged differences: Delta_x_{t-1}, Delta_x_{t-2}, ..., Delta_x_{t-p}
    for i in range(1, n_lags + 1):
        columns.append(dx[n_lags - i: -i if i < len(dx) - n_lags + n_lags - i + 1 else len(dx) - i])

    # Rebuild lagged differences more carefully
    columns = [x_lag]
    for i in range(1, n_lags + 1):
        lag_col = dx[n_lags - i: n_lags - i + n_eff]
        columns.append(lag_col)

    # Deterministic terms
    if regression in ("c", "ct"):
        columns.append(np.ones(n_eff))
    if regression == "ct":
        columns.append(np.arange(n_lags + 1, n_lags + 1 + n_eff, dtype=np.float64))

    X = np.column_stack(columns)

    # OLS via numpy.linalg.lstsq
    coeffs, residuals_arr, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    # Compute residuals and standard errors
    y_hat = X @ coeffs
    resid = y - y_hat
    dof = n_eff - X.shape[1]

    if dof <= 0:
        raise ValidationError(
            f"x: insufficient degrees of freedom ({dof}) for ADF test. "
            f"Reduce n_lags or provide a longer series."
        )

    sigma2 = np.dot(resid, resid) / dof

    # Standard error of gamma (first coefficient)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        raise ValidationError(
            "x: design matrix is singular in ADF test. "
            "The series may be constant or nearly constant."
        )

    se_gamma = np.sqrt(sigma2 * XtX_inv[0, 0])

    if se_gamma < 1e-15:
        raise ValidationError(
            "x: standard error of gamma is essentially zero in ADF test. "
            "The series may be constant."
        )

    tau = coeffs[0] / se_gamma

    # Critical values and p-value
    critical_values = _adf_critical_values_for_n(regression, n_eff)
    p_value = _adf_pvalue(float(tau), regression, n_eff)

    return StationarityResult(
        statistic=float(tau),
        p_value=p_value,
        method="Augmented Dickey-Fuller",
        alternative="stationary",
        n_lags=n_lags,
        n_obs=n_eff,
        critical_values=critical_values,
    )


# ---------------------------------------------------------------------------
# KPSS critical value tables
# ---------------------------------------------------------------------------
# From Kwiatkowski, Phillips, Schmidt, and Shin (1992), Table 1.

_KPSS_CRITICAL_VALUES = {
    "c": {
        # Level stationarity: critical values at alpha levels
        # alpha: critical_value (reject if statistic > cv)
        0.10: 0.347,
        0.05: 0.463,
        0.025: 0.574,
        0.01: 0.739,
    },
    "ct": {
        # Trend stationarity
        0.10: 0.119,
        0.05: 0.146,
        0.025: 0.176,
        0.01: 0.216,
    },
}


def _kpss_pvalue(statistic: float, regression: str) -> float:
    """
    Compute KPSS p-value by linear interpolation from critical value tables.

    The KPSS test has a limited table of critical values. P-values beyond
    the table range are clamped to 0.01 or 0.10.

    Parameters
    ----------
    statistic : float
        KPSS test statistic.
    regression : str
        'c' or 'ct'.

    Returns
    -------
    float
        Approximate p-value, clamped to [0.01, 0.10].
    """
    table = _KPSS_CRITICAL_VALUES[regression]

    # Sort by critical value (ascending)
    # Higher statistic -> lower p-value (more evidence against H0)
    alphas = sorted(table.keys(), reverse=True)  # 0.10, 0.05, 0.025, 0.01
    cv_vals = [table[a] for a in alphas]  # ascending CVs

    # If statistic is below the smallest CV (alpha=0.10), p > 0.10
    if statistic < cv_vals[0]:
        return 0.10

    # If statistic is above the largest CV (alpha=0.01), p < 0.01
    if statistic > cv_vals[-1]:
        return 0.01

    # Linear interpolation
    for i in range(len(cv_vals) - 1):
        if cv_vals[i] <= statistic <= cv_vals[i + 1]:
            frac = (statistic - cv_vals[i]) / (cv_vals[i + 1] - cv_vals[i])
            p = alphas[i] - frac * (alphas[i] - alphas[i + 1])
            return p

    return 0.05  # fallback


def kpss_test(
    x: ArrayLike,
    *,
    regression: str = "c",
    n_lags: int | None = None,
) -> StationarityResult:
    """
    KPSS test for stationarity.

    Matches R's tseries::kpss.test().

    Tests H0: x is (level or trend) stationary
    vs   H1: x has a unit root

    NOTE: KPSS has the OPPOSITE null hypothesis to ADF.

    Algorithm:
        1. Regress x on deterministic terms (constant, or constant+trend).
        2. Compute partial sums S_t = sum_{i=1}^{t} e_i of residuals.
        3. Statistic: eta = (1/n^2) * sum S_t^2 / sigma^2_LR
           where sigma^2_LR is the long-run variance estimator using a
           Bartlett kernel:
           sigma^2_LR = gamma(0) + 2 * sum_{j=1}^{l} (1 - j/(l+1)) * gamma(j)

    Parameters
    ----------
    x : ArrayLike
        Time series (1-D array). Must have at least 3 observations.
    regression : str
        'c' (level stationarity) or 'ct' (trend stationarity).
    n_lags : int or None
        Number of lags for Bartlett kernel. Default: floor(3*sqrt(n)/13),
        matching R.

    Returns
    -------
    StationarityResult
        Test result with statistic, p-value, and critical values.

    Raises
    ------
    ValidationError
        If inputs are invalid.

    Notes
    -----
    Validated against R tseries::kpss.test().
    """
    arr = check_array(x, "x")
    arr = arr.ravel()
    check_1d(arr, "x")
    check_finite(arr, "x")

    n = len(arr)
    if n < 3:
        raise ValidationError(
            f"x: requires at least 3 observations for KPSS test, got {n}"
        )

    valid_regression = ("c", "ct")
    if regression not in valid_regression:
        raise ValidationError(
            f"regression: must be one of {valid_regression}, got '{regression}'"
        )

    if n_lags is None:
        n_lags = int(np.floor(3.0 * np.sqrt(n) / 13.0))
    else:
        if not isinstance(n_lags, (int, np.integer)) or n_lags < 0:
            raise ValidationError(
                f"n_lags: must be a non-negative integer, got {n_lags}"
            )
        n_lags = int(n_lags)

    # Step 1: Regress x on deterministic terms
    t_vals = np.arange(1, n + 1, dtype=np.float64)
    if regression == "c":
        Z = np.ones((n, 1))
    else:
        Z = np.column_stack([np.ones(n), t_vals])

    coeffs, _, _, _ = np.linalg.lstsq(Z, arr, rcond=None)
    residuals = arr - Z @ coeffs

    # Step 2: Partial sums of residuals
    S = np.cumsum(residuals)

    # Step 3: Long-run variance with Bartlett kernel
    # gamma(j) = (1/n) * sum_{t=j+1}^{n} e_t * e_{t-j}
    gamma_0 = np.dot(residuals, residuals) / n

    sigma2_lr = gamma_0
    for j in range(1, n_lags + 1):
        gamma_j = np.dot(residuals[j:], residuals[:-j]) / n
        weight = 1.0 - j / (n_lags + 1.0)
        sigma2_lr += 2.0 * weight * gamma_j

    if sigma2_lr <= 0.0:
        raise ValidationError(
            "x: long-run variance estimate is non-positive in KPSS test. "
            "Try a different number of lags."
        )

    # KPSS statistic
    eta = np.dot(S, S) / (n * n * sigma2_lr)

    # Critical values and p-value
    table = _KPSS_CRITICAL_VALUES[regression]
    critical_values = {
        f"{int(alpha * 100)}%" if alpha * 100 == int(alpha * 100) else f"{alpha * 100}%": cv
        for alpha, cv in sorted(table.items())
    }
    # Clean up the keys to standard format
    critical_values = {}
    for alpha_level in sorted(table.keys()):
        pct = alpha_level * 100
        if pct == int(pct):
            key = f"{int(pct)}%"
        else:
            key = f"{pct}%"
        critical_values[key] = table[alpha_level]

    p_value = _kpss_pvalue(float(eta), regression)

    return StationarityResult(
        statistic=float(eta),
        p_value=p_value,
        method=f"KPSS Test for {'Level' if regression == 'c' else 'Trend'} Stationarity",
        alternative="unit root",
        n_lags=n_lags,
        n_obs=n,
        critical_values=critical_values,
    )
