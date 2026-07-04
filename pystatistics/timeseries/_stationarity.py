"""
Stationarity tests for time series.

Provides adf_test() matching R's tseries::adf.test() and kpss_test()
matching R's tseries::kpss.test(). ADF p-values and critical values
come from the MacKinnon response surfaces (:mod:`_adf_mackinnon`);
KPSS p-values use linear interpolation in the Kwiatkowski et al.
(1992) critical-value table, exactly as tseries does.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.result import Result
from pystatistics.core.validation import check_array, check_1d, check_finite
from pystatistics.timeseries._adf_mackinnon import (
    adf_critical_values,
    adf_pvalue,
)
from pystatistics.timeseries._common import StationarityParams, StationaritySolution


def adf_test(
    x: ArrayLike,
    *,
    n_lags: int | None = None,
    regression: str = "ct",
) -> StationaritySolution:
    """
    Augmented Dickey-Fuller test for unit root.

    Matches R's tseries::adf.test() (statistic and lag convention) with
    MacKinnon (1994) p-values (the surface used by statsmodels
    ``adfuller`` and by ``urca::ur.df``'s critical values).

    Tests H0: x has a unit root (non-stationary)
    vs   H1: x is stationary

    The test regression is:
        Delta_x_t = alpha + beta*t + gamma*x_{t-1}
                    + sum_{i=1}^{p} delta_i * Delta_x_{t-i} + eps_t

    The test statistic is the t-statistic for gamma. The ``regression``
    parameter controls which deterministic terms are included:
    - 'nc': no constant, no trend
    - 'c': constant only
    - 'ct': constant + linear trend (default). This is what R's
      tseries::adf.test always uses; with matching ``n_lags`` the
      statistic reproduces it exactly.

    Parameters
    ----------
    x : ArrayLike
        Time series (1-D array). Must have at least 3 observations.
    n_lags : int or None
        Number of lagged difference terms. Default: floor((n-1)^(1/3)),
        matching R's tseries::adf.test().
    regression : str
        'nc' (none), 'c' (constant), 'ct' (constant + trend, default).

    Returns
    -------
    StationaritySolution
        Test result with statistic, p-value, and critical values.

    Raises
    ------
    ValidationError
        If inputs are invalid.

    Notes
    -----
    Statistic validated against both R tseries::adf.test() and
    statsmodels ``adfuller``; p-values validated against statsmodels
    (MacKinnon 1994 surface, full range — both tails and the middle).
    tseries::adf.test itself interpolates a small table and caps its
    p-values at [0.01, 0.99]; inside that range the two agree, outside
    it this implementation keeps resolving while tseries saturates.
    Critical values use the MacKinnon (2010) finite-sample surface.
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

    # Critical values (MacKinnon 2010 finite-sample surface) and
    # p-value (MacKinnon 1994 surface — valid across the whole range,
    # not interpolated from the 1/5/10% points; RIGOR R18 fixed a
    # near-unit-root series reporting p=0.44 where the correct value
    # is ~0.92).
    critical_values = adf_critical_values(regression, n_eff)
    p_value = adf_pvalue(float(tau), regression)

    return StationaritySolution(
        _result=Result(
            params=StationarityParams(
                statistic=float(tau),
                p_value=p_value,
                method="Augmented Dickey-Fuller",
                alternative="stationary",
                n_lags=n_lags,
                n_obs=n_eff,
                critical_values=critical_values,
            ),
            info={"method": "adf", "regression": regression},
            timing=None,
            backend_name="cpu",
            warnings=(),
        )
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

    # Unreachable for finite statistics (the clamps above cover both
    # tails); reachable only on NaN, which upstream validation blocks.
    raise ValidationError(
        f"KPSS p-value interpolation failed for statistic={statistic!r}"
    )


def kpss_test(
    x: ArrayLike,
    *,
    regression: str = "c",
    n_lags: int | None = None,
    lshort: bool = True,
) -> StationaritySolution:
    """
    KPSS test for stationarity.

    Matches R's tseries::kpss.test(), including its default bandwidth.

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
        'c' (level stationarity, tseries ``null="Level"``) or 'ct'
        (trend stationarity, tseries ``null="Trend"``).
    n_lags : int or None
        Number of lags for the Bartlett kernel. Default ``None`` uses
        the tseries::kpss.test rule selected by ``lshort``:
        ``trunc(4*(n/100)^(1/4))`` when ``lshort=True`` (tseries
        default) or ``trunc(12*(n/100)^(1/4))`` when ``lshort=False``.
        An explicit ``n_lags`` overrides ``lshort``.
    lshort : bool
        Bandwidth rule used when ``n_lags`` is ``None`` — the
        equivalent of tseries's ``lshort`` argument. Default ``True``.

    Returns
    -------
    StationaritySolution
        Test result with statistic, p-value, and critical values.

    Raises
    ------
    ValidationError
        If inputs are invalid.

    Notes
    -----
    Validated against R tseries::kpss.test() for both ``null="Level"``
    and ``null="Trend"``: at a matched bandwidth the statistic is
    reproduced exactly, and the p-value uses the same linear
    interpolation of the Kwiatkowski et al. (1992) table, clamped to
    tseries's reporting range [0.01, 0.10].
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
    if not isinstance(lshort, (bool, np.bool_)):
        raise ValidationError(
            f"lshort: must be a bool, got {type(lshort).__name__}"
        )

    if n_lags is None:
        # tseries::kpss.test bandwidth (RIGOR R18 — previously
        # floor(3*sqrt(n)/13), which disagreed with tseries at every n).
        factor = 4.0 if lshort else 12.0
        n_lags = int(np.trunc(factor * (n / 100.0) ** 0.25))
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

    # Fail loud on a degenerate regression: if the deterministic terms
    # fit x exactly (constant series under 'c', exactly linear series
    # under 'ct'), the statistic is 0/0 and any returned value would be
    # rounding noise dressed up as a test result.
    scale = float(np.max(np.abs(arr)))
    if np.max(np.abs(residuals)) <= 1e-12 * max(scale, 1.0):
        raise ValidationError(
            "x: the deterministic terms fit the series exactly "
            "(constant, or exactly linear with regression='ct'); "
            "the KPSS statistic is undefined."
        )

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
    critical_values = {}
    for alpha_level in sorted(table.keys()):
        pct = alpha_level * 100
        if pct == int(pct):
            key = f"{int(pct)}%"
        else:
            key = f"{pct}%"
        critical_values[key] = table[alpha_level]

    p_value = _kpss_pvalue(float(eta), regression)

    return StationaritySolution(
        _result=Result(
            params=StationarityParams(
                statistic=float(eta),
                p_value=p_value,
                method=f"KPSS Test for {'Level' if regression == 'c' else 'Trend'} Stationarity",
                alternative="unit root",
                n_lags=n_lags,
                n_obs=n,
                critical_values=critical_values,
            ),
            info={"method": "kpss", "regression": regression},
            timing=None,
            backend_name="cpu",
            warnings=(),
        )
    )
