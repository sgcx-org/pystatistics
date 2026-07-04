"""
MacKinnon p-value and critical-value surfaces for the ADF test.

Implements the response-surface approximations for the (Augmented)
Dickey-Fuller tau statistic with a single I(1) series (N = 1):

- :func:`adf_pvalue` — MacKinnon (1994) approximate asymptotic p-value
  surface. This is the same surface used by statsmodels
  ``adfuller`` and (for its critical values) R's ``urca::ur.df``,
  and is valid across the whole p range — both tails and the middle.
- :func:`adf_critical_values` — MacKinnon (2010) finite-sample
  critical values at the 1% / 5% / 10% levels via the response
  surface ``cv(n) = b0 + b1/n + b2/n^2 + b3/n^3``. (The 'nc' row was
  not re-estimated in 2010 and comes from the 1996 paper.)

Kept in its own module so :mod:`_stationarity` holds the test
implementations only (one module, one job); the tables below are pure
published constants.

References
----------
MacKinnon, J.G. (1994). "Approximate Asymptotic Distribution Functions
for Unit-Root and Cointegration Tests." Journal of Business & Economic
Statistics, 12(2), 167-176.

MacKinnon, J.G. (1996). "Numerical Distribution Functions for Unit
Root and Cointegration Tests." Journal of Applied Econometrics, 11(6),
601-618.

MacKinnon, J.G. (2010). "Critical Values for Cointegration Tests."
Queen's Economics Department Working Paper No. 1227.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from pystatistics.core.exceptions import ValidationError

# ---------------------------------------------------------------------------
# MacKinnon (1994) tau p-value surface, N = 1
# ---------------------------------------------------------------------------
# The p-value is Phi(g(tau)) where g is a low-order polynomial in tau.
# Two polynomials per regression type: a "small-p" fit used in the
# left tail (tau <= tau_star) and a "large-p" fit used elsewhere.
# Outside [tau_min, tau_max] the p-value saturates at 0 / 1.
# Coefficients are ascending (c0, c1, c2[, c3]): g(t) = c0 + c1*t + ...

_TAU_MAX = {"nc": np.inf, "c": 2.74, "ct": 0.70}
_TAU_MIN = {"nc": -19.04, "c": -18.83, "ct": -16.18}
_TAU_STAR = {"nc": -1.04, "c": -1.61, "ct": -2.89}

_TAU_SMALLP = {
    "nc": (0.6344, 1.2378, 0.032496),
    "c": (2.1659, 1.4412, 0.038269),
    "ct": (3.2512, 1.6047, 0.049588),
}

_TAU_LARGEP = {
    "nc": (0.4797, 0.93557, -0.06999, 0.033066),
    "c": (1.7339, 0.93202, -0.12745, -0.010368),
    "ct": (2.5261, 0.61654, -0.37956, -0.060285),
}

# ---------------------------------------------------------------------------
# MacKinnon (2010) finite-sample critical values, N = 1
# ---------------------------------------------------------------------------
# cv(n) = b0 + b1/n + b2/n^2 + b3/n^3 per (regression, level).

_CRIT_SURFACE = {
    "nc": {
        "1%": (-2.56574, -2.2358, -3.627, 0.0),
        "5%": (-1.94100, -0.2686, -3.365, 31.223),
        "10%": (-1.61682, 0.2656, -2.714, 25.364),
    },
    "c": {
        "1%": (-3.43035, -6.5393, -16.786, -79.433),
        "5%": (-2.86154, -2.8903, -4.234, -40.040),
        "10%": (-2.56677, -1.5384, -2.809, 0.0),
    },
    "ct": {
        "1%": (-3.95877, -9.0531, -28.428, -134.155),
        "5%": (-3.41049, -4.3904, -9.036, -45.374),
        "10%": (-3.12705, -2.5856, -3.925, -22.380),
    },
}


def _check_regression(regression: str) -> None:
    """Fail loudly on an unknown regression type (Rule 1)."""
    if regression not in _TAU_STAR:
        raise ValidationError(
            f"regression: must be one of {tuple(_TAU_STAR)}, "
            f"got '{regression}'"
        )


def adf_pvalue(statistic: float, regression: str) -> float:
    """
    MacKinnon (1994) approximate p-value for an ADF tau statistic.

    Parameters
    ----------
    statistic : float
        ADF t-statistic (tau).
    regression : str
        Deterministic terms in the test regression: ``'nc'`` (none),
        ``'c'`` (constant), or ``'ct'`` (constant + trend).

    Returns
    -------
    float
        Approximate asymptotic p-value in [0, 1]. Saturates at exactly
        0.0 / 1.0 beyond the surface's fitted range.

    Raises
    ------
    ValidationError
        If ``regression`` is not one of 'nc', 'c', 'ct'.
    """
    _check_regression(regression)

    if statistic > _TAU_MAX[regression]:
        return 1.0
    if statistic < _TAU_MIN[regression]:
        return 0.0

    if statistic <= _TAU_STAR[regression]:
        coefs = _TAU_SMALLP[regression]
    else:
        coefs = _TAU_LARGEP[regression]

    g = 0.0
    for i, c in enumerate(coefs):
        g += c * statistic ** i
    return float(norm.cdf(g))


def adf_critical_values(regression: str, nobs: int) -> dict[str, float]:
    """
    MacKinnon (2010) finite-sample ADF critical values.

    Parameters
    ----------
    regression : str
        ``'nc'``, ``'c'``, or ``'ct'``.
    nobs : int
        Effective sample size of the ADF regression (observations
        actually used, after lags and differencing).

    Returns
    -------
    dict[str, float]
        Critical values at the ``'1%'``, ``'5%'``, ``'10%'`` levels.

    Raises
    ------
    ValidationError
        If ``regression`` is not one of 'nc', 'c', 'ct', or ``nobs``
        is not positive.
    """
    _check_regression(regression)
    if nobs <= 0:
        raise ValidationError(f"nobs: must be positive, got {nobs}")

    out: dict[str, float] = {}
    for level, (b0, b1, b2, b3) in _CRIT_SURFACE[regression].items():
        x = 1.0 / nobs
        out[level] = b0 + b1 * x + b2 * x * x + b3 * x ** 3
    return out
