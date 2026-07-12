"""
ARIMA log-likelihood computation.

Provides conditional sum of squares (CSS) and exact maximum likelihood
(ML) for ARMA(p, q) models on differenced data. Also provides
stationarity/invertibility checking for AR/MA polynomials.

The CSS approach reconstructs residuals recursively:
    e_t = y_t - mu - sum_i phi_i * (y_{t-i} - mu) - sum_j theta_j * e_{t-j}
conditioning on e_0 = ... = e_{-q} = 0.

Exact ML delegates to the state-space / Kalman-filter implementation in
``_arima_kalman`` (Gardner, Harvey & Phillips 1980), the same approach
R's ``stats::arima`` uses.

Design: self-contained with no imports from regression or other pystatistics
submodules beyond core validation/exceptions.
"""

from __future__ import annotations

from pystatistics.core.exceptions import ValidationError

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.signal import lfilter


# ---------------------------------------------------------------------------
# CSS residuals and log-likelihood
# ---------------------------------------------------------------------------

def arima_css_residuals(
    y: NDArray,
    ar: NDArray,
    ma: NDArray,
    mean: float,
) -> NDArray:
    """
    Compute CSS residuals for an ARMA model.

    Given a (possibly differenced) series *y* and AR/MA coefficients,
    reconstruct one-step-ahead prediction errors by conditioning on
    zero initial errors.

    The recursion is:
        e_t = (y_t - mean)
              - sum_{i=1}^{p} ar_i * (y_{t-i} - mean)
              - sum_{j=1}^{q} ma_j * e_{t-j}

    Parameters
    ----------
    y : NDArray
        1-D series (already differenced if d > 0).
    ar : NDArray
        AR coefficients [phi_1, ..., phi_p]. Empty array if p = 0.
    ma : NDArray
        MA coefficients [theta_1, ..., theta_q]. Empty array if q = 0.
    mean : float
        Estimated mean of the series (0.0 if no mean term).

    Returns
    -------
    NDArray
        Residual array of length ``len(y)``. The first ``max(p, q)``
        residuals are conditioned on zero initial errors and may be
        less reliable.
    """
    # The CSS recursion
    #     e[t] = (y[t] - mean) - sum ar_i (y[t-i] - mean) - sum ma_j e[t-j]
    # with e[<0] = 0 is exactly the difference equation implemented by
    # scipy.signal.lfilter (IIR form):
    #     a[0] y_out[t] + sum_{k>=1} a[k] y_out[t-k]
    #         = sum_{k>=0} b[k] x[t-k]
    # Mapping y_out -> e, x -> y_centered:
    #     a = [1, ma_1, ma_2, ..., ma_q]          (AR side on the output e)
    #     b = [1, -ar_1, -ar_2, ..., -ar_p]       (MA side on the input y)
    # lfilter runs the recursion in compiled C; this removes ~500k Python
    # `np.dot` calls per SARIMA fit on AirPassengers (the new hotspot
    # after the earlier vectorization pass).
    y_centered = y - mean
    a = np.concatenate(([1.0], ma)) if len(ma) else np.array([1.0])
    b = np.concatenate(([1.0], -ar)) if len(ar) else np.array([1.0])
    return lfilter(b, a, y_centered)


def css_loglik(
    params: NDArray,
    y: NDArray,
    order: tuple[int, int],
    include_mean: bool,
) -> float:
    """
    Conditional sum of squares log-likelihood (negated for minimization).

    CSS log-likelihood:
        L_css = -(n/2) * log(2*pi) - (n/2) * log(sigma2_css) - n/2
    where sigma2_css = (1/n) * sum(e_t^2).

    Parameters
    ----------
    params : NDArray
        Parameter vector laid out as ``[ar_1..ar_p, ma_1..ma_q, mean?]``.
    y : NDArray
        Differenced series.
    order : tuple[int, int]
        ``(p, q)`` — AR and MA orders for the differenced series.
    include_mean : bool
        Whether the last element of *params* is the mean.

    Returns
    -------
    float
        Negative log-likelihood (for minimization).
    """
    p, q = order
    ar, ma, mean = _unpack_params(params, p, q, include_mean)

    residuals = arima_css_residuals(y, ar, ma, mean)

    n = len(residuals)
    sse = np.dot(residuals, residuals)
    sigma2 = sse / n

    if sigma2 <= 0.0:
        return 1e18

    nll = 0.5 * n * np.log(2.0 * np.pi) + 0.5 * n * np.log(sigma2) + 0.5 * n
    return nll


# ---------------------------------------------------------------------------
# Exact ML via the Kalman filter
# ---------------------------------------------------------------------------

def exact_loglik(
    params: NDArray,
    y: NDArray,
    order: tuple[int, int],
    include_mean: bool,
) -> float:
    """
    Exact Gaussian log-likelihood via the Kalman filter (negated).

    Uses the state-space / Kalman-filter implementation in
    ``_arima_kalman.kalman_arma_loglik``, which is O(n * r^2) per fit
    (r = max(p, q+1)) and matches R's ``stats::arima`` approach (Gardner,
    Harvey & Phillips 1980).

    Parameters
    ----------
    params : NDArray
        Parameter vector: ``[ar_1..ar_p, ma_1..ma_q, mean?]``.
    y : NDArray
        Differenced series.
    order : tuple[int, int]
        ``(p, q)`` — AR and MA orders.
    include_mean : bool
        Whether *params* includes a mean term as the last element.

    Returns
    -------
    float
        Negative log-likelihood (for minimization).
    """
    from pystatistics.timeseries._arima_kalman import kalman_arma_loglik

    p, q = order
    ar, ma, mean = _unpack_params(params, p, q, include_mean)
    nll, _ = kalman_arma_loglik(y, ar, ma, mean)
    return nll


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def arima_negloglik(
    params: NDArray,
    y: NDArray,
    order: tuple[int, int],
    include_mean: bool,
    method: str,
) -> float:
    """
    Negative log-likelihood for ARMA optimization.

    Dispatches to CSS or exact ML based on *method*.

    Parameters
    ----------
    params : NDArray
        Parameter vector: ``[ar_1..ar_p, ma_1..ma_q, mean?]``.
    y : NDArray
        Differenced series.
    order : tuple[int, int]
        ``(p, q)`` — AR and MA orders.
    include_mean : bool
        Whether *params* includes a mean term.
    method : str
        ``'css'`` or ``'ml'``.

    Returns
    -------
    float
        Negative log-likelihood.

    Raises
    ------
    ValueError
        If *method* is not ``'css'`` or ``'ml'``.
    """
    if method == "css":
        return css_loglik(params, y, order, include_mean)
    if method == "ml":
        return exact_loglik(params, y, order, include_mean)
    raise ValidationError(f"method must be 'css' or 'ml', got '{method}'")


def minimize_quiet(*args, **kwargs):
    """``scipy.optimize.minimize`` with numpy invalid/overflow FP warnings
    silenced for the duration of the call.

    L-BFGS-B estimates gradients by finite differences.  When its line
    search probes non-stationary / non-invertible parameters, the exact-ML
    objective returns a non-finite value (``inf``) and scipy's ``numdiff``
    subtracts ``inf - inf``, so numpy emits a benign
    ``RuntimeWarning: invalid value encountered in subtract``.  The
    non-finite return is *intentional* — it steers the optimizer out of the
    infeasible region and the convergence guards in the ARIMA fitters rely
    on seeing it — so the objective is left untouched and only the cosmetic
    numpy warning is suppressed here.  No numerical result changes.
    """
    with np.errstate(invalid="ignore", over="ignore"):
        return minimize(*args, **kwargs)


def arima_gradient(
    params: NDArray,
    y: NDArray,
    order: tuple[int, int],
    include_mean: bool,
    method: str,
    step: float = 1e-5,
) -> NDArray:
    """
    Numerical gradient of the negative log-likelihood via central differences.

    Parameters
    ----------
    params : NDArray
        Parameter vector.
    y : NDArray
        Differenced series.
    order : tuple[int, int]
        ``(p, q)`` — AR and MA orders.
    include_mean : bool
        Whether *params* includes a mean term.
    method : str
        ``'css'`` or ``'ml'``.
    step : float
        Finite difference step size. Default ``1e-5``.

    Returns
    -------
    NDArray
        Gradient vector (same length as *params*).
    """
    grad = np.empty(len(params))
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += step
        params_minus[i] -= step
        f_plus = arima_negloglik(params_plus, y, order, include_mean, method)
        f_minus = arima_negloglik(params_minus, y, order, include_mean, method)
        grad[i] = (f_plus - f_minus) / (2.0 * step)
    return grad


def compute_numerical_hessian(
    fun,
    params: NDArray,
    step: float = 1e-4,
) -> NDArray:
    """
    Compute the numerical Hessian of ``fun`` at ``params``.

    Uses second-order central finite differences. ``fun`` maps a
    parameter vector to a scalar negative log-likelihood; callers supply
    a closure over the data and likelihood method, which lets seasonal
    (or regression) fits differentiate in whichever parameterization the
    reported coefficients live in.

    Parameters
    ----------
    fun : callable
        Scalar objective ``fun(params) -> float``.
    params : NDArray
        Point at which to evaluate the Hessian.
    step : float
        Finite difference step size.

    Returns
    -------
    NDArray
        Hessian matrix (k x k).
    """
    k = len(params)
    hessian = np.zeros((k, k))

    for i in range(k):
        for j in range(i, k):
            params_pp = params.copy()
            params_pm = params.copy()
            params_mp = params.copy()
            params_mm = params.copy()

            params_pp[i] += step
            params_pp[j] += step
            params_pm[i] += step
            params_pm[j] -= step
            params_mp[i] -= step
            params_mp[j] += step
            params_mm[i] -= step
            params_mm[j] -= step

            fpp = fun(params_pp)
            fpm = fun(params_pm)
            fmp = fun(params_mp)
            fmm = fun(params_mm)

            hessian[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * step * step)
            hessian[j, i] = hessian[i, j]

    return hessian


# ---------------------------------------------------------------------------
# Stationarity and invertibility checks
# ---------------------------------------------------------------------------

def check_stationary(ar_coeffs: NDArray) -> bool:
    """
    Check that the AR polynomial has all roots outside the unit circle.

    The characteristic polynomial is:
        1 - phi_1*z - phi_2*z^2 - ... - phi_p*z^p

    Stationarity requires all roots z_k to satisfy |z_k| > 1.

    Parameters
    ----------
    ar_coeffs : NDArray
        AR coefficients [phi_1, ..., phi_p].

    Returns
    -------
    bool
        ``True`` if all roots lie strictly outside the unit circle.
    """
    if len(ar_coeffs) == 0:
        return True

    # Polynomial coefficients for numpy: highest power first
    # p(z) = -phi_p * z^p - ... - phi_1 * z + 1
    poly = np.concatenate(([-ar_coeffs[-1 - i] for i in range(len(ar_coeffs))], [1.0]))
    roots = np.roots(poly)
    return bool(np.all(np.abs(roots) > 1.0))


def check_invertible(ma_coeffs: NDArray) -> bool:
    """
    Check that the MA polynomial has all roots outside the unit circle.

    The characteristic polynomial is:
        1 + theta_1*z + theta_2*z^2 + ... + theta_q*z^q

    Invertibility requires all roots z_k to satisfy |z_k| > 1.

    Parameters
    ----------
    ma_coeffs : NDArray
        MA coefficients [theta_1, ..., theta_q].

    Returns
    -------
    bool
        ``True`` if all roots lie strictly outside the unit circle.
    """
    if len(ma_coeffs) == 0:
        return True

    # Polynomial: theta_q * z^q + ... + theta_1 * z + 1
    poly = np.concatenate(([ma_coeffs[-1 - i] for i in range(len(ma_coeffs))], [1.0]))
    roots = np.roots(poly)
    return bool(np.all(np.abs(roots) > 1.0))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unpack_params(
    params: NDArray,
    p: int,
    q: int,
    include_mean: bool,
) -> tuple[NDArray, NDArray, float]:
    """
    Unpack the flat parameter vector into AR, MA, and mean components.

    Parameters
    ----------
    params : NDArray
        Flat parameter vector ``[ar_1..ar_p, ma_1..ma_q, mean?]``.
    p : int
        AR order.
    q : int
        MA order.
    include_mean : bool
        Whether a mean term is appended to *params*.

    Returns
    -------
    tuple[NDArray, NDArray, float]
        ``(ar, ma, mean)`` — AR coefficients, MA coefficients, and mean.
    """
    ar = params[:p]
    ma = params[p:p + q]
    mean = params[p + q] if include_mean else 0.0
    return ar, ma, mean
