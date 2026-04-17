"""
ARIMA log-likelihood computation.

Provides conditional sum of squares (CSS) and exact maximum likelihood (ML)
via the innovations algorithm for ARMA(p, q) models on differenced data.
Also provides stationarity/invertibility checking for AR/MA polynomials.

The CSS approach reconstructs residuals recursively:
    e_t = y_t - mu - sum_i phi_i * (y_{t-i} - mu) - sum_j theta_j * e_{t-j}
conditioning on e_0 = ... = e_{-q} = 0.

The exact ML approach uses the innovations algorithm (Brockwell & Davis, 2002)
to compute prediction errors and their variances from the ARMA autocovariance
structure, yielding the exact Gaussian log-likelihood.

Design: self-contained with no imports from regression or other pystatistics
submodules beyond core validation/exceptions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
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
# Exact ML via innovations algorithm
# ---------------------------------------------------------------------------

def _arma_autocovariance(
    ar: NDArray,
    ma: NDArray,
    sigma2: float,
    max_lag: int,
) -> NDArray:
    """
    Compute the autocovariance function of an ARMA(p, q) process.

    Uses the method of Brockwell & Davis (2002, Section 3.3).
    For an ARMA(p, q) with innovation variance sigma2:
        gamma(h) for h = 0, 1, ..., max_lag.

    The approach solves the Yule-Walker-type system for gamma(0..m-1)
    where m = max(p, q+1), then recursively computes higher lags.

    Parameters
    ----------
    ar : NDArray
        AR coefficients [phi_1, ..., phi_p].
    ma : NDArray
        MA coefficients [theta_1, ..., theta_q].
    sigma2 : float
        Innovation variance.
    max_lag : int
        Maximum lag for which to compute the autocovariance.

    Returns
    -------
    NDArray
        Autocovariances gamma(0), gamma(1), ..., gamma(max_lag).
    """
    p = len(ar)
    q = len(ma)
    m = max(p, q + 1)

    # Build the MA representation psi weights: psi_0 = 1, ...
    # psi_j = ma_j + sum_{i=1}^{min(j,p)} ar_i * psi_{j-i}  for j >= 1
    n_psi = m + 1
    psi = np.zeros(n_psi)
    psi[0] = 1.0
    for j in range(1, n_psi):
        val = 0.0
        if j <= q:
            val += ma[j - 1]
        for i in range(1, min(j, p) + 1):
            val += ar[i - 1] * psi[j - i]
        psi[j] = val

    # Compute cross-covariances sigma2 * sum_{j=0}^{q} theta_j * psi_{h+j}
    # where theta_0 = 1
    theta = np.concatenate(([1.0], ma))

    # Build the linear system A * gamma = b for gamma(0..m-1)
    # From the Yule-Walker equations for ARMA:
    # gamma(h) - sum_{i=1}^{p} ar_i * gamma(h-i) = sigma2 * sum_{j=h}^{q} theta_j * psi_{j-h}
    # for h = 0, 1, ..., m-1
    A = np.zeros((m, m))
    b = np.zeros(m)

    for h in range(m):
        A[h, h] = 1.0
        for i in range(1, p + 1):
            idx = abs(h - i)
            if idx < m:
                A[h, idx] -= ar[i - 1]

        # Right-hand side
        rhs = 0.0
        for j in range(h, q + 1):
            if j - h < n_psi:
                rhs += theta[j] * psi[j - h]
        b[h] = sigma2 * rhs

    gamma_base = np.linalg.solve(A, b)

    # Build full gamma array
    gamma = np.zeros(max_lag + 1)
    for h in range(min(m, max_lag + 1)):
        gamma[h] = gamma_base[h]

    # Recursion for h >= m: gamma(h) = sum_{i=1}^{p} ar_i * gamma(h-i)
    for h in range(m, max_lag + 1):
        val = 0.0
        for i in range(1, p + 1):
            val += ar[i - 1] * gamma[h - i]
        gamma[h] = val

    return gamma


def _innovations_algorithm(
    gamma: NDArray,
    n: int,
) -> tuple[NDArray, NDArray]:
    """
    Run the innovations algorithm given autocovariances.

    Computes the one-step-ahead prediction coefficients and
    prediction error variances for observations y_1, ..., y_n.

    Parameters
    ----------
    gamma : NDArray
        Autocovariances gamma(0), gamma(1), ..., gamma(n-1) at minimum.
    n : int
        Number of observations.

    Returns
    -------
    tuple[NDArray, NDArray]
        theta_mat : (n, n) array where theta_mat[t, j] is the coefficient
            for the j-th innovation in predicting y_{t+1}.
        v : (n,) array of prediction error variances v_0, ..., v_{n-1}.
    """
    theta_mat = np.zeros((n, n))
    v = np.zeros(n)
    v[0] = gamma[0]

    # This was the #1 hotspot: O(n^3) triple-nested Python loop with a
    # scalar np.clip and a builtin min() inside the innermost body. For
    # n=132 that was 127k scalar-clip calls and 792k min calls per fit.
    # We keep the outer (t) and middle (k) loops sequential — their
    # results feed each other — and vectorize the inner (j) dot product.
    # The numerical guards (clip theta, min contrib, max v_t) are kept
    # as Python-level comparisons which avoid numpy-scalar overhead.
    CLIP_THETA = 1e10
    CLIP_CONTRIB = 1e20
    for t in range(1, n):
        # theta_{t, t-1-k} for k = 0, ..., t-1
        for k in range(t):
            s = t - 1 - k
            val = gamma[t - k]
            if k > 0:
                # inner j-sum for j = 0..k-1 of
                #   theta_mat[t, t-1-j] * theta_mat[k, k-1-j] * v[j]
                # Indices t-1..t-k and k-1..0 — reversed slices of length k.
                t_slice = theta_mat[t, t - k:t][::-1]
                k_slice = theta_mat[k, :k][::-1]
                val -= np.dot(t_slice * k_slice, v[:k])
            if v[k] <= 0.0:
                theta_mat[t, s] = 0.0
            else:
                th = val / v[k]
                if th > CLIP_THETA:
                    th = CLIP_THETA
                elif th < -CLIP_THETA:
                    th = -CLIP_THETA
                theta_mat[t, s] = th

        # v_t = gamma(0) - sum_{j=0..t-1} theta_{t,j}^2 * v_j
        row = theta_mat[t, :t]
        contribs = row * row * v[:t]
        # Clip any contribution that overflowed to CLIP_CONTRIB.
        # (Scalar np.clip on individual contribs was the prior hotspot;
        #  array-level np.minimum is a single numpy call.)
        np.minimum(contribs, CLIP_CONTRIB, out=contribs)
        v_t = gamma[0] - contribs.sum()
        v[t] = v_t if v_t > 1e-15 else 1e-15

    return theta_mat, v


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
    Harvey & Phillips 1980). An earlier O(n^3) innovations-algorithm
    implementation is retained below for reference and as a fallback.

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


def _exact_loglik_innovations(
    params: NDArray,
    y: NDArray,
    order: tuple[int, int],
    include_mean: bool,
) -> float:
    """Deprecated O(n^3) innovations-algorithm implementation.

    Retained as a second-source reference for the Kalman-filter exact-ML
    path and for regression testing (the Kalman log-likelihood must agree
    with this to ~1e-6 at any stationary parameter point).
    """
    p, q = order
    ar, ma, mean = _unpack_params(params, p, q, include_mean)

    z = y - mean
    n = len(z)

    css_resid = arima_css_residuals(y, ar, ma, mean)
    sigma2_est = np.dot(css_resid, css_resid) / n
    if sigma2_est <= 0.0:
        return 1e18

    gamma = _arma_autocovariance(ar, ma, sigma2_est, n - 1)
    theta_mat, v = _innovations_algorithm(gamma, n)

    if np.any(np.isnan(v)):
        return 1e15

    yhat = np.zeros(n)
    errors = np.zeros(n)
    errors[0] = z[0]

    with np.errstate(over='ignore', invalid='ignore'):
        for t in range(1, n):
            yhat[t] = np.dot(theta_mat[t, :t], errors[:t][::-1])
            errors[t] = z[t] - yhat[t]

        if np.any(v <= 0.0):
            return 1e18
        nll = (
            0.5 * n * np.log(2.0 * np.pi)
            + 0.5 * np.sum(np.log(v))
            + 0.5 * np.sum(errors * errors / v)
        )

    # NUMERICAL GUARD: if log-likelihood is NaN or Inf, return penalty
    if not np.isfinite(nll):
        return 1e15

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
        ``'CSS'`` or ``'ML'``.

    Returns
    -------
    float
        Negative log-likelihood.

    Raises
    ------
    ValueError
        If *method* is not ``'CSS'`` or ``'ML'``.
    """
    if method == "CSS":
        return css_loglik(params, y, order, include_mean)
    if method == "ML":
        return exact_loglik(params, y, order, include_mean)
    raise ValueError(f"method must be 'CSS' or 'ML', got '{method}'")


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
        ``'CSS'`` or ``'ML'``.
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
