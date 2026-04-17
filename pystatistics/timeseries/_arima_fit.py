"""
ARIMA model fitting.

Fits ARIMA(p, d, q) and seasonal ARIMA(p, d, q)(P, D, Q)[m] models via
conditional sum of squares (CSS), exact maximum likelihood (ML), or a
two-stage CSS-ML procedure matching R's ``stats::arima()``.

Optimization uses ``scipy.optimize.minimize(method='L-BFGS-B')`` with
numerical gradients. Starting values come from Yule-Walker estimates
for AR parameters, zeros for MA parameters, and the sample mean of
the differenced series.

For seasonal models, non-seasonal and seasonal AR/MA polynomials are
multiplied out to form effective ARMA coefficients before likelihood
evaluation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from pystatistics.core.exceptions import ConvergenceError, ValidationError
from pystatistics.core.validation import check_array, check_1d, check_finite
from pystatistics.timeseries._differencing import diff
from pystatistics.timeseries._arima_factored import (
    _factored_to_effective,
    optimize_arima_factored as _optimize_arima_factored_impl,
)
from pystatistics.timeseries._arima_likelihood import (
    arima_css_residuals,
    arima_negloglik,
    check_invertible,
    check_stationary,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ARIMAResult:
    """
    Result from fitting an ARIMA model.

    Attributes
    ----------
    order : tuple[int, int, int]
        ``(p, d, q)`` — AR order, differencing order, MA order.
    seasonal_order : tuple[int, int, int, int] or None
        ``(P, D, Q, m)`` — seasonal orders and period, or ``None``.
    ar : NDArray
        AR coefficients (length p). For seasonal models, these are the
        non-seasonal AR coefficients only.
    ma : NDArray
        MA coefficients (length q). For seasonal models, these are the
        non-seasonal MA coefficients only.
    seasonal_ar : NDArray
        Seasonal AR coefficients (length P). Empty if non-seasonal.
    seasonal_ma : NDArray
        Seasonal MA coefficients (length Q). Empty if non-seasonal.
    mean : float or None
        Estimated mean of the differenced series (``None`` if
        ``include_mean=False``).
    sigma2 : float
        Estimated innovation variance.
    vcov : NDArray
        Variance-covariance matrix of the estimated coefficients
        (AR, MA, seasonal AR, seasonal MA, mean). Computed from the
        numerical Hessian of the negative log-likelihood.
    residuals : NDArray
        Innovation residuals (length of the differenced series).
    fitted_values : NDArray
        One-step-ahead fitted values (length of the differenced series).
    log_likelihood : float
        Maximized log-likelihood value.
    aic : float
        Akaike information criterion: ``-2*loglik + 2*k``.
    aicc : float
        Corrected AIC: ``AIC + 2*k*(k+1)/(n-k-1)``.
    bic : float
        Bayesian information criterion: ``-2*loglik + k*log(n)``.
    n_obs : int
        Length of the original (undifferenced) series.
    n_used : int
        Number of observations used in estimation (after differencing).
    method : str
        Estimation method: ``'CSS'``, ``'ML'``, or ``'CSS-ML'``.
    converged : bool
        Whether the optimizer converged.
    n_iter : int
        Number of optimizer iterations.
    """

    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int] | None
    ar: NDArray
    ma: NDArray
    seasonal_ar: NDArray
    seasonal_ma: NDArray
    mean: float | None
    sigma2: float
    vcov: NDArray
    residuals: NDArray
    fitted_values: NDArray
    log_likelihood: float
    aic: float
    aicc: float
    bic: float
    n_obs: int
    n_used: int
    method: str
    converged: bool
    n_iter: int

    @property
    def n_params(self) -> int:
        """Total number of estimated parameters (AR + MA + seasonal + mean + sigma2)."""
        p = len(self.ar)
        q = len(self.ma)
        sp = len(self.seasonal_ar)
        sq = len(self.seasonal_ma)
        k = p + q + sp + sq + (1 if self.mean is not None else 0) + 1
        return k

    def summary(self) -> str:
        """
        R-style summary matching ``stats::arima()`` output.

        Returns
        -------
        str
            Multi-line summary string.
        """
        p, d, q = self.order
        if self.seasonal_order is not None:
            sp, sd, sq_s, m = self.seasonal_order
            header = f"ARIMA({p},{d},{q})({sp},{sd},{sq_s})[{m}]"
        else:
            header = f"ARIMA({p},{d},{q})"

        lines = [header, ""]
        names, coefs = self._collect_coef_names_values()

        if names:
            n_coef = len(coefs)
            se = np.sqrt(np.abs(np.diag(self.vcov[:n_coef, :n_coef])))
            cw = max(10, max(len(n) for n in names) + 2)

            lines.append("Coefficients:")
            lines.append("".join(f"{n:>{cw}}" for n in names))
            lines.append("".join(f"{c:>{cw}.4f}" for c in coefs))
            lines.append("s.e." + "".join(f"{s:>{cw}.4f}" for s in se))
            lines.append("")

        lines.append(
            f"sigma^2 = {self.sigma2:.4f}:  "
            f"log likelihood = {self.log_likelihood:.2f}"
        )
        lines.append(
            f"AIC={self.aic:.2f}   AICc={self.aicc:.2f}   BIC={self.bic:.2f}"
        )
        return "\n".join(lines)

    def _collect_coef_names_values(self) -> tuple[list[str], list[float]]:
        """Build parallel lists of coefficient names and values."""
        names: list[str] = []
        coefs: list[float] = []
        for i, v in enumerate(self.ar):
            names.append(f"ar{i + 1}"); coefs.append(float(v))
        for i, v in enumerate(self.ma):
            names.append(f"ma{i + 1}"); coefs.append(float(v))
        for i, v in enumerate(self.seasonal_ar):
            names.append(f"sar{i + 1}"); coefs.append(float(v))
        for i, v in enumerate(self.seasonal_ma):
            names.append(f"sma{i + 1}"); coefs.append(float(v))
        if self.mean is not None:
            names.append("intercept"); coefs.append(self.mean)
        return names, coefs


# ---------------------------------------------------------------------------
# Seasonal polynomial multiplication
# ---------------------------------------------------------------------------

def _multiply_polynomials(
    nonseasonal: NDArray,
    seasonal: NDArray,
    period: int,
) -> NDArray:
    """
    Multiply non-seasonal and seasonal polynomials.

    For AR polynomials, computes the product:
        phi(B) * Phi(B^m) = (1 - phi_1*B - ... - phi_p*B^p)
                          * (1 - Phi_1*B^m - ... - Phi_P*B^{Pm})

    and returns the coefficients of the result (excluding the leading 1).

    Parameters
    ----------
    nonseasonal : NDArray
        Non-seasonal coefficients [c_1, ..., c_p] where the polynomial
        is ``1 - c_1*B - ... - c_p*B^p``.
    seasonal : NDArray
        Seasonal coefficients [C_1, ..., C_P] where the polynomial
        is ``1 - C_1*B^m - ... - C_P*B^{Pm}``.
    period : int
        Seasonal period *m*.

    Returns
    -------
    NDArray
        Combined coefficients of the product polynomial (excluding
        the leading 1 term), negated so they follow the sign convention
        ``1 - result_1*B - result_2*B^2 - ...``.
    """
    # Build polynomial representations with leading 1
    # For poly1 = 1 - c_1*B - c_2*B^2 ..., we store [1, -c_1, -c_2, ...]
    poly1 = np.zeros(1 + len(nonseasonal))
    poly1[0] = 1.0
    for i, c in enumerate(nonseasonal):
        poly1[i + 1] = -c

    max_seasonal_lag = len(seasonal) * period
    poly2 = np.zeros(1 + max_seasonal_lag)
    poly2[0] = 1.0
    for i, c in enumerate(seasonal):
        poly2[(i + 1) * period] = -c

    # Convolve the two polynomials
    product = np.convolve(poly1, poly2)

    # Return coefficients after leading 1, negated back to positive convention
    return -product[1:]


def _multiply_ma_polynomials(
    nonseasonal: NDArray,
    seasonal: NDArray,
    period: int,
) -> NDArray:
    """Multiply two MA polynomials under pystatistics' MA sign convention.

    pystatistics stores AR and MA with OPPOSITE sign conventions:
        AR:  e_t = y_t - Σ ar_i y_{t-i}       (AR polynomial 1 − Σ ar_i B^i)
        MA:  e_t = y_t - Σ ma_j e_{t-j}       (MA polynomial 1 + Σ ma_j B^j)

    ``_multiply_polynomials`` above handles the AR case (all signs
    negated). For MA we need the product of
        (1 + ma_1 B + … + ma_q B^q)(1 + sma_1 B^m + … + sma_Q B^{Qm})
    returned as ``[ma_eff_1, ma_eff_2, …]``. A straight convolution of
    ``[1, ma_1, …, ma_q]`` with ``[1, sma_1 at position m, …]`` yields
    this directly — no double negation.
    """
    poly1 = np.zeros(1 + len(nonseasonal))
    poly1[0] = 1.0
    for i, c in enumerate(nonseasonal):
        poly1[i + 1] = c

    max_seasonal_lag = len(seasonal) * period
    poly2 = np.zeros(1 + max_seasonal_lag)
    poly2[0] = 1.0
    for i, c in enumerate(seasonal):
        poly2[(i + 1) * period] = c

    product = np.convolve(poly1, poly2)
    return product[1:]


# ---------------------------------------------------------------------------
# Starting values
# ---------------------------------------------------------------------------

def _yule_walker_start(y: NDArray, p: int) -> NDArray:
    """
    Compute Yule-Walker starting values for AR coefficients.

    Uses biased (1/n) autocovariance estimates and solves the
    Yule-Walker equations via ``numpy.linalg.solve``.

    Parameters
    ----------
    y : NDArray
        Demeaned differenced series.
    p : int
        AR order.

    Returns
    -------
    NDArray
        Initial AR coefficient estimates. Returns zeros if the
        Yule-Walker system is singular.
    """
    if p == 0:
        return np.array([], dtype=np.float64)

    n = len(y)
    # Biased autocovariances
    acov = np.empty(p + 1)
    for k in range(p + 1):
        acov[k] = np.dot(y[:n - k], y[k:]) / n

    if acov[0] <= 0.0:
        return np.zeros(p)

    # Build Toeplitz matrix R and solve R * phi = r
    R = np.empty((p, p))
    for i in range(p):
        for j in range(p):
            R[i, j] = acov[abs(i - j)]

    r = acov[1:p + 1]

    try:
        phi = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        phi = np.zeros(p)

    return phi


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_arima_inputs(
    y: ArrayLike,
    order: tuple[int, int, int],
    seasonal: tuple[int, int, int, int] | None,
    include_mean: bool,
    method: str,
) -> NDArray:
    """
    Validate all inputs to :func:`arima` and return the series as a 1-D array.

    Raises :class:`ValidationError` on any invalid input.
    """
    arr = check_array(y, "y")
    arr = arr.ravel()
    check_1d(arr, "y")
    check_finite(arr, "y")

    if len(order) != 3:
        raise ValidationError(f"order: must be a 3-tuple (p, d, q), got length {len(order)}")
    p, d, q = order
    for name, val in [("p", p), ("d", d), ("q", q)]:
        if not isinstance(val, (int, np.integer)) or val < 0:
            raise ValidationError(f"order.{name}: must be a non-negative integer, got {val}")

    if seasonal is not None:
        if len(seasonal) != 4:
            raise ValidationError(f"seasonal: must be a 4-tuple (P, D, Q, m), got length {len(seasonal)}")
        sp, sd, sq, m = seasonal
        for name, val in [("P", sp), ("D", sd), ("Q", sq)]:
            if not isinstance(val, (int, np.integer)) or val < 0:
                raise ValidationError(f"seasonal.{name}: must be a non-negative integer, got {val}")
        if not isinstance(m, (int, np.integer)) or m < 2:
            raise ValidationError(f"seasonal.m: must be an integer >= 2, got {m}")

    valid_methods = ("CSS", "ML", "CSS-ML")
    if method not in valid_methods:
        raise ValidationError(f"method: must be one of {valid_methods}, got '{method}'")

    # Check series is long enough after differencing
    total_diff = d + (seasonal[1] * seasonal[3] if seasonal is not None else 0)
    min_obs = total_diff + max(p, q, 1) + 1
    if seasonal is not None:
        min_obs = max(min_obs, total_diff + max(seasonal[0] * seasonal[3], seasonal[2] * seasonal[3], 1) + 1)
    if len(arr) < min_obs:
        raise ValidationError(
            f"y: series of length {len(arr)} is too short for the specified "
            f"ARIMA model (requires at least {min_obs} observations)"
        )
    return arr


# ---------------------------------------------------------------------------
# Core optimization
# ---------------------------------------------------------------------------

def _compute_hessian(
    params: NDArray,
    y: NDArray,
    order_pq: tuple[int, int],
    include_mean: bool,
    method_ll: str,
    step: float = 1e-4,
) -> NDArray:
    """
    Compute the numerical Hessian of the negative log-likelihood.

    Uses second-order central finite differences.

    Parameters
    ----------
    params : NDArray
        Optimal parameter vector.
    y : NDArray
        Differenced series.
    order_pq : tuple[int, int]
        ``(p_eff, q_eff)`` effective ARMA orders.
    include_mean : bool
        Whether *params* includes a mean term.
    method_ll : str
        ``'CSS'`` or ``'ML'``.
    step : float
        Finite difference step size.

    Returns
    -------
    NDArray
        Hessian matrix (k x k).
    """
    k = len(params)
    hessian = np.zeros((k, k))
    f0 = arima_negloglik(params, y, order_pq, include_mean, method_ll)

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

            fpp = arima_negloglik(params_pp, y, order_pq, include_mean, method_ll)
            fpm = arima_negloglik(params_pm, y, order_pq, include_mean, method_ll)
            fmp = arima_negloglik(params_mp, y, order_pq, include_mean, method_ll)
            fmm = arima_negloglik(params_mm, y, order_pq, include_mean, method_ll)

            hessian[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * step * step)
            hessian[j, i] = hessian[i, j]

    return hessian


def _optimize_arima(
    y_diff: NDArray,
    p_eff: int,
    q_eff: int,
    include_mean: bool,
    method: str,
    tol: float,
    max_iter: int,
    start_params: NDArray,
) -> tuple[NDArray, float, bool, int, str]:
    """
    Run L-BFGS-B optimization for ARMA parameters.

    Parameters
    ----------
    y_diff : NDArray
        Differenced series.
    p_eff : int
        Effective AR order (after polynomial multiplication).
    q_eff : int
        Effective MA order (after polynomial multiplication).
    include_mean : bool
        Whether to include a mean parameter.
    method : str
        ``'CSS'``, ``'ML'``, or ``'CSS-ML'``.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.
    start_params : NDArray
        Initial parameter vector.

    Returns
    -------
    tuple
        ``(optimal_params, neg_loglik, converged, n_iter, method_used)``
    """
    order_pq = (p_eff, q_eff)

    # Closed-form case: no AR and no MA parameters. The negative
    # log-likelihood depends only on the mean (if included) and the
    # residual variance. The MLE of the mean is the sample mean of the
    # differenced series; there is no optimization to run. Handing this
    # to scipy.minimize with a near-MLE start causes L-BFGS-B to exit
    # with nit=0 "ABNORMAL" because the initial gradient is essentially
    # zero — which is indistinguishable from a real convergence failure.
    # Rule 1: handle this explicitly; do not paper over scipy's behavior
    # with a warning-and-fallback path.
    if p_eff == 0 and q_eff == 0:
        if include_mean:
            # MLE: mu_hat = mean(y_diff)
            opt_params = np.array([float(np.mean(y_diff))])
        else:
            opt_params = np.array([])
        nll_css = float(arima_negloglik(
            opt_params, y_diff, order_pq, include_mean, "CSS",
        ))
        if method == "CSS":
            return opt_params, nll_css, True, 0, "CSS"
        nll_ml = float(arima_negloglik(
            opt_params, y_diff, order_pq, include_mean, "ML",
        ))
        if not np.isfinite(nll_ml):
            raise ConvergenceError(
                "ARIMA ML likelihood is not finite for the closed-form "
                "case (p=q=0); the differenced series is degenerate "
                "(e.g. all identical values). Use method='CSS' or "
                "check the input series.",
                iterations=0,
                reason="degenerate_likelihood",
            )
        method_used = "ML" if method == "ML" else "CSS-ML"
        return opt_params, nll_ml, True, 0, method_used

    if method == "CSS" or method == "CSS-ML":
        result_css = minimize(
            arima_negloglik,
            start_params,
            args=(y_diff, order_pq, include_mean, "CSS"),
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": tol},
        )
        opt_params = result_css.x
        nll = result_css.fun
        converged = result_css.success
        n_iter = result_css.nit
        method_used = "CSS"

        if method == "CSS-ML":
            # Refine with exact ML starting from CSS solution.
            # Fail loud if ML refinement fails — user asked for CSS-ML and
            # silently returning CSS estimates would mask the failure.
            # Rule 1: raise on unexpected failure; do not return a default
            # to mask missing/invalid state.
            css_params = opt_params.copy()
            try:
                result_ml = minimize(
                    arima_negloglik,
                    css_params,
                    args=(y_diff, order_pq, include_mean, "ML"),
                    method="L-BFGS-B",
                    options={"maxiter": max_iter, "ftol": tol},
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise ConvergenceError(
                    "ARIMA CSS-ML: ML refinement failed numerically "
                    f"({type(exc).__name__}: {exc}). "
                    "Use method='CSS' if you want CSS estimates, or "
                    "adjust tol / max_iter / starting values.",
                    iterations=n_iter,
                    reason="ml_numerical_failure",
                ) from exc

            n_iter += result_ml.nit
            if not (result_ml.success and np.isfinite(result_ml.fun)):
                raise ConvergenceError(
                    "ARIMA CSS-ML: ML refinement did not converge after "
                    f"{result_ml.nit} iterations (scipy message: "
                    f"{result_ml.message}). "
                    "Use method='CSS' if you want CSS estimates, or "
                    "increase max_iter / relax tol.",
                    iterations=n_iter,
                    reason="ml_max_iter_or_infeasible",
                    threshold=tol,
                )

            opt_params = result_ml.x
            nll = result_ml.fun
            method_used = "CSS-ML"

    else:
        # Pure ML
        result_ml = minimize(
            arima_negloglik,
            start_params,
            args=(y_diff, order_pq, include_mean, "ML"),
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": tol},
        )
        opt_params = result_ml.x
        nll = result_ml.fun
        converged = result_ml.success
        n_iter = result_ml.nit
        method_used = "ML"

    return opt_params, nll, converged, n_iter, method_used


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def arima(
    y: ArrayLike,
    *,
    order: tuple[int, int, int] = (0, 0, 0),
    seasonal: tuple[int, int, int, int] | None = None,
    include_mean: bool = True,
    method: str = "CSS-ML",
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> ARIMAResult:
    """
    Fit an ARIMA(p, d, q) or seasonal ARIMA(p, d, q)(P, D, Q)[m] model.

    Matches the interface and numerical approach of R's ``stats::arima()``.

    For non-seasonal ARIMA(p, d, q):
        1. Difference the series *d* times.
        2. Fit ARMA(p, q) to the differenced series.

    For seasonal ARIMA(p, d, q)(P, D, Q)[m]:
        1. Seasonally difference *D* times (lag *m*).
        2. Difference *d* times.
        3. Multiply out the seasonal and non-seasonal AR/MA polynomials
           to form effective ARMA coefficients.

    Methods:
        - ``'CSS'``: Conditional sum of squares (fast, approximate).
        - ``'ML'``: Exact maximum likelihood via innovations algorithm.
        - ``'CSS-ML'``: CSS for initialization, then ML refinement (default).

    Starting values:
        - AR: Yule-Walker estimates from autocorrelations.
        - MA: zeros.
        - Mean: sample mean of the differenced series.

    Parameters
    ----------
    y : ArrayLike
        Time series (1-D array).
    order : tuple[int, int, int]
        ``(p, d, q)`` — AR order, differencing order, MA order.
    seasonal : tuple[int, int, int, int] or None
        ``(P, D, Q, m)`` — seasonal AR order, seasonal differencing order,
        seasonal MA order, and period. ``None`` for non-seasonal models.
    include_mean : bool
        Whether to include a mean term. Default ``True``.
    method : str
        ``'CSS'``, ``'ML'``, or ``'CSS-ML'``. Default ``'CSS-ML'``.
    tol : float
        Convergence tolerance for the optimizer. Default ``1e-8``.
    max_iter : int
        Maximum optimizer iterations. Default ``1000``.

    Returns
    -------
    ARIMAResult
        Fitted model with coefficients, residuals, and diagnostics.

    Raises
    ------
    ValidationError
        If inputs are invalid.
    ConvergenceError
        If the optimizer fails to converge.
    """
    arr = _validate_arima_inputs(y, order, seasonal, include_mean, method)
    n_obs = len(arr)
    p, d, q = order

    # ----- Apply differencing -----
    y_diff = arr.copy()

    # Seasonal differencing first
    if seasonal is not None:
        sp, sd, sq, m = seasonal
        for _ in range(sd):
            y_diff = diff(y_diff, differences=1, lag=m)

    # Regular differencing
    if d > 0:
        y_diff = diff(y_diff, differences=d, lag=1)

    n_used = len(y_diff)

    # ----- Build effective ARMA orders -----
    sp, sd, sq, m = seasonal if seasonal is not None else (0, 0, 0, 1)
    p_eff = p + sp * m
    q_eff = q + sq * m

    # ----- Starting values -----
    #
    # For SEASONAL models we optimize in the factored parameterization:
    #   params = [ar_1..ar_p, ma_1..ma_q, sar_1..sar_P, sma_1..sma_Q, mean?]
    # so scipy sees p + q + sp + sq (+1) dimensions instead of the
    # expanded p_eff + q_eff (+1). For the Box-Jenkins airline model this
    # is 2 dims vs 13 — roughly the factor difference between R's fit
    # time and ours. When there is no seasonal structure, p_eff == p
    # and q_eff == q, so the two parameterizations coincide.
    seasonal_fit = seasonal is not None and (sp > 0 or sq > 0)
    y_demean = y_diff - np.mean(y_diff) if include_mean else y_diff.copy()
    mean_start = np.mean(y_diff) if include_mean else None

    if seasonal_fit:
        # Factored starts. Zero starts are safe but for airline-type
        # models the likelihood has multiple nearby stationary points;
        # plain zero can land in an inferior local minimum 4+ log-lik
        # units above the MLE. We therefore:
        #   - initialize AR via Yule-Walker on the differenced series
        #     (same as non-seasonal path);
        #   - initialize MA/seasonal-MA to a small negative value (-0.1)
        #     — the same heuristic R's `stats::arima` uses because
        #     empirical data almost always has slight positive serial
        #     correlation at short lags, which corresponds to negative
        #     MA/θ coefficients.
        # These starts were verified to recover R's airline-model
        # optimum on log(AirPassengers) (ma1 ≈ -0.40, sma1 ≈ -0.56).
        if p > 0:
            ar_start = np.clip(_yule_walker_start(y_demean, p), -0.99, 0.99)
        else:
            ar_start = np.zeros(0)
        ma_start = np.full(q, -0.1)
        sar_start = np.zeros(sp)
        sma_start = np.full(sq, -0.1)
        parts = [ar_start, ma_start, sar_start, sma_start]
        if include_mean:
            parts.append(np.array([mean_start]))
        start_params = np.concatenate(parts)
    else:
        ar_start = np.clip(_yule_walker_start(y_demean, p_eff), -0.99, 0.99)
        ma_start = np.zeros(q_eff)
        start_params = np.concatenate([ar_start, ma_start])
        if include_mean:
            start_params = np.concatenate([start_params, [mean_start]])

    # ----- Handle trivial case: no parameters to estimate -----
    if p_eff == 0 and q_eff == 0 and not include_mean:
        residuals = y_diff.copy()
        sigma2 = np.dot(residuals, residuals) / n_used
        nll = (
            0.5 * n_used * np.log(2.0 * np.pi)
            + 0.5 * n_used * np.log(max(sigma2, 1e-15))
            + 0.5 * n_used
        )
        return ARIMAResult(
            order=order,
            seasonal_order=seasonal,
            ar=np.array([], dtype=np.float64),
            ma=np.array([], dtype=np.float64),
            seasonal_ar=np.array([], dtype=np.float64),
            seasonal_ma=np.array([], dtype=np.float64),
            mean=None,
            sigma2=sigma2,
            vcov=np.array([], dtype=np.float64).reshape(0, 0),
            residuals=residuals,
            fitted_values=y_diff - residuals,
            log_likelihood=-nll,
            aic=2.0 * nll + 2.0,
            aicc=2.0 * nll + 2.0 + (2.0 / max(n_used - 2, 1)),
            bic=2.0 * nll + np.log(n_used),
            n_obs=n_obs,
            n_used=n_used,
            method=method,
            converged=True,
            n_iter=0,
        )

    # ----- Optimize -----
    if seasonal_fit:
        opt_factored, nll, converged, n_iter, method_used = (
            _optimize_arima_factored_impl(
                y_diff, p, q, sp, sq, m, include_mean,
                method, tol, max_iter, start_params,
                _multiply_polynomials, _multiply_ma_polynomials,
            )
        )
        # Expand back to effective coefficients (for residuals, Hessian,
        # and the existing reporting path). Keep the factored version
        # around so we can report it separately below.
        opt_params = _factored_to_effective(
            opt_factored, p, q, sp, sq, m, include_mean,
            _multiply_polynomials, _multiply_ma_polynomials,
        )
    else:
        opt_params, nll, converged, n_iter, method_used = _optimize_arima(
            y_diff, p_eff, q_eff, include_mean, method, tol, max_iter, start_params,
        )
        opt_factored = None

    if not converged:
        raise ConvergenceError(
            f"ARIMA optimization did not converge after {n_iter} iterations",
            iterations=n_iter,
            reason="optimizer_failed",
        )

    # ----- Extract results -----
    ar_eff = opt_params[:p_eff]
    ma_eff = opt_params[p_eff:p_eff + q_eff]
    mean_est = opt_params[p_eff + q_eff] if include_mean else None

    mean_val = mean_est if include_mean else 0.0

    # Compute final residuals and fitted values
    residuals = arima_css_residuals(y_diff, ar_eff, ma_eff, mean_val)
    fitted = y_diff - residuals
    sigma2 = np.dot(residuals, residuals) / n_used

    # ----- Decompose effective coefficients into seasonal and non-seasonal -----
    if seasonal_fit:
        # Factored optimization: the factored params are already the
        # non-seasonal and seasonal pieces, no decomposition needed.
        ar_nonseasonal = opt_factored[:p]
        ma_nonseasonal = opt_factored[p:p + q]
        sar = opt_factored[p + q:p + q + sp]
        sma = opt_factored[p + q + sp:p + q + sp + sq]
    else:
        ar_nonseasonal = ar_eff[:p]
        ma_nonseasonal = ma_eff[:q]
        sar = np.array([], dtype=np.float64)
        sma = np.array([], dtype=np.float64)

    # ----- Variance-covariance matrix from Hessian -----
    # Use CSS Hessian even for CSS-ML to avoid expensive O(n^2) innovations
    # evaluations per finite-difference step. The CSS Hessian provides a
    # reasonable approximation to the exact information matrix.
    method_for_hessian = "ML" if method_used == "ML" else "CSS"
    n_coef = len(opt_params)

    try:
        hessian = _compute_hessian(
            opt_params, y_diff, (p_eff, q_eff), include_mean, method_for_hessian,
        )
        vcov = np.linalg.inv(hessian)
        # Ensure symmetry
        vcov = 0.5 * (vcov + vcov.T)
    except np.linalg.LinAlgError:
        vcov = np.full((n_coef, n_coef), np.nan)
        warnings.warn(
            "Hessian is singular; variance-covariance matrix could not be computed",
            stacklevel=2,
        )

    # ----- Stationarity and invertibility checks -----
    if not check_stationary(ar_eff):
        warnings.warn(
            "AR polynomial has roots inside or on the unit circle; "
            "the model may be non-stationary",
            stacklevel=2,
        )
    if not check_invertible(ma_eff):
        warnings.warn(
            "MA polynomial has roots inside or on the unit circle; "
            "the model may be non-invertible",
            stacklevel=2,
        )

    # ----- Information criteria -----
    # k = number of estimated parameters (coefficients + sigma2)
    k = n_coef + 1  # +1 for sigma2
    loglik = -nll
    aic = -2.0 * loglik + 2.0 * k
    bic = -2.0 * loglik + k * np.log(n_used)
    # AICc: corrected AIC
    if n_used - k - 1 > 0:
        aicc = aic + 2.0 * k * (k + 1.0) / (n_used - k - 1.0)
    else:
        aicc = np.inf

    return ARIMAResult(
        order=order,
        seasonal_order=seasonal,
        ar=ar_nonseasonal,
        ma=ma_nonseasonal,
        seasonal_ar=sar,
        seasonal_ma=sma,
        mean=mean_est,
        sigma2=sigma2,
        vcov=vcov,
        residuals=residuals,
        fitted_values=fitted,
        log_likelihood=loglik,
        aic=aic,
        aicc=aicc,
        bic=bic,
        n_obs=n_obs,
        n_used=n_used,
        method=method_used,
        converged=converged,
        n_iter=n_iter,
    )
