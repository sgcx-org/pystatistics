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

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pystatistics.core.exceptions import ConvergenceError, ValidationError
from pystatistics.core.result import Result
from pystatistics.core.validation import check_array, check_1d, check_finite
from pystatistics.timeseries._arima_solution import ARIMAParams, ARIMASolution
from pystatistics.timeseries._differencing import diff
from pystatistics.timeseries._arima_init import prepare_init
from pystatistics.timeseries._arima_factored import (
    _factored_to_effective,
    _multiply_ma_polynomials,
    _multiply_polynomials,
    normalize_to_invertible,
    optimize_arima_factored as _optimize_arima_factored_impl,
)
from pystatistics.timeseries._arima_likelihood import (
    arima_css_residuals,
    arima_negloglik,
    check_invertible,
    check_stationary,
    compute_numerical_hessian as _compute_hessian,
    minimize_quiet,
)


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

    valid_methods = ("css", "ml", "css-ml", "whittle")
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


# Public ``backend=`` tokens that request GPU execution. Only ``method='whittle'``
# has a GPU implementation (``backends/whittle_gpu.py``); the exact-ML / CSS-ML /
# CSS path is CPU-only. An explicit GPU request on any other method must fail loud
# rather than compute on the CPU and report ``backend_name='cpu'`` — silently
# honoring a different backend than asked for violates the Fidelity guarantee
# (CONVENTIONS.md A6/A7). ``'cpu'``/``None`` pass through; ``'auto'`` is a disclosed
# default-selection (it resolves to CPU here, reported via ``backend_name``), not an
# explicit request, so it is not rejected.
_GPU_BACKEND_TOKENS = frozenset({"gpu", "gpu_fp64", "mps", "cuda"})


def _reject_gpu_backend_without_whittle(backend: str | None, method: str) -> None:
    """Fail loud on an explicit GPU ``backend=`` for a method with no GPU path.

    The exact-ML/CSS ARIMA path has no GPU kernel; only ``method='whittle'`` does.
    Honoring ``backend='gpu'`` by silently running on the CPU would be an A6/A7
    Fidelity violation, so raise with the real remedies instead.
    """
    if backend in _GPU_BACKEND_TOKENS and method != "whittle":
        raise ValidationError(
            f"arima()'s exact-ML/CSS path has no GPU backend "
            f"(backend={backend!r}, method={method!r}). Use method='whittle' with "
            f"backend='gpu' for the frequency-domain GPU path, arima_batch(...) for "
            f"batched GPU fits, or backend='cpu' (the default reference path)."
        )


# ---------------------------------------------------------------------------
# Core optimization
# ---------------------------------------------------------------------------

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
        ``'css'``, ``'ml'``, or ``'css-ml'``.
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
            opt_params, y_diff, order_pq, include_mean, "css",
        ))
        if method == "css":
            return opt_params, nll_css, True, 0, "css"
        nll_ml = float(arima_negloglik(
            opt_params, y_diff, order_pq, include_mean, "ml",
        ))
        if not np.isfinite(nll_ml):
            raise ConvergenceError(
                "ARIMA ML likelihood is not finite for the closed-form "
                "case (p=q=0); the differenced series is degenerate "
                "(e.g. all identical values). Use method='css' or "
                "check the input series.",
                iterations=0,
                reason="degenerate_likelihood",
            )
        method_used = "ml" if method == "ml" else "css-ml"
        return opt_params, nll_ml, True, 0, method_used

    if method == "css" or method == "css-ml":
        result_css = minimize_quiet(
            arima_negloglik,
            start_params,
            args=(y_diff, order_pq, include_mean, "css"),
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": tol},
        )
        opt_params = result_css.x
        nll = result_css.fun
        converged = result_css.success
        n_iter = result_css.nit
        method_used = "css"

        if method == "css-ml":
            # Refine with exact ML starting from CSS solution.
            # Fail loud if ML refinement fails — user asked for CSS-ML and
            # silently returning CSS estimates would mask the failure.
            # Rule 1: raise on unexpected failure; do not return a default
            # to mask missing/invalid state.
            css_params = opt_params.copy()
            try:
                result_ml = minimize_quiet(
                    arima_negloglik,
                    css_params,
                    args=(y_diff, order_pq, include_mean, "ml"),
                    method="L-BFGS-B",
                    options={"maxiter": max_iter, "ftol": tol},
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise ConvergenceError(
                    "ARIMA CSS-ML: ML refinement failed numerically "
                    f"({type(exc).__name__}: {exc}). "
                    "Use method='css' if you want CSS estimates, or "
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
                    "Use method='css' if you want CSS estimates, or "
                    "increase max_iter / relax tol.",
                    iterations=n_iter,
                    reason="ml_max_iter_or_infeasible",
                    threshold=tol,
                )

            opt_params = result_ml.x
            nll = result_ml.fun
            method_used = "css-ml"
            # The ML refinement succeeded (the raise above guards it):
            # a failed CSS warm-start (e.g. a NaN objective step) must
            # not shadow a converged ML optimum. Previously `converged`
            # kept the CSS stage's flag and arima() raised a spurious
            # ConvergenceError on fits that matched R's optimum (co2
            # (2,1,1): ML at -466.830 vs R -466.830, CSS stage
            # ABNORMAL).
            converged = True

            # Second ML start from the original (Yule-Walker + sample
            # mean) point, kept only if strictly better. With a mean in
            # the parameter vector the (ar, mean) surface has a flat
            # canyon toward the AR unit root (a near-unit AR barely
            # identifies the level); the CSS stage can hand the ML
            # stage a basin with a drifted mean that L-BFGS-B cannot
            # leave (AirPassengers (1,0,1): mean 115.7 / loglik 1.84
            # below R vs the sample-mean start's 280.3 / R-matching).
            # Same better-of-two-starts pattern as the damped-ETS fix:
            # the fit can only improve or stay identical (RIGOR R18
            # follow-up).
            if include_mean:
                try:
                    result_ml2 = minimize_quiet(
                        arima_negloglik,
                        start_params,
                        args=(y_diff, order_pq, include_mean, "ml"),
                        method="L-BFGS-B",
                        options={"maxiter": max_iter, "ftol": tol},
                    )
                except (ValueError, np.linalg.LinAlgError):
                    result_ml2 = None
                if result_ml2 is not None:
                    n_iter += result_ml2.nit
                    if (result_ml2.success and np.isfinite(result_ml2.fun)
                            and result_ml2.fun < nll):
                        opt_params = result_ml2.x
                        nll = result_ml2.fun

    else:
        # Pure ML
        result_ml = minimize_quiet(
            arima_negloglik,
            start_params,
            args=(y_diff, order_pq, include_mean, "ml"),
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": tol},
        )
        opt_params = result_ml.x
        nll = result_ml.fun
        converged = result_ml.success
        n_iter = result_ml.nit
        method_used = "ml"

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
    xreg: ArrayLike | None = None,
    include_drift: bool = False,
    fixed: dict | ArrayLike | None = None,
    method: str = "css-ml",
    init: ArrayLike | None = None,
    tol: float = 1e-8,
    max_iter: int = 1000,
    backend: str | None = None,
) -> ARIMASolution:
    """
    Fit an ARIMA(p, d, q) or seasonal ARIMA(p, d, q)(P, D, Q)[m] model.

    Matches the numerical behaviour of R's ``stats::arima()`` — exact
    maximum likelihood via the same Kalman-filter approach, with
    results (log-likelihood, coefficients, information criteria,
    forecasts, standard errors) verified against R — and supports a
    documented subset of its interface (see *R interface coverage*
    below).

    For non-seasonal ARIMA(p, d, q):
        1. Difference the series *d* times.
        2. Fit ARMA(p, q) to the differenced series.

    For seasonal ARIMA(p, d, q)(P, D, Q)[m]:
        1. Seasonally difference *D* times (lag *m*).
        2. Difference *d* times.
        3. Multiply out the seasonal and non-seasonal AR/MA polynomials
           to form effective ARMA coefficients.

    Methods:
        - ``'css'``: Conditional sum of squares (fast, approximate).
        - ``'ml'``: Exact maximum likelihood via the Kalman filter.
        - ``'css-ml'``: CSS for initialization, then ML refinement (default).

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
        Whether to include a mean term. Default ``True``. Ignored (no
        mean is estimated) when the model has any differencing
        (``d + D > 0``), matching R ``stats::arima``'s ``include.mean``.
        With ``xreg`` / ``include_drift`` the mean is carried as the
        ``'intercept'`` regression coefficient.
    xreg : ArrayLike or None
        External regressors for regression with ARIMA errors: fit
        ``y = X @ beta + eta`` where ``eta`` follows the ARIMA process
        (R's ``stats::arima(xreg=)`` / ``forecast::Arima(xreg=)``). Shape
        ``(n,)`` or ``(n, k)`` with one row per observation of *y*. The
        regression coefficients appear on the solution as ``xreg_coef``
        (named ``xreg1..xregk``), with joint standard errors in ``vcov``.
        Forecasting a model with ``xreg`` requires future regressor values
        (``forecast_arima(..., new_xreg=)``). Default ``None``.
    include_drift : bool
        Include a linear time-trend (drift) regressor — the models R
        reports "with drift" (``forecast::Arima(include.drift=TRUE)``).
        Reported as the ``'drift'`` regression coefficient. Requires total
        differencing ``d + D <= 1`` (the trend vanishes under higher-order
        differencing). Default ``False``.
    fixed : dict or ArrayLike or None
        Hold coefficients fixed during estimation (R's
        ``stats::arima(fixed=)`` parameter masking). Primary form is a
        ``{name: value}`` mapping over the coefficient names
        (``ar1``, ``ma1``, ..., ``intercept``, ``drift``, ``xreg1``, ...)
        — e.g. ``fixed={'ma1': 0}`` holds ma1 at 0 and estimates the
        rest. A positional array aligned to the coefficient order with
        ``nan`` for free parameters (R's convention) is also accepted.
        Fixed coefficients carry zero variance in ``vcov`` and do not
        count toward the information criteria. Default ``None``.
    method : str
        ``'css'``, ``'ml'``, or ``'css-ml'``. Default ``'css-ml'``.
    init : ArrayLike or None
        Initial parameter values for the optimizer, in R ``coef()``
        order: ``[ar_1..ar_p, ma_1..ma_q, sar_1..sar_P, sma_1..sma_Q,
        mean?]`` (the mean slot exists only when a mean is estimated,
        i.e. ``include_mean=True`` and ``d + D == 0``). ``numpy.nan``
        entries use the defaults (zero for coefficients, the sample
        mean of the differenced series for the mean). AR parts must be
        stationary (as in R); non-invertible MA parts are normalized
        to the invertible representative before optimization (R's
        documented ``maInvert`` intent — R's own implementation errors
        on such inits). Not supported with ``method='whittle'``.
        Default ``None`` (internal starting values).
    tol : float
        Convergence tolerance for the optimizer. Default ``1e-8``.
    max_iter : int
        Maximum optimizer iterations. Default ``1000``.

    Returns
    -------
    ARIMASolution
        Fitted model with coefficients, residuals, and diagnostics.

    Raises
    ------
    ValidationError
        If inputs are invalid.
    ConvergenceError
        If the optimizer fails to converge.

    Notes
    -----
    **R interface coverage.** Supported R parameters: ``order``,
    ``seasonal``, ``include.mean`` (``include_mean``), ``xreg``
    (regression with ARIMA errors), ``include.drift`` (``include_drift``),
    ``fixed`` (parameter masking), ``method`` ('css-ml'/'ml'/'css'), and
    ``init``. Not exposed, by design: ``transform.pars``, ``SSinit``,
    ``kappa``,
    ``n.cond``, and ``optim.method``/``optim.control`` are knobs over
    R's optimizer and state-space internals; pystatistics guarantees
    parity of RESULTS, not of internal knobs — AR stationarity and MA
    invertibility are handled internally (the latter with R's own
    ``maInvert`` convention), the stationary state initialization is
    solved exactly, and the ML stage uses a better-of-two-starts
    strategy verified to reach equal-or-better optima than R on every
    reference model.

    **CSS convention.** ``method='css'`` uses a zero-initialized
    conditional recursion over ALL observations, whereas R conditions
    on (and excludes) the first ``n.cond``. Pure-CSS estimates can
    therefore differ slightly from R's (coefficients typically ~1e-3,
    sigma2 ~1% on reference fits; weakly identified fits may reach
    different local optima) and are NOT covered by the parity
    guarantee. ``'css-ml'`` and ``'ml'`` results are — CSS supplies
    starting values only.
    """
    arr = _validate_arima_inputs(y, order, seasonal, include_mean, method)
    _reject_gpu_backend_without_whittle(backend, method)
    n_obs = len(arr)
    p, d, q = order

    if xreg is not None:
        from pystatistics.timeseries._arima_xreg import validate_xreg
        xreg_arr = validate_xreg(xreg, n_obs)
    else:
        xreg_arr = None
    use_xreg = xreg_arr is not None or include_drift or fixed is not None
    if use_xreg and method == "whittle":
        raise ValidationError(
            "xreg / include_drift / fixed are not supported with "
            "method='whittle' (the frequency-domain path has no regression "
            "term). Use method='css-ml' (default), 'ml', or 'css'."
        )
    if use_xreg and init is not None:
        raise ValidationError(
            "init: not supported together with xreg / include_drift / "
            "fixed. Supply starting values via fixed= or omit init."
        )

    # R parity (stats::arima): ``include.mean`` "is ignored for ARIMA
    # models with differencing" — when d + D > 0 no mean/intercept is
    # estimated. Estimating one anyway would act as an implicit drift
    # term, changing both the fit and the free-parameter count used by
    # the information criteria (RIGOR R18).
    d_seasonal = seasonal[1] if seasonal is not None else 0
    if d + d_seasonal > 0:
        include_mean = False

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

    # ----- Regression with ARIMA errors (xreg / drift / fixed) -----
    # Dispatched to the dedicated regression fitter, which subtracts the
    # differenced regression fit from y_diff and evaluates the SAME ARMA
    # likelihood used below. The plain-ARIMA path (no regressor, no drift,
    # no mask) is left byte-for-byte unchanged.
    if use_xreg:
        from pystatistics.timeseries._arima_xreg import fit_arima_xreg
        return fit_arima_xreg(
            arr=arr, y_diff=y_diff, order=order, seasonal=seasonal,
            xreg=xreg_arr, include_mean=include_mean,
            include_drift=include_drift, fixed=fixed, method=method,
            tol=tol, max_iter=max_iter, n_obs=n_obs, n_used=n_used,
            yule_walker_start=_yule_walker_start,
        )

    # ----- Whittle (frequency-domain approximate MLE) fast path -----
    # Non-seasonal ARMA only. Operates on the already-differenced series
    # (stationarity pre-condition). Exact ML via Kalman remains the
    # default; Whittle is for long series (n ≳ 10⁴) where FFT wins over
    # O(n) Kalman recursion. GPU version lives in
    # ``backends/whittle_gpu.py`` and is selected via ``backend=``.
    if method == "whittle":
        if seasonal is not None and (seasonal[0] > 0 or seasonal[2] > 0
                                     or seasonal[1] > 0):
            raise ValidationError(
                "method='whittle' supports non-seasonal ARMA(p, d, q) "
                "only. For seasonal models use method='css-ml' (default)."
            )
        if init is not None:
            raise ValidationError(
                "init: not supported with method='whittle' (the "
                "frequency-domain path has its own starting values)"
            )
        from pystatistics.timeseries._whittle import fit_arima_whittle
        return fit_arima_whittle(
            y_diff=y_diff, n_obs=n_obs, order=order,
            include_mean=include_mean, tol=tol, max_iter=max_iter,
            backend=backend,
            yule_walker_start=_yule_walker_start,
            css_residuals=arima_css_residuals,
            ARIMASolution=ARIMASolution,
        )

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

    if init is not None:
        # User-supplied start (R's init= semantics; see _arima_init).
        # The factored layout equals the effective layout when there is
        # no seasonal AR/MA part, so this serves both optimizer paths.
        # Note the length check uses the EFFECTIVE include_mean: for
        # d + D > 0 no mean is estimated, so init carries no mean slot.
        start_params = prepare_init(
            init, p, q, sp, sq, include_mean,
            mean_start if mean_start is not None else 0.0,
        )
    elif seasonal_fit:
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
        # Only free parameter is sigma2 (k = 1), so
        # AICc = AIC + 2k(k+1)/(n-k-1) = AIC + 4/(n-2).
        aic_triv = 2.0 * nll + 2.0
        aicc_triv = (
            aic_triv + 4.0 / (n_used - 2) if n_used > 2 else np.inf
        )
        return ARIMASolution(
            _result=Result(
                params=ARIMAParams(
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
                    aic=aic_triv,
                    aicc=aicc_triv,
                    bic=2.0 * nll + np.log(n_used),
                    n_obs=n_obs,
                    n_used=n_used,
                    method=method,
                    converged=True,
                    n_iter=0,
                ),
                info={"method": method, "converged": True},
                timing=None,
                backend_name="cpu",
                warnings=(),
            )
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

    # ----- Canonical (invertible) MA representation -----
    # Exact-ML fits are reported in the invertible representative (the
    # identified/fundamental parameterization; ~10% of boundary-ish
    # fits landed on the non-invertible mirror with sigma2 up to 53%
    # below R). Not applied to pure-CSS fits — the CSS criterion is not
    # reflection-invariant. See normalize_to_invertible for the full
    # rationale and the likelihood-invariance guard.
    if method_used in ("ml", "css-ml"):
        opt_params, opt_factored, nll = normalize_to_invertible(
            opt_params, opt_factored, nll, y_diff,
            p, q, sp, sq, m, include_mean,
        )

    # ----- Extract results -----
    ar_eff = opt_params[:p_eff]
    ma_eff = opt_params[p_eff:p_eff + q_eff]
    mean_est = opt_params[p_eff + q_eff] if include_mean else None

    mean_val = mean_est if include_mean else 0.0

    # Residuals and innovation variance. ML-family fits report the
    # standardized Kalman innovations v_t/sqrt(F_t) — homoscedastic
    # white noise with variance sigma2 under the model (the object
    # Ljung-Box/ACF diagnostics assume) and R's residuals() convention
    # (arima.c scales by sqrt(gain)). The profile ML variance is then
    # EXACTLY mean(residuals^2) ((1/n) sum v_t^2/F_t — R's sigma2; the
    # CSS SSE/n overstated it 2.9% at an MA-boundary fit and CSS
    # residuals carry a conditioning transient decaying like the
    # largest MA root modulus^t, still ~7% at t=131 for ma1=-0.98).
    # kalman_arma_innovations raises loudly if the filter fails at the
    # fitted parameters (Rule 1). Pure-CSS fits keep the CSS recursion
    # and SSE/n — R's CSS convention.
    if method_used in ("ml", "css-ml"):
        from pystatistics.timeseries._arima_kalman import (
            kalman_arma_innovations,
        )
        residuals = kalman_arma_innovations(
            y_diff - mean_val, ar_eff, ma_eff,
        )
        sigma2 = float(np.mean(residuals * residuals))
    else:
        residuals = arima_css_residuals(y_diff, ar_eff, ma_eff, mean_val)
        sigma2 = np.dot(residuals, residuals) / n_used
    fitted = y_diff - residuals

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
    if seasonal_fit:
        # Differentiate in the FACTORED parameterization so vcov
        # rows/columns align with the reported coefficients
        # (ar, ma, sar, sma, mean). It was previously computed over the
        # expanded polynomial, so summary() read seasonal standard
        # errors from structurally-zero expanded lags — the airline
        # model printed sma1 s.e. 0.38 where R gives 0.08 (RIGOR R18
        # follow-up). The final CSS-ML stage is ML and the factored
        # dimension is small (p+q+P+Q), so the exact-ML Hessian via the
        # O(n r^2) Kalman filter is affordable — and it is what R's
        # optim Hessian approximates.
        method_for_hessian = "css" if method_used == "css" else "ml"

        def _nll_for_hessian(theta: NDArray) -> float:
            eff = _factored_to_effective(
                theta, p, q, sp, sq, m, include_mean,
                _multiply_polynomials, _multiply_ma_polynomials,
            )
            return arima_negloglik(
                eff, y_diff, (p_eff, q_eff), include_mean,
                method_for_hessian,
            )

        hess_point = opt_factored
    else:
        # Non-seasonal path unchanged: CSS Hessian even for CSS-ML (a
        # reasonable approximation to the information matrix; validated
        # against R on the non-seasonal reference fits).
        method_for_hessian = "ml" if method_used == "ml" else "css"

        def _nll_for_hessian(theta: NDArray) -> float:
            return arima_negloglik(
                theta, y_diff, (p_eff, q_eff), include_mean,
                method_for_hessian,
            )

        hess_point = opt_params

    n_coef = len(hess_point)

    try:
        hessian = _compute_hessian(_nll_for_hessian, hess_point)
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
    # k = number of FREE estimated parameters, matching R stats::arima:
    # p + q + P + Q coefficients, + 1 if a mean is estimated, + 1 for
    # sigma2 (R's AIC uses length(coef) + 1). NOT len(opt_params): for
    # seasonal models opt_params holds the multiplied-out effective
    # polynomials (order p + P*m etc.), which silently inflated the IC —
    # the airline model (0,1,1)(0,1,1)[12] was counted as k=15 instead
    # of k=3, making auto_arima rank and select the wrong models
    # (RIGOR R18). Must equal ARIMASolution.n_params.
    k = p + q + sp + sq + (1 if include_mean else 0) + 1
    loglik = -nll
    aic = -2.0 * loglik + 2.0 * k
    bic = -2.0 * loglik + k * np.log(n_used)
    # AICc: corrected AIC
    if n_used - k - 1 > 0:
        aicc = aic + 2.0 * k * (k + 1.0) / (n_used - k - 1.0)
    else:
        aicc = np.inf

    return ARIMASolution(
        _result=Result(
            params=ARIMAParams(
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
            ),
            info={"method": method_used, "converged": converged},
            timing=None,
            backend_name="cpu",
            warnings=(),
        )
    )
