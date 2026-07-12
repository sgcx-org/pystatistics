"""
Regression with ARIMA errors: ``xreg``, drift, and ``fixed=`` masking.

Fits ``y = X @ beta + eta`` where the error series ``eta`` follows the
ARIMA(p, d, q)(P, D, Q)[m] process, matching R's
``stats::arima(y, order, xreg=X)`` / ``forecast::Arima(..., include.drift=)``.

Key idea — differencing commutes with the linear regression, so the
model can be fit on the *differenced* scale exactly as the plain-ARIMA
path already does::

    diff(y - X @ beta) = diff(y) - diff(X) @ beta

The joint objective therefore evaluates the existing ARMA likelihood
(``_arima_likelihood.arima_negloglik``) on the regression residual of the
*differenced* series, with ``beta`` sitting in the optimizer vector next
to the (factored) ARMA parameters. Estimates, log-likelihood, and the
joint-Hessian standard errors match R because R's ``armafn`` does the
same thing (``x - xreg %*% par``).

Design-column and coefficient order matches R's ``coef()``::

    [ar, ma, sar, sma]  then  [intercept?, drift?, user_1, ..., user_k]

- ``intercept`` is present only when a mean is estimated (``include_mean``
  and no differencing, ``d + D == 0``) — a column of ones, which R folds
  into ``xreg``; under differencing ``diff(1) == 0`` so it drops out,
  which is exactly why R ignores ``include.mean`` when differencing.
- ``drift`` is the time index ``1..n``; ``diff(1:n) == 1`` (constant) so
  an ARIMA(p,1,q) "with drift" is an ARMA-with-mean on the differenced
  scale — the drift coefficient IS that mean, reproducing R's
  ``include.drift``/``d`` interaction with no special-casing.

The exact-ML / CSS path is CPU-only (as elsewhere in ``arima``); this
module never touches the Whittle / ``arima_batch`` GPU kernels.

This module owns only the regression machinery; the plain-ARIMA fit path
in ``_arima_fit`` is untouched when no regressor / drift / mask is
requested.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ConvergenceError, ValidationError
from pystatistics.core.result import Result
from pystatistics.timeseries._arima_factored import (
    _factored_to_effective,
    _multiply_ma_polynomials,
    _multiply_polynomials,
    normalize_to_invertible,
)
from pystatistics.core.validation import check_array, check_finite
from pystatistics.timeseries._arima_likelihood import (
    arima_css_residuals,
    arima_negloglik,
    check_invertible,
    check_stationary,
    compute_numerical_hessian,
    minimize_quiet,
)
from pystatistics.timeseries._arima_solution import ARIMAParams, ARIMASolution
from pystatistics.timeseries._differencing import diff
from numpy.typing import ArrayLike


def validate_xreg(xreg: ArrayLike, n_obs: int) -> NDArray:
    """Validate external regressors and return a 2-D ``(n_obs, k)`` array.

    Accepts a 1-D column or a 2-D matrix with one row per observation of
    *y*. Fails loud on non-finite entries or a row-count mismatch
    (Rule 1 / Rule 2 — validate external input at the boundary).
    """
    X = check_array(xreg, "xreg").astype(np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValidationError(
            f"xreg: must be 1-D or 2-D, got {X.ndim} dimensions"
        )
    if X.shape[0] != n_obs:
        raise ValidationError(
            f"xreg: has {X.shape[0]} rows but y has {n_obs} observations; "
            "xreg must have one row per observation of y"
        )
    if X.shape[1] == 0:
        raise ValidationError("xreg: must have at least one column")
    check_finite(X, "xreg")
    return X


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

def build_design(
    n: int,
    xreg: NDArray | None,
    include_intercept: bool,
    include_drift: bool,
) -> tuple[NDArray, tuple[str, ...]]:
    """Build the regression design matrix and coefficient names.

    Column order matches R: ``[intercept?, drift?, user_1..user_k]``.

    Parameters
    ----------
    n : int
        Length of the original (un-differenced) series.
    xreg : NDArray or None
        User regressors, shape ``(n, k)`` (or ``(n,)`` for a single
        column). ``None`` for no user regressors.
    include_intercept : bool
        Whether to prepend a column of ones ("intercept").
    include_drift : bool
        Whether to include a linear time-trend column ``1..n`` ("drift").

    Returns
    -------
    tuple[NDArray, tuple[str, ...]]
        Design matrix ``(n, m)`` and the ``m`` coefficient names.
    """
    cols: list[NDArray] = []
    names: list[str] = []
    if include_intercept:
        cols.append(np.ones(n, dtype=np.float64))
        names.append("intercept")
    if include_drift:
        cols.append(np.arange(1, n + 1, dtype=np.float64))
        names.append("drift")
    if xreg is not None:
        for j in range(xreg.shape[1]):
            cols.append(xreg[:, j].astype(np.float64))
            names.append(_user_name(xreg, j))
    if not cols:
        return np.empty((n, 0), dtype=np.float64), ()
    return np.column_stack(cols), tuple(names)


def _user_name(xreg: NDArray, j: int) -> str:
    """Column name for user regressor *j* (``xreg1``, ``xreg2``, ...)."""
    return f"xreg{j + 1}" if xreg.shape[1] > 1 else "xreg"


def difference_design(
    X: NDArray,
    d: int,
    seasonal_d: int,
    period: int,
) -> NDArray:
    """Difference the design matrix exactly as the series is differenced.

    Seasonal differencing (lag ``period``, ``seasonal_d`` times) is
    applied first, then regular differencing (``d`` times), matching the
    order in :func:`arima`. Column-wise linear differencing keeps the
    regression identity ``diff(y - X beta) = diff(y) - diff(X) beta``.
    """
    Xd = X
    if seasonal_d > 0 and period > 1:
        for _ in range(seasonal_d):
            Xd = np.column_stack([diff(Xd[:, j], differences=1, lag=period)
                                  for j in range(Xd.shape[1])]) \
                if Xd.shape[1] else Xd[period:]
    if d > 0:
        Xd = np.column_stack([diff(Xd[:, j], differences=d, lag=1)
                              for j in range(Xd.shape[1])]) \
            if Xd.shape[1] else Xd[d:]
    return Xd


# ---------------------------------------------------------------------------
# Fixed-parameter masking
# ---------------------------------------------------------------------------

def resolve_fixed(
    fixed: dict | NDArray | None,
    coef_names: tuple[str, ...],
) -> tuple[NDArray, NDArray]:
    """Resolve a ``fixed=`` specification into (values, free-mask) arrays.

    Parameters
    ----------
    fixed : dict or array-like or None
        Either a ``{name: value}`` mapping (the primary, Pythonic form —
        e.g. ``{'ma1': 0}`` holds ma1 at 0 and estimates the rest), or a
        positional array aligned to ``coef_names`` with ``nan`` marking
        free parameters (R's ``stats::arima`` convention). ``None`` means
        nothing is fixed.
    coef_names : tuple[str, ...]
        Full ordered coefficient names ``[ar.., ma.., sar.., sma..,
        reg..]``.

    Returns
    -------
    tuple[NDArray, NDArray]
        ``(values, free)`` — ``values[i]`` is the pinned value where
        ``free[i]`` is False (``nan`` where free), and ``free`` is a
        boolean mask of the estimated positions.

    Raises
    ------
    ValidationError
        On an unknown name, a length mismatch, or an all-fixed model.
    """
    k = len(coef_names)
    values = np.full(k, np.nan, dtype=np.float64)
    free = np.ones(k, dtype=bool)

    if fixed is None:
        return values, free

    if isinstance(fixed, dict):
        index = {name: i for i, name in enumerate(coef_names)}
        for name, val in fixed.items():
            if name not in index:
                raise ValidationError(
                    f"fixed: unknown coefficient name {name!r}. Valid "
                    f"names for this model: {list(coef_names)}"
                )
            values[index[name]] = float(val)
            free[index[name]] = False
    else:
        arr = np.asarray(fixed, dtype=np.float64).ravel()
        if arr.shape[0] != k:
            raise ValidationError(
                f"fixed: positional vector has length {arr.shape[0]} but "
                f"the model has {k} coefficients {list(coef_names)}. Pass "
                f"a length-{k} array (nan = estimate) or a name->value dict."
            )
        for i, val in enumerate(arr):
            if not np.isnan(val):
                values[i] = val
                free[i] = False

    if not free.any():
        raise ValidationError(
            "fixed: all coefficients are fixed — there is nothing to "
            "estimate. Leave at least one entry free (nan / omit from the "
            "dict)."
        )
    return values, free


# ---------------------------------------------------------------------------
# Joint objective
# ---------------------------------------------------------------------------

class _XregObjective:
    """Negative log-likelihood over the FREE parameters of a regression-

    with-ARIMA-errors model. Splices free params into the fixed template,
    forms the differenced regression residual, and delegates to the
    existing factored ARMA likelihood. Held as a small object so the
    (constant) problem structure is captured once and the optimizer sees
    a plain ``f(free) -> float``.
    """

    def __init__(
        self,
        y_diff: NDArray,
        X_diff: NDArray,
        shape: tuple[int, int, int, int],
        period: int,
        values: NDArray,
        free: NDArray,
    ) -> None:
        self.y_diff = y_diff
        self.X_diff = X_diff
        self.p, self.q, self.sp, self.sq = shape
        self.period = period
        self.narma = self.p + self.q + self.sp + self.sq
        self.p_eff = self.p + self.sp * period
        self.q_eff = self.q + self.sq * period
        self.values = values
        self.free = free
        self.k_reg = X_diff.shape[1]

    def full(self, free_params: NDArray) -> NDArray:
        """Splice the free parameter vector into the full coefficient vector."""
        theta = self.values.copy()
        theta[self.free] = free_params
        return theta

    def eta(self, theta: NDArray) -> NDArray:
        """Regression residual of the differenced series at ``theta``."""
        if self.k_reg:
            beta = theta[self.narma:]
            return self.y_diff - self.X_diff @ beta
        return self.y_diff

    def effective_arma(self, theta: NDArray) -> NDArray:
        """Expand the factored ARMA block to effective ``[ar_eff, ma_eff]``."""
        return _factored_to_effective(
            theta[:self.narma], self.p, self.q, self.sp, self.sq,
            self.period, False, _multiply_polynomials, _multiply_ma_polynomials,
        )

    def __call__(self, free_params: NDArray, method: str) -> float:
        theta = self.full(free_params)
        eff = self.effective_arma(theta)
        r = self.eta(theta)
        return arima_negloglik(eff, r, (self.p_eff, self.q_eff), False, method)


# ---------------------------------------------------------------------------
# Starting values
# ---------------------------------------------------------------------------

def _start_params(
    obj: _XregObjective,
    yule_walker_start,
    seasonal_fit: bool,
) -> NDArray:
    """Initial full coefficient vector: OLS for beta, Yule-Walker for AR.

    Mirrors R's preliminary regression (init xreg coefs by least squares,
    then AR by Yule-Walker on the regression residual). MA / seasonal-MA
    start at zero (non-seasonal) or a small negative value (seasonal),
    matching the plain-ARIMA path's heuristics.
    """
    p, q, sp, sq = obj.p, obj.q, obj.sp, obj.sq
    if obj.k_reg:
        beta0, *_ = np.linalg.lstsq(obj.X_diff, obj.y_diff, rcond=None)
        r0 = obj.y_diff - obj.X_diff @ beta0
    else:
        beta0 = np.zeros(0)
        r0 = obj.y_diff
    r0 = r0 - np.mean(r0)
    ar0 = np.clip(yule_walker_start(r0, p), -0.99, 0.99) if p else np.zeros(0)
    ma_fill = -0.1 if seasonal_fit else 0.0
    ma0 = np.full(q, ma_fill)
    sar0 = np.zeros(sp)
    sma0 = np.full(sq, -0.1)
    return np.concatenate([ar0, ma0, sar0, sma0, beta0])


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def _optimize(
    obj: _XregObjective,
    start_full: NDArray,
    method: str,
    tol: float,
    max_iter: int,
    intercept_free: bool,
) -> tuple[NDArray, float, bool, int, str]:
    """Optimize the free parameters (CSS / ML / CSS-ML).

    Returns ``(full_theta, nll, converged, n_iter, method_used)``.
    """
    free = obj.free
    start_free = start_full[free]
    opts = {"maxiter": max_iter, "ftol": tol}

    def run(m: str, x0: NDArray):
        return minimize_quiet(
            obj, x0, args=(m,), method="L-BFGS-B", options=opts,
        )

    if method == "css":
        res = run("css", start_free)
        return obj.full(res.x), res.fun, res.success, res.nit, "css"

    if method == "ml":
        res = run("ml", start_free)
        return obj.full(res.x), res.fun, res.success, res.nit, "ml"

    # CSS-ML: CSS warm-start, then ML refinement (with a second ML start
    # from the original point kept only if strictly better — the same
    # flat-canyon guard the plain-ARIMA path uses when a mean/intercept
    # is estimated).
    res_css = run("css", start_free)
    n_iter = res_css.nit
    try:
        res_ml = run("ml", res_css.x)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ConvergenceError(
            "ARIMA (xreg) CSS-ML: ML refinement failed numerically "
            f"({type(exc).__name__}: {exc}). Use method='css' for CSS "
            "estimates, or adjust tol / max_iter / starting values.",
            iterations=n_iter, reason="ml_numerical_failure",
        ) from exc
    n_iter += res_ml.nit
    if not (res_ml.success and np.isfinite(res_ml.fun)):
        raise ConvergenceError(
            "ARIMA (xreg) CSS-ML: ML refinement did not converge after "
            f"{res_ml.nit} iterations ({res_ml.message}). Use method='css' "
            "for CSS estimates, or increase max_iter / relax tol.",
            iterations=n_iter, reason="ml_max_iter_or_infeasible",
            threshold=tol,
        )
    best_x, best_nll = res_ml.x, res_ml.fun

    if intercept_free:
        try:
            res_ml2 = run("ml", start_free)
        except (ValueError, np.linalg.LinAlgError):
            res_ml2 = None
        if res_ml2 is not None:
            n_iter += res_ml2.nit
            if (res_ml2.success and np.isfinite(res_ml2.fun)
                    and res_ml2.fun < best_nll):
                best_x, best_nll = res_ml2.x, res_ml2.fun

    return obj.full(best_x), best_nll, True, n_iter, "css-ml"


# ---------------------------------------------------------------------------
# Public fit entry point
# ---------------------------------------------------------------------------

def fit_arima_xreg(
    *,
    arr: NDArray,
    y_diff: NDArray,
    order: tuple[int, int, int],
    seasonal: tuple[int, int, int, int] | None,
    xreg: NDArray | None,
    include_mean: bool,
    include_drift: bool,
    fixed: dict | NDArray | None,
    method: str,
    tol: float,
    max_iter: int,
    n_obs: int,
    n_used: int,
    yule_walker_start,
) -> ARIMASolution:
    """Fit a regression-with-ARIMA-errors model and build its solution.

    Called by :func:`arima` when any of ``xreg`` / ``include_drift`` /
    ``fixed`` is active. ``arr`` is the original series, ``y_diff`` the
    already-differenced series (both produced by :func:`arima`).
    ``yule_walker_start`` is injected from ``_arima_fit`` to avoid a
    circular import.
    """
    p, d, q = order
    if seasonal is not None:
        sp, sd, sq, m = seasonal
    else:
        sp, sd, sq, m = 0, 0, 0, 1

    include_intercept = include_mean and (d + sd == 0)

    # ----- Design matrix (original scale) then differenced -----
    X_full, reg_names = build_design(n_obs, xreg, include_intercept, include_drift)
    X_diff = difference_design(X_full, d, sd, m)
    if X_diff.shape[0] != n_used:
        raise ValidationError(
            f"xreg: differenced design has {X_diff.shape[0]} rows but the "
            f"differenced series has {n_used}; check that xreg has one row "
            f"per observation of y (n={n_obs})."
        )
    if include_drift and X_diff.shape[1] and "drift" in reg_names:
        di = reg_names.index("drift")
        if not np.any(np.abs(X_diff[:, di]) > 1e-12):
            raise ValidationError(
                "include_drift=True with total differencing order "
                f"d + D = {d + sd} >= 2: the drift regressor vanishes "
                "under differencing and cannot be identified. Drop the "
                "drift term or reduce differencing."
            )

    # ----- Coefficient layout and fixed mask -----
    arma_names = (
        tuple(f"ar{i + 1}" for i in range(p))
        + tuple(f"ma{i + 1}" for i in range(q))
        + tuple(f"sar{i + 1}" for i in range(sp))
        + tuple(f"sma{i + 1}" for i in range(sq))
    )
    coef_names = arma_names + reg_names
    values, free = resolve_fixed(fixed, coef_names)

    seasonal_fit = seasonal is not None and (sp > 0 or sq > 0)
    obj = _XregObjective(
        y_diff, X_diff, (p, q, sp, sq), m, values, free,
    )

    # ----- Starting values (respecting fixed values) -----
    start_full = _start_params(obj, yule_walker_start, seasonal_fit)
    start_full[~free] = values[~free]

    intercept_free = "intercept" in reg_names and free[coef_names.index("intercept")]
    theta, nll, converged, n_iter, method_used = _optimize(
        obj, start_full, method, tol, max_iter, intercept_free,
    )
    if not converged:
        raise ConvergenceError(
            f"ARIMA (xreg) optimization did not converge after {n_iter} "
            "iterations", iterations=n_iter, reason="optimizer_failed",
        )

    # ----- Canonical invertible MA representation (ML-family only) -----
    if method_used in ("ml", "css-ml"):
        arma = theta[:obj.narma]
        eff = obj.effective_arma(theta)
        r = obj.eta(theta)
        eff_norm, fac_norm, nll = normalize_to_invertible(
            eff, arma.copy(), nll, r, p, q, sp, sq, m, False,
        )
        if fac_norm is not None:
            theta[:obj.narma] = fac_norm

    # ----- Split final parameters -----
    arma = theta[:obj.narma]
    ar_ns = arma[:p]
    ma_ns = arma[p:p + q]
    sar = arma[p + q:p + q + sp]
    sma = arma[p + q + sp:p + q + sp + sq]
    beta = theta[obj.narma:] if obj.k_reg else np.empty(0)

    ar_eff = _multiply_polynomials(ar_ns, sar, m) if sp > 0 else ar_ns
    ma_eff = _multiply_ma_polynomials(ma_ns, sma, m) if sq > 0 else ma_ns
    r = obj.eta(theta)

    # ----- Residuals / sigma2 (same convention as the plain path) -----
    if method_used in ("ml", "css-ml"):
        from pystatistics.timeseries._arima_kalman import kalman_arma_innovations
        residuals = kalman_arma_innovations(r, ar_eff, ma_eff)
        sigma2 = float(np.mean(residuals * residuals))
    else:
        residuals = arima_css_residuals(r, ar_eff, ma_eff, 0.0)
        sigma2 = float(np.dot(residuals, residuals) / n_used)
    fitted = y_diff - residuals

    if len(ar_eff) and not check_stationary(ar_eff):
        warnings.warn(
            "AR polynomial has roots inside or on the unit circle; the "
            "model may be non-stationary", stacklevel=2,
        )
    if len(ma_eff) and not check_invertible(ma_eff):
        warnings.warn(
            "MA polynomial has roots inside or on the unit circle; the "
            "model may be non-invertible", stacklevel=2,
        )

    # ----- Joint variance-covariance over the FREE parameters -----
    vcov = _joint_vcov(obj, theta, method_used)

    # ----- Information criteria (free params + sigma2) -----
    n_free = int(free.sum())
    k = n_free + 1
    loglik = -nll
    aic = -2.0 * loglik + 2.0 * k
    bic = -2.0 * loglik + k * np.log(n_used)
    aicc = (aic + 2.0 * k * (k + 1.0) / (n_used - k - 1.0)
            if n_used - k - 1 > 0 else np.inf)

    return ARIMASolution(
        _result=Result(
            params=ARIMAParams(
                order=order,
                seasonal_order=seasonal,
                ar=ar_ns,
                ma=ma_ns,
                seasonal_ar=sar,
                seasonal_ma=sma,
                mean=None,
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
                xreg_coef=beta,
                xreg_names=reg_names,
                include_drift=include_drift,
                xreg=np.asarray(xreg, dtype=np.float64).reshape(n_obs, -1)
                if xreg is not None else None,
            ),
            info={"method": method_used, "converged": converged},
            timing=None,
            backend_name="cpu",
            warnings=(),
        )
    )


def _joint_vcov(
    obj: _XregObjective,
    theta: NDArray,
    method_used: str,
) -> NDArray:
    """Full ``(narma + k_reg)`` vcov from the numeric Hessian of the NLL.

    The Hessian is taken over the FREE parameters only (the objective's
    natural argument); fixed coefficients get zero variance rows/columns
    so the matrix stays aligned with the reported coefficient vector.
    """
    k_total = obj.narma + obj.k_reg
    hess_method = "ml" if method_used == "ml" else "css"
    if method_used == "css-ml":
        hess_method = "ml"
    free_theta = theta[obj.free]

    def nll_free(fp: NDArray) -> float:
        return obj(fp, hess_method)

    vcov = np.zeros((k_total, k_total), dtype=np.float64)
    try:
        hess = compute_numerical_hessian(nll_free, free_theta)
        free_vcov = np.linalg.inv(hess)
        free_vcov = 0.5 * (free_vcov + free_vcov.T)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Hessian is singular; variance-covariance matrix could not be "
            "computed", stacklevel=2,
        )
        free_vcov = np.full((int(obj.free.sum()), int(obj.free.sum())), np.nan)

    idx = np.where(obj.free)[0]
    for a, ia in enumerate(idx):
        for b, ib in enumerate(idx):
            vcov[ia, ib] = free_vcov[a, b]
    return vcov
