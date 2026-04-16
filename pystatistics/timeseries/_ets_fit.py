"""
ETS model fitting via maximum likelihood.

Fits ETS(error, trend, season) models by maximizing the Gaussian
log-likelihood over smoothing parameters and initial states.  Uses
``scipy.optimize.minimize(method='L-BFGS-B')`` with logit-transformed
parameters for numerical stability.

Design matches R's ``forecast::ets()`` for specified model types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from pystatistics.core.exceptions import ConvergenceError, ValidationError
from pystatistics.core.validation import check_array, check_1d, check_finite
from pystatistics.timeseries._ets_models import (
    ETSSpec,
    ets_recursion,
    parse_ets_spec,
    unpack_params,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ETSResult:
    """
    Result from fitting an ETS model.

    Attributes
    ----------
    spec : ETSSpec
        The fitted model specification.
    alpha : float
        Level smoothing parameter.
    beta : float or None
        Trend smoothing parameter (``None`` if no trend).
    gamma : float or None
        Seasonal smoothing parameter (``None`` if no season).
    phi : float or None
        Damping parameter (``None`` if not damped).
    init_level : float
        Estimated initial level.
    init_trend : float or None
        Estimated initial trend (``None`` if no trend).
    init_season : NDArray or None
        Estimated initial seasonal indices (``None`` if no season).
    fitted_values : NDArray
        One-step-ahead fitted values, length *n*.
    residuals : NDArray
        Residuals, length *n*.
    states : NDArray
        Full state history, shape ``(n + 1, n_states)``.
    log_likelihood : float
        Maximised log-likelihood.
    aic : float
        Akaike Information Criterion.
    aicc : float
        Corrected AIC (for small samples).
    bic : float
        Bayesian Information Criterion.
    mse : float
        Mean squared error of residuals.
    mae : float
        Mean absolute error of residuals.
    n_obs : int
        Number of observations.
    n_params : int
        Total number of estimated parameters (smoothing + initial states + sigma^2).
    converged : bool
        Whether the optimiser converged.
    """

    spec: ETSSpec
    alpha: float
    beta: float | None
    gamma: float | None
    phi: float | None
    init_level: float
    init_trend: float | None
    init_season: NDArray | None
    fitted_values: NDArray
    residuals: NDArray
    states: NDArray
    log_likelihood: float
    aic: float
    aicc: float
    bic: float
    mse: float
    mae: float
    n_obs: int
    n_params: int
    converged: bool

    def summary(self) -> str:
        """
        Return a human-readable summary matching R's ``forecast::ets()`` style.

        Returns
        -------
        str
            Multi-line summary.
        """
        lines = [
            self.spec.name,
            "",
            "  Smoothing parameters:",
            f"    alpha = {self.alpha:.4f}",
        ]
        if self.beta is not None:
            lines.append(f"    beta  = {self.beta:.4f}")
        if self.gamma is not None:
            lines.append(f"    gamma = {self.gamma:.4f}")
        if self.phi is not None:
            lines.append(f"    phi   = {self.phi:.4f}")
        lines.append("")
        lines.append("  Initial states:")
        lines.append(f"    l = {self.init_level:.4f}")
        if self.init_trend is not None:
            lines.append(f"    b = {self.init_trend:.4f}")
        if self.init_season is not None:
            s_str = ", ".join(f"{v:.4f}" for v in self.init_season)
            lines.append(f"    s = [{s_str}]")
        lines.extend([
            "",
            f"  sigma^2: {self.mse:.4f}",
            "",
            f"  Log-likelihood: {self.log_likelihood:.2f}",
            f"  AIC:  {self.aic:.2f}",
            f"  AICc: {self.aicc:.2f}",
            f"  BIC:  {self.bic:.2f}",
            "",
            f"  MSE: {self.mse:.4f}",
            f"  MAE: {self.mae:.4f}",
            f"  n = {self.n_obs}, k = {self.n_params}",
            f"  Converged: {self.converged}",
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Logit / inv-logit transforms for bounded optimisation
# ---------------------------------------------------------------------------

def _logit(x: float, lo: float, hi: float) -> float:
    """Map (lo, hi) -> (-inf, inf)."""
    p = (x - lo) / (hi - lo)
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    return float(np.log(p / (1.0 - p)))


def _inv_logit(z: float, lo: float, hi: float) -> float:
    """Map (-inf, inf) -> (lo, hi)."""
    return lo + (hi - lo) / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Initial state heuristics
# ---------------------------------------------------------------------------

def _init_level_trend(y: NDArray, spec: ETSSpec) -> tuple[float, float | None]:
    """
    Estimate initial level and trend from the data.

    For non-seasonal models, uses the first 10 observations (or fewer).
    For seasonal models, uses the mean of the first complete period for
    level and a simple slope for trend.

    Parameters
    ----------
    y : NDArray
        Time series.
    spec : ETSSpec
        Model specification.

    Returns
    -------
    tuple
        ``(level, trend_or_None)``
    """
    m = spec.period
    has_season = spec.season in ("A", "M")

    if has_season and len(y) >= 2 * m:
        level = float(np.mean(y[:m]))
    else:
        k = min(10, len(y))
        level = float(np.mean(y[:k]))

    trend: float | None = None
    if spec.trend in ("A", "Ad"):
        if has_season and len(y) >= 2 * m:
            # Average slope across first two periods
            trend = float(np.mean(y[m : 2 * m] - y[:m]) / m)
        else:
            k = min(10, len(y))
            if k >= 2:
                trend = float((y[k - 1] - y[0]) / (k - 1))
            else:
                trend = 0.0

    return level, trend


def _init_season(y: NDArray, spec: ETSSpec, level: float) -> NDArray | None:
    """
    Estimate initial seasonal indices via classical decomposition.

    Parameters
    ----------
    y : NDArray
        Time series.
    spec : ETSSpec
        Model specification.
    level : float
        Initial level estimate.

    Returns
    -------
    NDArray or None
        Seasonal indices of length ``period``, or ``None`` if no season.
    """
    if spec.season == "N":
        return None

    m = spec.period
    n_full = min(len(y), 3 * m)
    y_sub = y[:n_full]

    if spec.season == "A":
        # Additive: s_i = mean(y[i::m]) - level
        season = np.array([float(np.mean(y_sub[i::m])) - level for i in range(m)])
        # Centre so they sum to zero
        season -= np.mean(season)
    else:
        # Multiplicative: s_i = mean(y[i::m]) / level
        if abs(level) < 1e-15:
            season = np.ones(m)
        else:
            season = np.array(
                [float(np.mean(y_sub[i::m])) / level for i in range(m)]
            )
            # Normalise so product = 1 (equivalent: mean = 1)
            season *= m / np.sum(season)

    return season


# ---------------------------------------------------------------------------
# Negative log-likelihood
# ---------------------------------------------------------------------------

def _neg_loglik(
    theta: NDArray,
    y: NDArray,
    spec: ETSSpec,
    bounds_info: list[tuple[float, float]],
    n_smooth: int,
) -> float:
    """
    Compute negative log-likelihood for the optimiser.

    Parameters are received on an unconstrained (logit) scale and mapped
    back to their bounded ranges before evaluation.

    Parameters
    ----------
    theta : NDArray
        Unconstrained parameter vector.
    y : NDArray
        Time series.
    spec : ETSSpec
        Model specification.
    bounds_info : list of (lo, hi)
        Bounds for each element of the parameter vector.
    n_smooth : int
        Number of smoothing parameters at the front of ``theta``.

    Returns
    -------
    float
        Negative log-likelihood (large positive means bad fit).
    """
    # Map from unconstrained to bounded
    real = np.empty(len(theta))
    for i in range(n_smooth):
        real[i] = _inv_logit(theta[i], bounds_info[i][0], bounds_info[i][1])
    # Initial states are unbounded
    real[n_smooth:] = theta[n_smooth:]

    params = real[:n_smooth]
    init_states = real[n_smooth:]

    try:
        fitted, residuals, _ = ets_recursion(y, spec, params, init_states)
    except (FloatingPointError, ZeroDivisionError):
        return 1e20

    n = len(y)

    if spec.error == "A":
        sigma2 = float(np.mean(residuals ** 2))
        if sigma2 < 1e-30:
            sigma2 = 1e-30
        nll = 0.5 * n * np.log(2.0 * np.pi) + 0.5 * n * np.log(sigma2) + 0.5 * n
    else:
        # Multiplicative error: residuals are relative errors (y/mu - 1)
        sigma2 = float(np.mean(residuals ** 2))
        if sigma2 < 1e-30:
            sigma2 = 1e-30
        # Jacobian term: sum of log|mu_t|
        log_abs_fitted = np.log(np.abs(fitted) + 1e-30)
        nll = (
            0.5 * n * np.log(2.0 * np.pi)
            + 0.5 * n * np.log(sigma2)
            + 0.5 * n
            + float(np.sum(log_abs_fitted))
        )

    if not np.isfinite(nll):
        return 1e20

    return float(nll)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ets(
    y: ArrayLike,
    *,
    model: str = "ANN",
    period: int = 1,
    damped: bool | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    phi: float | None = None,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> ETSResult:
    """
    Fit an ETS (ExponenTial Smoothing) state space model.

    Estimates smoothing parameters and initial states by maximising the
    Gaussian log-likelihood, matching R's ``forecast::ets()`` for
    specified model types.

    Parameters
    ----------
    y : ArrayLike
        Time series (1-D, must be positive for multiplicative error/season).
    model : str
        ETS model string, e.g. ``'ANN'``, ``'AAN'``, ``'AAA'``, ``'MAM'``.
    period : int
        Seasonal period (e.g. 12 for monthly, 4 for quarterly).
    damped : bool or None
        Force damped trend.  Overrides the model string when not ``None``.
    alpha, beta, gamma, phi : float or None
        Fix specific smoothing parameters (skip optimisation for them).
    tol : float
        Convergence tolerance for the optimiser.
    max_iter : int
        Maximum optimiser iterations.

    Returns
    -------
    ETSResult
        Fitted model with diagnostics.

    Raises
    ------
    ValidationError
        If inputs are invalid.
    ConvergenceError
        If the optimiser fails to converge.
    """
    # ---- validate ---------------------------------------------------------
    y_arr = check_array(y, "y").ravel()
    check_1d(y_arr, "y")
    check_finite(y_arr, "y")
    n = len(y_arr)
    if n < 3:
        raise ValidationError(f"y: requires at least 3 observations, got {n}")

    spec = parse_ets_spec(model, period=period)

    # Handle explicit damped override
    if damped is not None:
        if damped and spec.trend == "N":
            raise ValidationError("damped=True requires a trend component (model has trend='N')")
        if damped and spec.trend == "A":
            spec = parse_ets_spec(
                f"{spec.error}AdN" if spec.season == "N" else f"{spec.error}Ad{spec.season}",
                period=spec.period,
            )
        elif not damped and spec.trend == "Ad":
            spec = parse_ets_spec(
                f"{spec.error}AN" if spec.season == "N" else f"{spec.error}A{spec.season}",
                period=spec.period,
            )

    # Seasonal needs enough data
    if spec.season in ("A", "M") and n < 2 * spec.period:
        raise ValidationError(
            f"y: seasonal model with period={spec.period} requires at least "
            f"{2 * spec.period} observations, got {n}"
        )

    # Multiplicative models need positive data
    if spec.error == "M" or spec.season == "M":
        if np.any(y_arr <= 0):
            raise ValidationError(
                "y: multiplicative error or season requires strictly positive values"
            )

    # ---- initial states ---------------------------------------------------
    init_l, init_b = _init_level_trend(y_arr, spec)
    init_s = _init_season(y_arr, spec, init_l)

    # ---- build parameter vector and bounds --------------------------------
    # Smoothing params first, then initial states (unbounded)
    smooth_vals: list[float] = []
    smooth_bounds: list[tuple[float, float]] = []
    smooth_fixed: list[bool] = []
    eps = 1e-4

    # alpha
    smooth_vals.append(alpha if alpha is not None else 0.1)
    smooth_bounds.append((eps, 1.0 - eps))
    smooth_fixed.append(alpha is not None)

    # beta
    if spec.trend in ("A", "Ad"):
        smooth_vals.append(beta if beta is not None else 0.01)
        smooth_bounds.append((eps, 1.0 - eps))
        smooth_fixed.append(beta is not None)

    # gamma
    if spec.season in ("A", "M"):
        smooth_vals.append(gamma if gamma is not None else 0.01)
        smooth_bounds.append((eps, 1.0 - eps))
        smooth_fixed.append(gamma is not None)

    # phi
    if spec.damped:
        smooth_vals.append(phi if phi is not None else 0.98)
        smooth_bounds.append((0.8, 0.999))
        smooth_fixed.append(phi is not None)

    n_smooth = len(smooth_vals)

    # Initial states (unbounded)
    init_state_vals: list[float] = [init_l]
    if init_b is not None:
        init_state_vals.append(init_b)
    if init_s is not None:
        init_state_vals.extend(init_s.tolist())

    all_vals = np.array(smooth_vals + init_state_vals, dtype=np.float64)
    all_bounds = smooth_bounds + [(-np.inf, np.inf)] * len(init_state_vals)

    # Transform smoothing params to unconstrained scale
    theta0 = np.empty_like(all_vals)
    for i in range(n_smooth):
        if smooth_fixed[i]:
            theta0[i] = _logit(all_vals[i], all_bounds[i][0], all_bounds[i][1])
        else:
            theta0[i] = _logit(all_vals[i], all_bounds[i][0], all_bounds[i][1])
    theta0[n_smooth:] = all_vals[n_smooth:]

    # Build mask for fixed parameters
    fixed_mask = smooth_fixed + [False] * len(init_state_vals)

    # If all fixed, skip optimisation
    all_fixed = all(fixed_mask)

    if all_fixed:
        params_opt = all_vals[:n_smooth]
        init_states_opt = all_vals[n_smooth:]
        converged = True
    else:
        # Optimise only free parameters
        free_idx = [i for i, f in enumerate(fixed_mask) if not f]
        fixed_idx = [i for i, f in enumerate(fixed_mask) if f]

        def _objective(theta_free: NDArray) -> float:
            theta_full = theta0.copy()
            theta_full[free_idx] = theta_free
            return _neg_loglik(theta_full, y_arr, spec, all_bounds, n_smooth)

        theta_free0 = theta0[free_idx]

        result = minimize(
            _objective,
            theta_free0,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
        )
        converged = result.success

        # Reconstruct full theta
        theta_opt = theta0.copy()
        theta_opt[free_idx] = result.x

        # Map back to bounded
        real_opt = np.empty(len(theta_opt))
        for i in range(n_smooth):
            real_opt[i] = _inv_logit(theta_opt[i], all_bounds[i][0], all_bounds[i][1])
        real_opt[n_smooth:] = theta_opt[n_smooth:]

        params_opt = real_opt[:n_smooth]
        init_states_opt = real_opt[n_smooth:]

    # ---- final recursion --------------------------------------------------
    fitted_vals, resid, states = ets_recursion(
        y_arr, spec, params_opt, init_states_opt
    )

    # ---- extract named parameters -----------------------------------------
    a_opt, b_opt, g_opt, p_opt = unpack_params(params_opt, spec)

    # ---- information criteria ---------------------------------------------
    if spec.error == "A":
        sigma2 = float(np.mean(resid ** 2))
        if sigma2 < 1e-30:
            sigma2 = 1e-30
        ll = -0.5 * n * np.log(2.0 * np.pi) - 0.5 * n * np.log(sigma2) - 0.5 * n
    else:
        sigma2 = float(np.mean(resid ** 2))
        if sigma2 < 1e-30:
            sigma2 = 1e-30
        log_abs_fitted = np.log(np.abs(fitted_vals) + 1e-30)
        ll = (
            -0.5 * n * np.log(2.0 * np.pi)
            - 0.5 * n * np.log(sigma2)
            - 0.5 * n
            - float(np.sum(log_abs_fitted))
        )

    # Total parameters: smoothing + init states + sigma^2
    k = n_smooth + len(init_state_vals) + 1
    aic = -2.0 * ll + 2.0 * k
    if n - k - 1 > 0:
        aicc = aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)
    else:
        aicc = float("inf")
    bic = -2.0 * ll + k * np.log(n)

    mse = float(np.mean((y_arr - fitted_vals) ** 2))
    mae = float(np.mean(np.abs(y_arr - fitted_vals)))

    # ---- unpack init states for result ------------------------------------
    idx = 0
    result_init_l = float(init_states_opt[idx]); idx += 1
    result_init_b: float | None = None
    if spec.trend in ("A", "Ad"):
        result_init_b = float(init_states_opt[idx]); idx += 1
    result_init_s: NDArray | None = None
    if spec.season in ("A", "M"):
        result_init_s = np.array(init_states_opt[idx : idx + spec.period])

    return ETSResult(
        spec=spec,
        alpha=a_opt,
        beta=b_opt,
        gamma=g_opt,
        phi=p_opt,
        init_level=result_init_l,
        init_trend=result_init_b,
        init_season=result_init_s,
        fitted_values=fitted_vals,
        residuals=resid,
        states=states,
        log_likelihood=float(ll),
        aic=float(aic),
        aicc=float(aicc),
        bic=float(bic),
        mse=mse,
        mae=mae,
        n_obs=n,
        n_params=k,
        converged=converged,
    )
