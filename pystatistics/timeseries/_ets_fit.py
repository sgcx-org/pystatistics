"""
ETS model fitting via maximum likelihood.

Fits ETS(error, trend, season) models by maximizing the Gaussian
log-likelihood over smoothing parameters and initial states.  Uses
``scipy.optimize.minimize(method='L-BFGS-B')`` with logit-transformed
parameters for numerical stability.

Design matches R's ``forecast::ets()`` for specified model types.  The
optimised parameter space is R's default "usual" region (forecast's
``bounds="usual"`` box constraints plus cross-constraints): ``1e-4 <
alpha < 1 - 1e-4``, ``1e-4 < beta < alpha``, ``1e-4 < gamma < 1 -
alpha``, ``0.8 < phi < 0.98``; seasonal models estimate ``m - 1`` free
initial seasonal states with the remaining one (the index used at the
first observation) determined by the normalisation ``sum(s) = 0``
(additive) / ``sum(s) = m`` (multiplicative), exactly as R does — which
also makes ``n_params`` match R's parameter count.  Free-parameter
starting values follow R's ``initparam``.  (R's ``bounds="both"``
default additionally intersects with the admissible region; that check
is not implemented — for the usual region it only matters on its
boundary.)

There is one deliberate reporting divergence: PyStatistics reports the
**full Gaussian log-likelihood**, while ``forecast::ets`` reports
Hyndman's concentrated pseudo-log-likelihood ``-0.5 * n * log(SSE)``
(plus the same multiplicative-error Jacobian term both conventions
share).  The
two differ by the exact deterministic constant

    ``0.5 * n * [log(n / (2*pi)) - 1]``

which depends only on the sample size — e.g. +88.36 for n = 100.  The
parameter count ``k`` is identical under both conventions, so AIC/AICc/
BIC *differences and rankings* between models on the same data are
identical, and automatic model selection (``model="ZZZ"``) picks the
same model either way.  Only the printed log-likelihood/AIC numbers
differ from R's.  Public model selection lives in ``_ets_select.py``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from pystatistics.core.exceptions import ConvergenceError, ValidationError
from pystatistics.core.result import Result
from pystatistics.core.validation import check_array, check_1d, check_finite
from pystatistics.timeseries._ets_models import (
    ETSSpec,
    ets_recursion,
    parse_ets_spec,
    unpack_params,
)
# Result containers live in _ets_result.py; re-exported here because the
# public surface (timeseries/__init__.py, _ets_select.py, _ets_forecast.py,
# existing user code) imports them from this module.
from pystatistics.timeseries._ets_result import ETSParams, ETSSolution

__all__ = ["ETSParams", "ETSSolution", "fit_ets_model"]


# ---------------------------------------------------------------------------
# Parameter space: R forecast::ets "usual" region
# ---------------------------------------------------------------------------

_EPS = 1e-4                  # R: lower = 1e-4, upper = 1 - 1e-4 (alpha/beta/gamma)
_PHI_BOUNDS = (0.8, 0.98)    # R: lower[4] = 0.8, upper[4] = 0.98


def _alpha_box(beta_fixed: float | None, gamma_fixed: float | None) -> tuple[float, float]:
    """Alpha's box, narrowed by user-fixed beta/gamma exactly as R's
    etsmodel does (``lower[1] <- max(beta, lower[1])``;
    ``upper[1] <- min(1 - gamma, upper[1])``) so the usual-region
    cross-constraints stay satisfiable."""
    lo = max(_EPS, beta_fixed) if beta_fixed is not None else _EPS
    hi = min(1.0 - _EPS, 1.0 - gamma_fixed) if gamma_fixed is not None else 1.0 - _EPS
    return lo, hi


def _check_fixed_smoothing(
    alpha: float | None,
    beta: float | None,
    gamma: float | None,
    phi: float | None,
) -> None:
    """Validate user-fixed smoothing parameters against R's usual region
    (forecast's check.param), failing loud instead of silently coercing
    an out-of-range value into bounds.  Only parameters the model uses
    are passed in (others arrive as ``None``)."""
    if alpha is not None and not _EPS <= alpha <= 1.0 - _EPS:
        raise ValidationError(
            f"alpha: must be in [{_EPS}, {1.0 - _EPS}] (R forecast::ets "
            f"usual region), got {alpha}"
        )
    if beta is not None:
        if not _EPS <= beta <= 1.0 - _EPS:
            raise ValidationError(
                f"beta: must be in [{_EPS}, {1.0 - _EPS}] (R forecast::ets "
                f"usual region), got {beta}"
            )
        if alpha is not None and beta > alpha:
            raise ValidationError(
                f"beta: usual region requires beta <= alpha, got "
                f"beta={beta} > alpha={alpha}"
            )
    if gamma is not None:
        if not _EPS <= gamma <= 1.0 - _EPS:
            raise ValidationError(
                f"gamma: must be in [{_EPS}, {1.0 - _EPS}] (R forecast::ets "
                f"usual region), got {gamma}"
            )
        if alpha is not None and gamma > 1.0 - alpha:
            raise ValidationError(
                f"gamma: usual region requires gamma <= 1 - alpha, got "
                f"gamma={gamma} > 1 - alpha={1.0 - alpha}"
            )
    if phi is not None and not _PHI_BOUNDS[0] <= phi <= _PHI_BOUNDS[1]:
        raise ValidationError(
            f"phi: must be in [{_PHI_BOUNDS[0]}, {_PHI_BOUNDS[1]}] "
            f"(R forecast::ets usual region), got {phi}"
        )


# ---------------------------------------------------------------------------
# Logit / inv-logit transforms for bounded optimisation
# ---------------------------------------------------------------------------

def _logit(x: float, lo: float, hi: float) -> float:
    """Map (lo, hi) -> (-inf, inf)."""
    p = (x - lo) / (hi - lo)
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    return float(np.log(p / (1.0 - p)))


def _inv_logit(z: float, lo: float, hi: float) -> float:
    """Map (-inf, inf) -> (lo, hi).

    ``z`` is clipped to [-500, 500] before exponentiation: beyond that the
    sigmoid is saturated to lo/hi far below one ulp (exp(500) ~ 1.4e217),
    so results are bit-identical while ``np.exp`` can no longer overflow
    (L-BFGS-B probes such z during line searches).
    """
    z = min(max(z, -500.0), 500.0)
    return lo + (hi - lo) / (1.0 + np.exp(-z))


def _decode_smooth(
    theta_smooth: NDArray,
    spec: ETSSpec,
    fixed_smooth: list[float | None],
    alpha_box: tuple[float, float],
) -> NDArray:
    """Map the smoothing block of the unconstrained parameter vector into
    R's usual region.

    Alpha is decoded first; beta/gamma bounds then depend on the current
    alpha value (``beta < alpha``, ``gamma < 1 - alpha``), so the
    cross-constraints hold at every optimiser iterate.  User-fixed values
    pass through untransformed (their theta entries are unused).
    """
    out = np.empty(len(fixed_smooth), dtype=np.float64)
    a = (
        fixed_smooth[0]
        if fixed_smooth[0] is not None
        else _inv_logit(theta_smooth[0], alpha_box[0], alpha_box[1])
    )
    out[0] = a
    i = 1
    if spec.trend in ("A", "Ad"):
        out[i] = (
            fixed_smooth[i]
            if fixed_smooth[i] is not None
            else _inv_logit(theta_smooth[i], _EPS, a)
        )
        i += 1
    if spec.season in ("A", "M"):
        out[i] = (
            fixed_smooth[i]
            if fixed_smooth[i] is not None
            else _inv_logit(theta_smooth[i], _EPS, 1.0 - a)
        )
        i += 1
    if spec.damped:
        out[i] = (
            fixed_smooth[i]
            if fixed_smooth[i] is not None
            else _inv_logit(theta_smooth[i], _PHI_BOUNDS[0], _PHI_BOUNDS[1])
        )
    return out


def _assemble_init_states(free_states: NDArray, spec: ETSSpec) -> NDArray:
    """Expand the optimiser's free initial states to the full state vector.

    Seasonal models optimise ``m - 1`` initial seasonal states (as R
    forecast::ets does); the remaining one — the index used at the first
    observation — is determined by the normalisation ``sum(s) = 0``
    (additive) / ``sum(s) = m`` (multiplicative).
    """
    if spec.season == "N":
        return np.asarray(free_states, dtype=np.float64)
    n_lead = 1 + (1 if spec.trend in ("A", "Ad") else 0)
    lead = free_states[:n_lead]
    s_free = free_states[n_lead:]
    target = 0.0 if spec.season == "A" else float(spec.period)
    s_first = target - float(np.sum(s_free))
    return np.concatenate([lead, [s_first], s_free])


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

def _sigma2_floor(y: NDArray) -> float:
    """Variance floor for the (concentrated) Gaussian likelihood.

    A perfectly-fit (noiseless) series drives residual variance toward zero,
    which sends the additive/multiplicative log-likelihood to -infinity and
    leaves L-BFGS-B chasing an open boundary with an exploding gradient
    (it terminates ABNORMAL in the line search). Flooring sigma^2 relative to
    the data scale keeps the objective bounded and its gradient finite, so
    degenerate inputs converge cleanly. The floor is ~1e-12 of the data
    variance — far below any real noise level — so fits on genuine (noisy)
    series are numerically unchanged.
    """
    return 1e-12 * (float(np.var(y)) + 1e-30)


def _neg_loglik(
    theta: NDArray,
    y: NDArray,
    spec: ETSSpec,
    fixed_smooth: list[float | None],
    alpha_box: tuple[float, float],
    n_smooth: int,
) -> float:
    """
    Compute negative log-likelihood for the optimiser.

    Smoothing parameters are received on an unconstrained (logit) scale
    and mapped into R's usual region before evaluation; the free initial
    states follow unbounded and are expanded to the full state vector
    (seasonal normalisation).

    Parameters
    ----------
    theta : NDArray
        Unconstrained parameter vector.
    y : NDArray
        Time series.
    spec : ETSSpec
        Model specification.
    fixed_smooth : list of float or None
        Per-smoothing-parameter user-fixed values (``None`` = free).
    alpha_box : (lo, hi)
        Alpha's box bounds.
    n_smooth : int
        Number of smoothing parameters at the front of ``theta``.

    Returns
    -------
    float
        Negative log-likelihood (large positive means bad fit).
    """
    params = _decode_smooth(theta[:n_smooth], spec, fixed_smooth, alpha_box)
    init_states = _assemble_init_states(theta[n_smooth:], spec)

    try:
        fitted, residuals, _ = ets_recursion(y, spec, params, init_states)
    except (FloatingPointError, ZeroDivisionError):
        return 1e20

    n = len(y)

    floor = _sigma2_floor(y)
    if spec.error == "A":
        sigma2 = float(np.mean(residuals ** 2))
        if sigma2 < floor:
            sigma2 = floor
        nll = 0.5 * n * np.log(2.0 * np.pi) + 0.5 * n * np.log(sigma2) + 0.5 * n
    else:
        # Multiplicative error: residuals are relative errors (y/mu - 1)
        sigma2 = float(np.mean(residuals ** 2))
        if sigma2 < floor:
            sigma2 = floor
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

def fit_ets_model(
    y: ArrayLike,
    *,
    model: str,
    period: int = 1,
    damped: bool | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    phi: float | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> ETSSolution:
    """
    Fit one fully-specified ETS state space model (the fitting engine).

    Estimates smoothing parameters and initial states by maximising the
    Gaussian log-likelihood, matching R's ``forecast::ets()`` for
    specified model types (see the module docstring for the
    log-likelihood reporting convention).  The public entry point —
    including ``"Z"`` wildcard model selection — is
    ``pystatistics.timeseries.ets`` in ``_ets_select.py``; this function
    accepts only concrete model letters.

    Parameters
    ----------
    y : ArrayLike
        Time series (1-D, must be positive for multiplicative error/season).
    model : str
        Concrete ETS model string, e.g. ``'ANN'``, ``'AAN'``, ``'AAdA'``,
        ``'MAM'`` (no ``'Z'`` wildcards).
    period : int
        Seasonal period (e.g. 12 for monthly, 4 for quarterly).
    damped : bool or None
        Force damped trend.  Overrides the model string when not ``None``.
    alpha, beta, gamma, phi : float or None
        Fix specific smoothing parameters (skip optimisation for them).
        Fixed values must lie in R's "usual" region — ``[1e-4, 1 - 1e-4]``
        for alpha/beta/gamma with ``beta <= alpha`` and
        ``gamma <= 1 - alpha`` when jointly fixed, ``[0.8, 0.98]`` for
        phi; out-of-range values raise ``ValidationError`` as R errors
        "Parameters out of range" (previous versions silently coerced
        them into bounds).
    tol : float
        Convergence tolerance for the optimiser.
    max_iter : int
        Maximum optimiser iterations.

    Returns
    -------
    ETSSolution
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

    # ---- build parameter vector -------------------------------------------
    # Smoothing params first (logit scale over R's usual region), then the
    # free initial states (unbounded).
    has_trend = spec.trend in ("A", "Ad")
    has_season = spec.season in ("A", "M")

    _check_fixed_smoothing(
        alpha,
        beta if has_trend else None,
        gamma if has_season else None,
        phi if spec.damped else None,
    )
    alpha_box = _alpha_box(
        beta if has_trend else None, gamma if has_season else None
    )

    fixed_smooth: list[float | None] = [alpha]
    if has_trend:
        fixed_smooth.append(beta)
    if has_season:
        fixed_smooth.append(gamma)
    if spec.damped:
        fixed_smooth.append(phi)
    n_smooth = len(fixed_smooth)

    # Free-parameter starting values follow R forecast::ets initparam.
    # Fixed positions keep theta = 0; _decode_smooth ignores them.
    a0 = alpha if alpha is not None else (
        alpha_box[0] + 0.2 * (alpha_box[1] - alpha_box[0]) / spec.period
    )
    theta_smooth0 = np.zeros(n_smooth, dtype=np.float64)
    if alpha is None:
        theta_smooth0[0] = _logit(a0, alpha_box[0], alpha_box[1])
    i = 1
    if has_trend:
        if beta is None:
            b0 = _EPS + 0.1 * (min(1.0 - _EPS, a0) - _EPS)
            theta_smooth0[i] = _logit(b0, _EPS, a0)
        i += 1
    if has_season:
        if gamma is None:
            g0 = _EPS + 0.05 * (min(1.0 - _EPS, 1.0 - a0) - _EPS)
            theta_smooth0[i] = _logit(g0, _EPS, 1.0 - a0)
        i += 1
    if spec.damped and phi is None:
        p0 = _PHI_BOUNDS[0] + 0.99 * (_PHI_BOUNDS[1] - _PHI_BOUNDS[0])
        theta_smooth0[i] = _logit(p0, _PHI_BOUNDS[0], _PHI_BOUNDS[1])

    # Free initial states (unbounded).  Seasonal models optimise m - 1
    # seasonal states; the first-used one is reconstructed from the
    # normalisation by _assemble_init_states (init_s from _init_season is
    # already normalised, so dropping element 0 loses no information).
    init_state_vals: list[float] = [init_l]
    if init_b is not None:
        init_state_vals.append(init_b)
    if init_s is not None:
        init_state_vals.extend(init_s[1:].tolist())

    theta0 = np.concatenate([theta_smooth0, init_state_vals])
    fixed_mask = [v is not None for v in fixed_smooth] + [False] * len(init_state_vals)

    # Optimise only free parameters (initial states are always free, so
    # there is always something to optimise)
    free_idx = [i for i, f in enumerate(fixed_mask) if not f]

    def _objective(theta_free: NDArray) -> float:
        theta_full = theta0.copy()
        theta_full[free_idx] = theta_free
        return _neg_loglik(
            theta_full, y_arr, spec, fixed_smooth, alpha_box, n_smooth
        )

    result = minimize(
        _objective,
        theta0[free_idx],
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )
    converged = result.success

    # Reconstruct full theta and map back to bounded values
    theta_opt = theta0.copy()
    theta_opt[free_idx] = result.x
    params_opt = _decode_smooth(
        theta_opt[:n_smooth], spec, fixed_smooth, alpha_box
    )
    init_states_opt = _assemble_init_states(theta_opt[n_smooth:], spec)

    # ---- final recursion --------------------------------------------------
    fitted_vals, resid, states = ets_recursion(
        y_arr, spec, params_opt, init_states_opt
    )

    # ---- extract named parameters -----------------------------------------
    a_opt, b_opt, g_opt, p_opt = unpack_params(params_opt, spec)

    # ---- information criteria ---------------------------------------------
    floor = _sigma2_floor(y_arr)
    if spec.error == "A":
        sigma2 = float(np.mean(resid ** 2))
        if sigma2 < floor:
            sigma2 = floor
        ll = -0.5 * n * np.log(2.0 * np.pi) - 0.5 * n * np.log(sigma2) - 0.5 * n
    else:
        sigma2 = float(np.mean(resid ** 2))
        if sigma2 < floor:
            sigma2 = floor
        log_abs_fitted = np.log(np.abs(fitted_vals) + 1e-30)
        ll = (
            -0.5 * n * np.log(2.0 * np.pi)
            - 0.5 * n * np.log(sigma2)
            - 0.5 * n
            - float(np.sum(log_abs_fitted))
        )

    # Total parameters: smoothing + free init states + sigma^2 (seasonal
    # models estimate m - 1 initial seasonal states, matching R's count)
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

    return ETSSolution(
        _result=Result(
            params=ETSParams(
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
            ),
            info={"model": spec.name, "converged": converged},
            timing=None,
            backend_name="cpu",
            warnings=(),
        )
    )
