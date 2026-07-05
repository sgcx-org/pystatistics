"""
Numba-compiled ETS state-space forward recursion (the hot inner loop).

The ETS one-step recursion — one-step prediction, one-step error, and the
level/trend/season state update — is inherently SEQUENTIAL (``state[t]``
depends on ``state[t-1]``), so numpy vectorisation cannot help; the
maximum-likelihood optimiser calls it once per objective evaluation, many
times per fit.  This module compiles that loop with ``numba`` at
``fastmath=False``, the same house pattern as the SARIMA Kalman filter
(:mod:`._arima_kalman`) and the STL loop (:mod:`._stl_core`).

``fastmath=False`` is load-bearing: it forbids floating-point
reassociation, so the compiled kernel reproduces the pure-numpy reference
recursion (:func:`._ets_models._ets_recursion_reference`) BIT-FOR-BIT in
fp64.  The reference remains the blessed definition; this kernel is an
implementation swap that only changes speed.  Bit-identity is enforced by
``tests/timeseries/test_ets_kernel_parity.py``.

The model type — error (A/M), trend (N/A/Ad) and season (N/A/M) — is
passed in as integer/boolean flags rather than specialised per family, so
one kernel covers every ETS(error, trend, season) combination.  Damping is
carried entirely by ``phi_val`` (1.0 when the trend is undamped), exactly
as in the reference: the A-vs-Ad distinction never appears in the
arithmetic beyond that factor.

Flag encoding (matching the reference branch structure):

    mult_error : bool   True for multiplicative error ('M'), else additive.
    has_trend  : bool   True for trend 'A' or 'Ad', False for 'N'.
    season_code: int    0 = 'N' (none), 1 = 'A' (additive), 2 = 'M' (mult).

References
----------
Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008).
    Forecasting with Exponential Smoothing: The State Space Approach.
    Springer.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["ets_recursion_nb"]


@njit(cache=True, fastmath=False)
def ets_recursion_nb(
    y,            # observed series, shape (n,)
    alpha,        # level smoothing
    beta,         # trend smoothing (0.0 when has_trend is False; unused there)
    gamma,        # season smoothing (0.0 when season_code == 0; unused there)
    phi_val,      # damping factor (1.0 when undamped)
    mult_error,   # bool: multiplicative error
    has_trend,    # bool: trend present
    season_code,  # int: 0 = none, 1 = additive, 2 = multiplicative
    m,            # seasonal period (1 when non-seasonal)
    l0,           # initial level
    b0,           # initial trend (0.0 when no trend)
    s0,           # initial seasonal ring buffer, shape (m,)
):
    """Run the ETS forward recursion (numba-JIT).

    Structural twin of :func:`._arima_kalman._kalman_loop`: numpy arrays and
    scalar flags in, ``(fitted, residuals, states)`` out.  The seasonal
    states are held in a length-``m`` ring buffer indexed by ``t % m``.  The
    branch tree (prediction, error, six-way state update, state storage)
    mirrors the pure-numpy reference line-for-line so the two agree to the
    last bit under ``fastmath=False``.

    Returns
    -------
    fitted : shape (n,)      one-step-ahead fitted values (mu_t).
    residuals : shape (n,)   additive: ``y - fitted``; multiplicative:
                             ``y / fitted - 1`` (with a small-mu guard).
    states : shape (n + 1, n_cols)  state history; row 0 is the initial
                             state.  ``n_cols`` = 1 (level) + 1 (trend, if
                             any) + m (season, if any).
    """
    n = y.shape[0]

    # Local seasonal ring buffer (copied so the caller's array is untouched).
    s = s0.copy()

    l_prev = l0
    b_prev = b0

    n_cols = 1
    if has_trend:
        n_cols += 1
    if season_code != 0:
        n_cols += m

    states = np.empty((n + 1, n_cols), dtype=np.float64)
    fitted = np.empty(n, dtype=np.float64)
    residuals = np.empty(n, dtype=np.float64)

    # Store initial state row (row 0).
    states[0, 0] = l_prev
    col = 1
    if has_trend:
        states[0, 1] = b_prev
        col = 2
    if season_code != 0:
        for j in range(m):
            states[0, col + j] = s[j]

    for t in range(n):
        s_idx = t % m  # index into seasonal ring buffer

        # --- one-step-ahead prediction (mu_t) ---
        if season_code == 0:
            mu = l_prev if not has_trend else l_prev + phi_val * b_prev
        elif season_code == 1:
            mu = (
                l_prev + s[s_idx] if not has_trend
                else l_prev + phi_val * b_prev + s[s_idx]
            )
        else:  # season_code == 2 (multiplicative)
            mu = (
                l_prev * s[s_idx] if not has_trend
                else (l_prev + phi_val * b_prev) * s[s_idx]
            )

        fitted[t] = mu

        # --- error ---
        if mult_error:
            e = (y[t] - mu) if abs(mu) < 1e-15 else (y[t] / mu) - 1.0
            residuals[t] = e
        else:
            e = y[t] - mu
            residuals[t] = e

        # --- state update ---
        s_old = s[s_idx]
        l_old = l_prev
        b_old = b_prev

        if season_code == 0 and not has_trend:
            # ETS(.,N,N)
            l_prev = l_old * (1.0 + alpha * e) if mult_error else l_old + alpha * e

        elif season_code == 0 and has_trend:
            # ETS(.,A,N) or ETS(.,Ad,N)
            if mult_error:
                l_prev = (l_old + phi_val * b_old) * (1.0 + alpha * e)
                b_prev = phi_val * b_old + beta * (l_old + phi_val * b_old) * e
            else:
                l_prev = l_old + phi_val * b_old + alpha * e
                b_prev = phi_val * b_old + beta * e

        elif season_code == 1 and not has_trend:
            # ETS(.,N,A)
            if mult_error:
                l_prev = l_old + alpha * (l_old + s_old) * e
                s[s_idx] = s_old + gamma * (l_old + s_old) * e
            else:
                l_prev = l_old + alpha * e
                s[s_idx] = s_old + gamma * e

        elif season_code == 1 and has_trend:
            # ETS(.,A,A) or ETS(.,Ad,A)
            if mult_error:
                mu_val = l_old + phi_val * b_old + s_old
                l_prev = l_old + phi_val * b_old + alpha * mu_val * e
                b_prev = phi_val * b_old + beta * mu_val * e
                s[s_idx] = s_old + gamma * mu_val * e
            else:
                l_prev = l_old + phi_val * b_old + alpha * e
                b_prev = phi_val * b_old + beta * e
                s[s_idx] = s_old + gamma * e

        elif season_code == 2 and not has_trend:
            # ETS(.,N,M)
            if mult_error:
                l_prev = l_old * (1.0 + alpha * e)
                s[s_idx] = s_old * (1.0 + gamma * e)
            else:
                denom_s = s_old if abs(s_old) > 1e-15 else 1e-15
                denom_l = l_old if abs(l_old) > 1e-15 else 1e-15
                l_prev = l_old + alpha * e / denom_s
                s[s_idx] = s_old + gamma * e / denom_l

        else:  # season_code == 2 and has_trend
            # ETS(.,A,M) or ETS(.,Ad,M)
            base = l_old + phi_val * b_old
            if mult_error:
                l_prev = base * (1.0 + alpha * e)
                b_prev = phi_val * b_old + beta * base * e
                s[s_idx] = s_old * (1.0 + gamma * e)
            else:
                denom_s = s_old if abs(s_old) > 1e-15 else 1e-15
                denom_l = base if abs(base) > 1e-15 else 1e-15
                l_prev = base + alpha * e / denom_s
                b_prev = phi_val * b_old + beta * e / denom_s
                s[s_idx] = s_old + gamma * e / denom_l

        # --- store state row (t + 1) ---
        states[t + 1, 0] = l_prev
        col = 1
        if has_trend:
            states[t + 1, 1] = b_prev
            col = 2
        if season_code != 0:
            for j in range(m):
                states[t + 1, col + j] = s[j]

    return fitted, residuals, states
