"""
Fused STL inner/outer loop, matching R ``stats::stl`` (numba-compiled).

The whole seasonal-trend iteration — detrend, cycle-subseries loess with
one-period end extension, low-pass moving-average cascade + loess,
deseasonalise, trend loess, and the bisquare robustness outer loop — is
compiled into a single ``numba`` driver (:func:`stl_core_nb`), mirroring
R's single Fortran routine.  Running the entire loop in one compiled pass
(rather than orchestrating per-step numpy calls from Python) is what
matches R's speed on robust decompositions, where the outer loop repeats
the inner passes up to fifteen times.

Clean-room from the algorithm documented in :mod:`._stl` / :mod:`._loess`
/ :mod:`._stl_robust` (Cleveland, Cleveland, McRae & Terpenning, 1990) —
no transliteration of R's GPL Fortran.  All arithmetic is done in the
same order as the reference, at ``fastmath=False``, so components match R
to floating-point noise (acceptance gate: ``test_stl_r_parity.py``).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from pystatistics.timeseries._loess import (
    _eval_window,
    _grid_width_nleft,
    loess_smooth_nb,
)
from pystatistics.timeseries._stl_robust import robustness_weights_nb


@njit(cache=True, fastmath=False)
def moving_average_nb(x, width):
    """Running mean of *width* consecutive values, length ``len(x)-width+1``.

    Uses the reference's sequential running-sum update order
    ``(v - x[out]) + x[in]`` so the floating-point result matches R
    exactly (a vectorised cumulative sum would round differently)."""
    n = x.shape[0]
    m = n - width + 1
    out = np.empty(m, dtype=np.float64)
    v = 0.0
    for i in range(width):
        v += x[i]
    out[0] = v / width
    for j in range(1, m):
        v = v - x[j - 1] + x[j + width - 1]
        out[j] = v / width
    return out


@njit(cache=True, fastmath=False)
def _cycle_subseries_into(detrended, period, span, degree, jump,
                          rob_w, use_w, ext, yrow, wrow):
    """Smooth every cycle-subseries and write the one-period-extended
    result into ``ext`` (length ``n + 2*period``), interleaved by cycle
    position.  ``yrow``/``wrow`` are caller-provided length-*n* scratch
    buffers."""
    n = detrended.shape[0]
    for pos in range(period):
        # Gather subseries detrended[pos::period] into yrow[:k].
        k = 0
        t = pos
        while t < n:
            yrow[k] = detrended[t]
            if use_w:
                wrow[k] = rob_w[t]
            k += 1
            t += period
        smoothed = loess_smooth_nb(yrow[:k], span, degree, jump,
                                   wrow[:k], use_w)
        _, width, _ = _grid_width_nleft(k, span, jump)
        ws = np.empty(width, dtype=np.float64)
        # Extension estimates at positions 0 and k+1.
        val, ok = _eval_window(yrow[:k], wrow[:k], use_w, k, span, degree,
                               0.0, 1, width, ws)
        head = val if ok else smoothed[0]
        nleft_t = k - span + 1
        if nleft_t < 1:
            nleft_t = 1
        val, ok = _eval_window(yrow[:k], wrow[:k], use_w, k, span, degree,
                               float(k + 1), nleft_t, width, ws)
        tail = val if ok else smoothed[k - 1]
        # Write ext[pos::period] = [head, smoothed..., tail].
        idx = pos
        ext[idx] = head
        idx += period
        for tt in range(k):
            ext[idx] = smoothed[tt]
            idx += period
        ext[idx] = tail


@njit(cache=True, fastmath=False)
def stl_core_nb(y, period, s_win, s_deg, s_jump, t_win, t_deg, t_jump,
                l_win, l_deg, l_jump, n_inner, n_outer, periodic):
    """
    Run the full STL inner/outer loop.

    Returns ``(seasonal, trend, weights)``.  On the first outer iteration
    the seasonal/trend loess are unweighted; subsequent iterations weight
    them by the bisquare robustness weights of the previous fit, exactly
    as the reference algorithm does.  When ``periodic`` is set the final
    seasonal is replaced by its cycle-position means.
    """
    n = y.shape[0]
    trend = np.zeros(n, dtype=np.float64)
    seasonal = np.zeros(n, dtype=np.float64)
    rob_w = np.ones(n, dtype=np.float64)
    use_w = False
    ext = np.empty(n + 2 * period, dtype=np.float64)
    detr = np.empty(n, dtype=np.float64)
    deseas = np.empty(n, dtype=np.float64)
    yrow = np.empty(n, dtype=np.float64)
    wrow = np.empty(n, dtype=np.float64)

    for iteration in range(n_outer + 1):
        for _ in range(n_inner):
            for i in range(n):
                detr[i] = y[i] - trend[i]
            _cycle_subseries_into(detr, period, s_win, s_deg, s_jump,
                                  rob_w, use_w, ext, yrow, wrow)
            # Low-pass: MA(period), MA(period), MA(3), then loess.  The
            # low-pass loess is never robustness-weighted (reference rule).
            ma1 = moving_average_nb(ext, period)
            ma2 = moving_average_nb(ma1, period)
            ma3 = moving_average_nb(ma2, 3)
            low = loess_smooth_nb(ma3, l_win, l_deg, l_jump, wrow, False)
            for i in range(n):
                seasonal[i] = ext[period + i] - low[i]
                deseas[i] = y[i] - seasonal[i]
            trend = loess_smooth_nb(deseas, t_win, t_deg, t_jump,
                                    rob_w, use_w)
        if iteration == n_outer:
            break
        # Robustness weights from y - (trend + seasonal) feed next round.
        fit = np.empty(n, dtype=np.float64)
        for i in range(n):
            fit[i] = trend[i] + seasonal[i]
        rob_w = robustness_weights_nb(y, fit)
        use_w = True

    if periodic:
        for pos in range(period):
            cnt = 0
            acc = 0.0
            t = pos
            while t < n:
                acc += seasonal[t]
                cnt += 1
                t += period
            mean = acc / cnt
            t = pos
            while t < n:
                seasonal[t] = mean
                t += period
    return seasonal, trend, rob_w
