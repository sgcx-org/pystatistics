"""
R-faithful univariate loess smoother for STL (numba-compiled).

Clean-room implementation of the local-regression smoother used inside
R's ``stats::stl`` (the netlib STL routines described in Cleveland,
Cleveland, McRae & Terpenning (1990), "STL: A Seasonal-Trend
Decomposition Procedure Based on Loess", J. Official Statistics 6(1),
3-73).  The *algorithm* was studied from the paper and from R's sources;
the code here is written fresh (no transliteration of R's GPL Fortran).

Semantics replicated exactly (all positions are 1-based, as in the
algorithm; the neighbourhood is defined on the integer design points
``1..n``):

* **Neighbourhood weights.**  For an estimate at position ``xs`` over the
  window ``[nleft, nright]``, the bandwidth is
  ``h = max(xs - nleft, nright - xs)``, increased by ``(span - n) // 2``
  (integer division) when the requested span exceeds the series length.
  A point at distance ``r`` receives tricube weight
  ``(1 - (r/h)^3)^3``, clamped to 1 for ``r <= 0.001*h`` and to 0 for
  ``r > 0.999*h``.  External (robustness) weights multiply the tricube
  weights.
* **Local degree.**  Degree 0 is a weighted mean.  Degree 1 first
  normalises the weights, then applies the exact linear-adjustment
  ``w_j *= 1 + (xs - a)/c * (j - a)`` (with ``a`` the weighted mean
  position and ``c`` the weighted variance), *skipped* when
  ``sqrt(c) <= 0.001*(n - 1)`` — i.e. the fit falls back to the weighted
  mean when the effective design spread is degenerate.
* **Jump/interpolation.**  ``loess_smooth`` evaluates the local
  regression only at positions ``1, 1+jump, 1+2*jump, ...`` and joins
  them by linear interpolation.  If the last evaluated position is not
  ``n``, one extra estimate is taken at ``n`` — reusing the *window of
  the last regularly evaluated position* (a documented quirk of the
  reference implementation that matters for exact parity).
* **Zero-weight fallback.**  If every weight in a window is zero (possible
  only with external robustness weights), the estimate falls back to the
  observed value at that position.

Each window is evaluated by a scalar loop (``_eval_window``) in the same
left-to-right summation order as the reference algorithm, compiled with
``numba`` at ``fastmath=False`` so IEEE semantics are preserved.  This
tracks R's Fortran to floating-point noise (see ``test_stl_r_parity.py``)
while running at native speed.

Input contract (validated by the caller, per the STL public API):
series are finite 1-D float64 arrays, ``span`` is an odd integer >= 3,
``degree`` is 0 or 1, ``jump`` is an integer >= 1, and weights (when
given) are non-negative arrays aligned with the series.  These functions
are internal building blocks and do not re-validate.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _eval_window(y, w, use_w, n, span, degree, xs, nleft, width, ws):
    """Local-regression estimate at 1-based position ``xs``.

    ``ws`` is a caller-provided scratch buffer of length >= ``width``.
    Returns ``(value, ok)``; ``ok`` is False when the window carried no
    weight (the caller supplies the fallback).
    """
    nright = nleft + width - 1
    h = float(xs - nleft)
    other = float(nright - xs)
    if other > h:
        h = other
    if span > n:
        h += float((span - n) // 2)

    tot = 0.0
    lo = 0.001 * h
    hi = 0.999 * h
    for jj in range(width):
        posj = nleft + jj
        r = posj - xs
        if r < 0.0:
            r = -r
        if r <= hi:
            if r <= lo:
                wj = 1.0
            else:
                rr = r / h
                t = 1.0 - rr * rr * rr
                wj = t * t * t
        else:
            wj = 0.0
        if use_w:
            wj *= w[nleft - 1 + jj]
        ws[jj] = wj
        tot += wj

    ok = tot > 0.0
    if ok:
        for jj in range(width):
            ws[jj] /= tot

    if degree > 0 and h > 0.0 and ok:
        centre = 0.0
        for jj in range(width):
            centre += ws[jj] * (nleft + jj)
        spread = 0.0
        for jj in range(width):
            d = (nleft + jj) - centre
            spread += ws[jj] * d * d
        # Linear adjustment only when the weighted design spread is
        # non-degenerate relative to the full design range n - 1.
        if np.sqrt(spread) > 0.001 * (n - 1):
            slope = (xs - centre) / spread
            for jj in range(width):
                ws[jj] *= slope * ((nleft + jj) - centre) + 1.0

    val = 0.0
    for jj in range(width):
        val += ws[jj] * y[nleft - 1 + jj]
    return val, ok


@njit(cache=True, fastmath=False)
def _grid_width_nleft(n, span, jump):
    """Window geometry: ``(nj, width, span_ge_n)`` for a length-*n* series."""
    nj = jump
    if nj > n - 1:
        nj = n - 1
    if span >= n:
        return nj, n, True
    return nj, span, False


@njit(cache=True, fastmath=False)
def _nleft_for(xs, span, n, span_ge_n):
    """Left window edge (1-based) for evaluation position ``xs``."""
    if span_ge_n:
        return 1
    half = (span + 1) // 2
    v = xs - half + 1
    if v < 1:
        v = 1
    hi = n - span + 1
    if v > hi:
        v = hi
    return v


@njit(cache=True, fastmath=False)
def loess_smooth_nb(y, span, degree, jump, w, use_w):
    """Smooth a whole series, evaluating every ``jump``-th point.

    Evaluates the local regression at positions ``1, 1+jump, ...`` (plus
    the trailing-endpoint rule) and linearly interpolates between them.
    ``w`` is aligned with ``y``; ``use_w`` toggles external weighting.
    """
    n = y.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n < 2:
        for i in range(n):
            out[i] = y[i]
        return out
    nj, width, span_ge_n = _grid_width_nleft(n, span, jump)
    ws = np.empty(width, dtype=np.float64)

    xs = 1
    last = 1
    while xs <= n:
        nleft = _nleft_for(xs, span, n, span_ge_n)
        val, ok = _eval_window(y, w, use_w, n, span, degree,
                               float(xs), nleft, width, ws)
        out[xs - 1] = val if ok else y[xs - 1]
        last = xs
        xs += nj

    # Trailing endpoint at n reuses the last grid window.
    if nj != 1 and last != n:
        nleft = _nleft_for(last, span, n, span_ge_n)
        val, ok = _eval_window(y, w, use_w, n, span, degree,
                               float(n), nleft, width, ws)
        out[n - 1] = val if ok else y[n - 1]

    # Linear interpolation between grid points (v_left + delta*offset, the
    # same two float ops per element as the sequential reference form).
    if nj != 1:
        g = 1
        while g + nj <= n:
            vl = out[g - 1]
            delta = (out[g - 1 + nj] - vl) / nj
            for off in range(1, nj):
                out[g - 1 + off] = vl + delta * off
            g += nj
        if last != n:
            gap = n - last
            vl = out[last - 1]
            delta = (out[n - 1] - vl) / gap
            for off in range(1, gap):
                out[last - 1 + off] = vl + delta * off
    return out


@njit(cache=True, fastmath=False)
def loess_subseries_nb(sub_y, span, degree, jump, sub_w, use_w):
    """Smooth ``g`` equal-length subseries; extend each to positions 0 and
    ``k + 1``.  Returns ``(smoothed (g,k), head (g,), tail (g,))``."""
    g, k = sub_y.shape
    smoothed = np.empty((g, k), dtype=np.float64)
    head = np.empty(g, dtype=np.float64)
    tail = np.empty(g, dtype=np.float64)
    _, width, _ = _grid_width_nleft(k, span, jump)
    ws = np.empty(width, dtype=np.float64)
    yrow = np.empty(k, dtype=np.float64)
    wrow = np.empty(k, dtype=np.float64)

    for row in range(g):
        for j in range(k):
            yrow[j] = sub_y[row, j]
            if use_w:
                wrow[j] = sub_w[row, j]
        out = loess_smooth_nb(yrow, span, degree, jump, wrow, use_w)
        for j in range(k):
            smoothed[row, j] = out[j]
        # Extension windows: [1, width] at position 0, and the mirror at k+1.
        val, ok = _eval_window(yrow, wrow, use_w, k, span, degree,
                               0.0, 1, width, ws)
        head[row] = val if ok else out[0]
        nleft_t = k - span + 1
        if nleft_t < 1:
            nleft_t = 1
        val, ok = _eval_window(yrow, wrow, use_w, k, span, degree,
                               float(k + 1), nleft_t, width, ws)
        tail[row] = val if ok else out[k - 1]
    return smoothed, head, tail


# ---------------------------------------------------------------------------
# Python wrappers (stable public surface over the numba kernels)
# ---------------------------------------------------------------------------

def loess_smooth(
    y: NDArray,
    span: int,
    degree: int,
    jump: int,
    weights: NDArray | None = None,
) -> NDArray:
    """
    Smooth a whole series, evaluating every *jump*-th point.

    Thin wrapper over :func:`loess_smooth_nb`.  With ``jump=1`` every
    point is evaluated directly and no interpolation occurs.

    Parameters
    ----------
    y : NDArray
        Series to smooth, length *n*.
    span : int
        Loess span (odd, >= 3).
    degree : int
        Local polynomial degree, 0 or 1.
    jump : int
        Evaluation stride (>= 1); internally capped at ``n - 1``.
    weights : NDArray or None
        External per-observation weights aligned with *y*.

    Returns
    -------
    NDArray
        Smoothed series, length *n*.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    if weights is None:
        w = np.empty(0, dtype=np.float64)
        use_w = False
    else:
        w = np.ascontiguousarray(weights, dtype=np.float64)
        use_w = True
    return loess_smooth_nb(y, span, degree, jump, w, use_w)


def loess_subseries_smooth(
    sub_y: NDArray,
    span: int,
    degree: int,
    jump: int,
    sub_weights: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Smooth a group of equal-length cycle-subseries and extend each by one
    position at both ends — STL's cycle-subseries kernel.

    Thin wrapper over :func:`loess_subseries_nb`.

    Parameters
    ----------
    sub_y : NDArray
        Subseries values, shape ``(g, k)`` with ``k >= 2``.
    span, degree, jump : int
        Loess parameters (see :func:`loess_smooth`).
    sub_weights : NDArray or None
        External weights, shape ``(g, k)``, or None.

    Returns
    -------
    smoothed : NDArray
        Smoothed subseries, shape ``(g, k)``.
    head : NDArray
        Extension estimates at position 0, shape ``(g,)``.
    tail : NDArray
        Extension estimates at position ``k + 1``, shape ``(g,)``.
    """
    sub_y = np.ascontiguousarray(sub_y, dtype=np.float64)
    if sub_weights is None:
        w = np.empty((1, 1), dtype=np.float64)
        use_w = False
    else:
        w = np.ascontiguousarray(sub_weights, dtype=np.float64)
        use_w = True
    return loess_subseries_nb(sub_y, span, degree, jump, w, use_w)
