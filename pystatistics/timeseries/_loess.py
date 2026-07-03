"""
R-faithful univariate loess smoother for STL.

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

Everything is evaluated in batched matrix form — all evaluation points of
a series (or of a whole group of equal-length cycle-subseries, see
:func:`loess_subseries_smooth`) go through one vectorised computation —
but each row's arithmetic is element-wise identical to the sequential
formulation, so results match the reference bit-for-bit up to summation
order.

Input contract (validated by the caller, per the STL public API):
series are finite 1-D float64 arrays, ``span`` is an odd integer >= 3,
``degree`` is 0 or 1, ``jump`` is an integer >= 1, and weights (when
given) are non-negative arrays aligned with the series.  These functions
are internal building blocks and do not re-validate.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _estimate_windows(
    yw: NDArray,
    n: int,
    span: int,
    degree: int,
    xs: NDArray,
    nleft: NDArray,
    ww: NDArray | None,
) -> tuple[NDArray, NDArray]:
    """
    Vectorised core: evaluate the local regression for a batch of windows.

    Parameters
    ----------
    yw : NDArray
        Window values, shape ``(m, width)``; row *i* holds the series
        values at positions ``nleft[i] .. nleft[i] + width - 1``.
    n : int
        Length of the underlying series (drives the ``span > n``
        bandwidth widening and the degree-1 spread threshold).
    span : int
        Requested loess span.
    degree : int
        Local polynomial degree, 0 or 1.
    xs : NDArray
        Evaluation positions (1-based, float; may lie outside the
        window), shape ``(m,)``.
    nleft : NDArray
        Left window edges (1-based), shape ``(m,)``.
    ww : NDArray or None
        External weights aligned with *yw* (same shape), or None.

    Returns
    -------
    values : NDArray
        Estimates, shape ``(m,)``; rows with all-zero weight hold 0.0.
    ok : NDArray
        Boolean mask; False where the window carried no weight (the
        caller decides the fallback).
    """
    width = yw.shape[1]
    xs = xs.astype(np.float64)
    # (m, width) matrix of 1-based design positions per row
    pos = nleft[:, None] + np.arange(width, dtype=np.int64)[None, :]
    pos_f = pos.astype(np.float64)

    h = np.maximum(xs - nleft, (nleft + width - 1) - xs)
    if span > n:
        h = h + float((span - n) // 2)

    r = np.abs(pos_f - xs[:, None])
    h_col = h[:, None]
    w = np.zeros_like(r)
    inside = r <= 0.999 * h_col
    unit = r <= 0.001 * h_col
    tri = inside & ~unit
    # Tricube only where 0.001*h < r <= 0.999*h, so h > 0 there: no 0/0.
    np.divide(r, h_col, out=r, where=tri)
    w[tri] = (1.0 - r[tri] ** 3) ** 3
    w[unit] = 1.0
    if ww is not None:
        w *= ww

    total = w.sum(axis=1)
    ok = total > 0.0
    if ok.all():
        w /= total[:, None]
    else:
        w[ok] /= total[ok, None]

    if degree > 0:
        adjustable = ok & (h > 0.0)
        if np.any(adjustable):
            centre = (w * pos_f).sum(axis=1)
            dev = pos_f - centre[:, None]
            spread = (w * dev**2).sum(axis=1)
            # Linear adjustment only when the weighted design spread is
            # non-degenerate relative to the full design range n - 1.
            adjust = adjustable & (np.sqrt(spread) > 0.001 * (n - 1))
            if adjust.all():
                slope = (xs - centre) / spread
                w *= slope[:, None] * dev + 1.0
            elif np.any(adjust):
                slope = (xs[adjust] - centre[adjust]) / spread[adjust]
                w[adjust] *= slope[:, None] * dev[adjust] + 1.0

    return (w * yw).sum(axis=1), ok


def _grid_geometry(
    n: int, span: int, jump: int
) -> tuple[NDArray, NDArray, int, int]:
    """
    Evaluation grid for a full-series smooth.

    Returns ``(xs, nleft, width, nj)``: the 1-based evaluation positions
    ``1, 1+nj, ...`` (plus a trailing ``n`` reusing the previous window
    when the stride does not land on it), their left window edges, the
    common window width, and the effective stride ``nj``.
    """
    nj = min(jump, n - 1)
    xs = np.arange(1, n + 1, nj, dtype=np.int64)
    if span >= n:
        width = n
        nleft = np.ones(len(xs), dtype=np.int64)
    else:
        width = span
        half = (span + 1) // 2
        nleft = np.clip(xs - half + 1, 1, n - span + 1).astype(np.int64)
    if nj != 1 and xs[-1] != n:
        # Trailing estimate at n reuses the last grid window (reference-
        # implementation quirk; see module docstring).
        xs = np.append(xs, n)
        nleft = np.append(nleft, nleft[-1])
    return xs, nleft, width, nj


def _interpolate_rows(out: NDArray, grid: NDArray, nj: int, n: int) -> None:
    """
    Fill non-evaluated positions by exact linear interpolation, in place.

    ``out`` is ``(g, n)`` with the evaluated positions already set;
    ``grid`` is the regular evaluation grid ``1, 1+nj, ...`` (excluding
    any trailing endpoint estimate at *n*, whose segment is handled by
    the ``last != n`` branch).  Each filled element is computed as
    ``v_left + delta * offset`` with ``delta = (v_right - v_left) /
    stride`` — the same two floating-point operations per element as the
    sequential formulation, so results are bit-identical.
    """
    if nj == 1:
        return
    lefts = grid[:-1]
    if len(lefts):
        v_left = out[:, lefts - 1]
        delta = (out[:, lefts - 1 + nj] - v_left) / nj
        for offset in range(1, nj):
            out[:, lefts - 1 + offset] = v_left + delta * offset
    last = int(grid[-1])
    if last != n:
        # Trailing segment between the last grid point and n.
        gap = n - last
        v_last = out[:, last - 1]
        delta = (out[:, n - 1] - v_last) / gap
        for offset in range(1, gap):
            out[:, last - 1 + offset] = v_last + delta * offset


def loess_smooth(
    y: NDArray,
    span: int,
    degree: int,
    jump: int,
    weights: NDArray | None = None,
) -> NDArray:
    """
    Smooth a whole series, evaluating every *jump*-th point.

    Estimates the local regression at positions ``1, 1+jump, ...`` and
    linearly interpolates between them (plus the trailing-endpoint rule
    described in the module docstring).  With ``jump=1`` every point is
    evaluated directly and no interpolation occurs.

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
    n = len(y)
    if n < 2:
        return y.copy()

    xs, nleft, width, nj = _grid_geometry(n, span, jump)
    pos = nleft[:, None] + np.arange(width, dtype=np.int64)[None, :]
    ww = weights[pos - 1] if weights is not None else None
    values, ok = _estimate_windows(
        y[pos - 1], n, span, degree, xs.astype(np.float64), nleft, ww
    )
    # Zero-weight windows fall back to the observed value.
    values = np.where(ok, values, y[xs - 1])

    out = np.empty((1, n), dtype=np.float64)
    out[0, xs - 1] = values
    grid = np.arange(1, n + 1, nj, dtype=np.int64)
    _interpolate_rows(out, grid, nj, n)
    return out[0]


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

    All subseries share the same evaluation geometry, so the whole group
    (every grid point, the trailing endpoint when the stride skips it,
    and the two extension estimates at positions ``0`` and ``k + 1``) is
    evaluated in a single vectorised batch.  The extension windows are
    ``[1, min(span, k)]`` and ``[max(1, k - span + 1), k]``; both have
    width ``min(span, k)``, the same as every grid window, which is what
    makes the single-batch formulation possible.

    Parameters
    ----------
    sub_y : NDArray
        Subseries values, shape ``(g, k)`` with ``k >= 2`` — one row per
        cycle position (guaranteed by STL's ``n > 2 * period`` input
        contract).
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
    g, k = sub_y.shape
    xs, nleft, width, nj = _grid_geometry(k, span, jump)
    n_grid = len(xs)
    # Append the two extension rows; min(span, k) == width in both the
    # span >= k and span < k regimes, so all rows share one width.
    xs_all = np.concatenate([xs.astype(np.float64), [0.0, float(k + 1)]])
    nleft_all = np.concatenate([nleft, [1, max(1, k - span + 1)]]).astype(np.int64)

    pos = nleft_all[:, None] + np.arange(width, dtype=np.int64)[None, :]
    # Row-major layout: for each subseries, all evaluation rows.
    m = len(xs_all)
    sub_idx = np.repeat(np.arange(g), m)
    pos_tiled = np.tile(pos, (g, 1))
    yw = sub_y[sub_idx[:, None], pos_tiled - 1]
    ww = (sub_weights[sub_idx[:, None], pos_tiled - 1]
          if sub_weights is not None else None)

    values, ok = _estimate_windows(
        yw, k, span, degree,
        np.tile(xs_all, g), np.tile(nleft_all, g), ww,
    )
    values = values.reshape(g, m)
    ok = ok.reshape(g, m)

    grid_vals = values[:, :n_grid]
    grid_ok = ok[:, :n_grid]
    if not grid_ok.all():
        fallback = sub_y[:, xs - 1]
        grid_vals = np.where(grid_ok, grid_vals, fallback)

    smoothed = np.empty((g, k), dtype=np.float64)
    smoothed[:, xs - 1] = grid_vals
    grid = np.arange(1, k + 1, nj, dtype=np.int64)
    _interpolate_rows(smoothed, grid, nj, k)

    head = np.where(ok[:, n_grid], values[:, n_grid], smoothed[:, 0])
    tail = np.where(ok[:, n_grid + 1], values[:, n_grid + 1], smoothed[:, -1])
    return smoothed, head, tail
