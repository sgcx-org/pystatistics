"""Pure-Python/numpy reference for the STL / LOESS / robustness kernels.

Bit-identity oracle for the compiled kernels in ``_stl_kernels``. These are the
**verbatim** original Numba kernel bodies (loess, robustness, STL driver) with
``njit`` neutered to a no-op, so they run as plain Python and are, by
construction, the exact reference the compiled kernels are checked against
(``tests/timeseries/test_stl_cython_parity.py``).

One deliberate deviation from verbatim, marked inline: the single ``(x) ** 2``
in ``robustness_weights_nb`` is written as an explicit ``x * x`` here (and in
the compiled kernel) so both avoid libm ``pow`` — deterministic across
platforms, and ≤1 ULP under the R-parity tolerance gate.

Test-only; never imported on the hot path. Do not "optimise".
"""

from __future__ import annotations

import numpy as np


def njit(*args, **kwargs):  # no-op: run the kernel bodies as plain Python
    if args and callable(args[0]):
        return args[0]
    def deco(f):
        return f
    return deco


# --- loess (from _loess.py) ------------------------------------------------

@njit(cache=True, fastmath=False)
def _eval_window(y, w, use_w, n, span, degree, xs, nleft, width, ws):
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
    nj = jump
    if nj > n - 1:
        nj = n - 1
    if span >= n:
        return nj, n, True
    return nj, span, False


@njit(cache=True, fastmath=False)
def _nleft_for(xs, span, n, span_ge_n):
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

    if nj != 1 and last != n:
        nleft = _nleft_for(last, span, n, span_ge_n)
        val, ok = _eval_window(y, w, use_w, n, span, degree,
                               float(n), nleft, width, ws)
        out[n - 1] = val if ok else y[n - 1]

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


# --- robustness (from _stl_robust.py) --------------------------------------

@njit(cache=True, fastmath=False)
def psort_pair_nb(a, first, second):
    n = a.shape[0]
    ind0 = first
    ind1 = second
    if n < 2:
        return a[0], a[0]
    MAXD = 128
    seg_i = np.empty(MAXD, dtype=np.int64)
    seg_j = np.empty(MAXD, dtype=np.int64)
    seg_jl = np.empty(MAXD, dtype=np.int64)
    seg_ju = np.empty(MAXD, dtype=np.int64)
    seg_i[0] = 0
    seg_j[0] = 0
    seg_jl[0] = 1
    seg_ju[0] = 2
    jl = 1
    ju = 2
    i = 1
    j = n
    depth = 1
    state = 0
    while True:
        if state == 0:
            state = 3 if i < j else 1
        elif state == 1:
            depth -= 1
            if depth == 0:
                break
            i = seg_i[depth - 1]
            j = seg_j[depth - 1]
            jl = seg_jl[depth - 1]
            ju = seg_ju[depth - 1]
            if jl <= ju:
                state = 2
        elif state == 2:
            state = 3 if j - i > 10 else 4
        elif state == 3:
            k = i
            mid_pos = (i + j) // 2
            pivot = a[mid_pos - 1]
            if a[i - 1] > pivot:
                a[mid_pos - 1] = a[i - 1]
                a[i - 1] = pivot
                pivot = a[mid_pos - 1]
            low = j
            if a[j - 1] < pivot:
                a[mid_pos - 1] = a[j - 1]
                a[j - 1] = pivot
                pivot = a[mid_pos - 1]
                if a[i - 1] > pivot:
                    a[mid_pos - 1] = a[i - 1]
                    a[i - 1] = pivot
                    pivot = a[mid_pos - 1]
            while True:
                low -= 1
                if a[low - 1] <= pivot:
                    swap = a[low - 1]
                    while True:
                        k += 1
                        if a[k - 1] >= pivot:
                            break
                    if k > low:
                        break
                    a[low - 1] = a[k - 1]
                    a[k - 1] = swap
            seg_jl[depth - 1] = jl
            seg_ju[depth - 1] = ju
            pushed = depth
            depth += 1
            if low - i <= j - k:
                seg_i[pushed - 1] = k
                seg_j[pushed - 1] = j
                j = low
                bail = False
                while True:
                    if jl > ju:
                        bail = True
                        break
                    if (ind1 if ju == 2 else ind0) > j:
                        ju -= 1
                    else:
                        break
                if bail:
                    state = 1
                    continue
                seg_jl[pushed - 1] = ju + 1
            else:
                seg_i[pushed - 1] = i
                seg_j[pushed - 1] = low
                i = k
                bail = False
                while True:
                    if jl > ju:
                        bail = True
                        break
                    if (ind0 if jl == 1 else ind1) < i:
                        jl += 1
                    else:
                        break
                if bail:
                    state = 1
                    continue
                seg_ju[pushed - 1] = jl - 1
            state = 2
        else:  # state == 4
            if i != 1:
                pos = i
                while True:
                    if pos == j:
                        state = 1
                        break
                    val = a[pos]
                    if a[pos - 1] > val:
                        k = pos
                        while True:
                            a[k] = a[k - 1]
                            k -= 1
                            if val >= a[k - 1]:
                                break
                        a[k] = val
                    pos += 1
            else:
                state = 0
    return a[first - 1], a[second - 1]


@njit(cache=True, fastmath=False)
def robustness_weights_nb(y, fit):
    n = y.shape[0]
    r = np.empty(n, dtype=np.float64)
    for i in range(n):
        d = y[i] - fit[i]
        r[i] = d if d >= 0.0 else -d
    work = r.copy()
    lo, hi = psort_pair_nb(work, n // 2 + 1, n - n // 2)
    cmad = 3.0 * (lo + hi)
    w = np.zeros(n, dtype=np.float64)
    if cmad <= 0.0:
        for i in range(n):
            if r[i] <= 0.0:
                w[i] = 1.0
        return w
    c1 = 0.001 * cmad
    c9 = 0.999 * cmad
    for i in range(n):
        ri = r[i]
        if ri <= c1:
            w[i] = 1.0
        elif ri <= c9:
            tmp = ri / cmad          # was (ri/cmad) ** 2; explicit mul, no pow
            t = 1.0 - tmp * tmp
            w[i] = t * t
    return w


# --- STL driver (from _stl_core.py) ----------------------------------------

@njit(cache=True, fastmath=False)
def moving_average_nb(x, width):
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
    n = detrended.shape[0]
    for pos in range(period):
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
        val, ok = _eval_window(yrow[:k], wrow[:k], use_w, k, span, degree,
                               0.0, 1, width, ws)
        head = val if ok else smoothed[0]
        nleft_t = k - span + 1
        if nleft_t < 1:
            nleft_t = 1
        val, ok = _eval_window(yrow[:k], wrow[:k], use_w, k, span, degree,
                               float(k + 1), nleft_t, width, ws)
        tail = val if ok else smoothed[k - 1]
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
