# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
"""
Compiled STL / LOESS / robustness kernels (Cython port of the former Numba
``@njit`` kernels in ``_loess``, ``_stl_robust``, ``_stl_core``).

The three modules form one tightly-coupled call graph — the STL driver calls
the loess smoother and the bisquare robustness step inside its inner/outer loop
— so they are compiled as a single translation unit here (internal calls are
``cdef``, no Python boundary). Faithful scalar translations of the pure-Python
reference in ``_stl_ref``; compiled with ``-ffp-contract=off`` (no FMA
contraction, no reassociation) so results are platform-reproducible and match
the reference bit-for-bit (``tests/timeseries/test_stl_cython_parity.py``). The
R-parity tolerance gate (``test_stl_r_parity.py``) is the standing correctness
bar and is unchanged.

The one ``x ** 2`` site in the robustness step is written as an explicit
``x * x`` (both here and in the reference) to avoid libm ``pow``.
"""

import numpy as np
from libc.math cimport sqrt


# ---------------------------------------------------------------------------
# LOESS
# ---------------------------------------------------------------------------

cdef double _eval_window_c(double[::1] y, double[::1] w, bint use_w,
                           Py_ssize_t n, Py_ssize_t span, Py_ssize_t degree,
                           double xs, Py_ssize_t nleft, Py_ssize_t width,
                           double[::1] ws, bint* ok) noexcept nogil:
    """Local-regression estimate at 1-based position ``xs``. Sets ok[0]."""
    cdef Py_ssize_t nright = nleft + width - 1
    cdef double h = <double>(xs - nleft)
    cdef double other = <double>(nright - xs)
    if other > h:
        h = other
    if span > n:
        h += <double>((span - n) // 2)

    cdef double tot = 0.0
    cdef double lo = 0.001 * h
    cdef double hi = 0.999 * h
    cdef Py_ssize_t jj, posj
    cdef double r, rr, t, wj, centre, spread, d, slope, val
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

    ok[0] = tot > 0.0
    if ok[0]:
        for jj in range(width):
            ws[jj] /= tot

    if degree > 0 and h > 0.0 and ok[0]:
        centre = 0.0
        for jj in range(width):
            centre += ws[jj] * (nleft + jj)
        spread = 0.0
        for jj in range(width):
            d = (nleft + jj) - centre
            spread += ws[jj] * d * d
        if sqrt(spread) > 0.001 * (n - 1):
            slope = (xs - centre) / spread
            for jj in range(width):
                ws[jj] *= slope * ((nleft + jj) - centre) + 1.0

    val = 0.0
    for jj in range(width):
        val += ws[jj] * y[nleft - 1 + jj]
    return val


cpdef tuple _eval_window(double[::1] y, double[::1] w, bint use_w,
                         Py_ssize_t n, Py_ssize_t span, Py_ssize_t degree,
                         double xs, Py_ssize_t nleft, Py_ssize_t width,
                         double[::1] ws):
    """Python-facing wrapper (used by tests). Returns (value, ok)."""
    cdef bint ok
    cdef double val = _eval_window_c(y, w, use_w, n, span, degree, xs,
                                     nleft, width, ws, &ok)
    return val, ok


cdef inline void _grid_width_nleft(Py_ssize_t n, Py_ssize_t span,
                                   Py_ssize_t jump, Py_ssize_t* nj,
                                   Py_ssize_t* width, bint* span_ge_n) noexcept nogil:
    nj[0] = jump
    if nj[0] > n - 1:
        nj[0] = n - 1
    if span >= n:
        width[0] = n
        span_ge_n[0] = True
    else:
        width[0] = span
        span_ge_n[0] = False


cdef inline Py_ssize_t _nleft_for(Py_ssize_t xs, Py_ssize_t span, Py_ssize_t n,
                                  bint span_ge_n) noexcept nogil:
    if span_ge_n:
        return 1
    cdef Py_ssize_t half = (span + 1) // 2
    cdef Py_ssize_t v = xs - half + 1
    if v < 1:
        v = 1
    cdef Py_ssize_t hi = n - span + 1
    if v > hi:
        v = hi
    return v


cpdef loess_smooth_nb(double[::1] y, Py_ssize_t span, Py_ssize_t degree,
                      Py_ssize_t jump, double[::1] w, bint use_w):
    """Smooth a whole series, evaluating every ``jump``-th point."""
    cdef Py_ssize_t n = y.shape[0]
    cdef double[::1] out = np.empty(n, dtype=np.float64)
    cdef Py_ssize_t i
    if n < 2:
        for i in range(n):
            out[i] = y[i]
        return np.asarray(out)

    cdef Py_ssize_t nj, width
    cdef bint span_ge_n
    _grid_width_nleft(n, span, jump, &nj, &width, &span_ge_n)
    cdef double[::1] ws = np.empty(width, dtype=np.float64)

    cdef Py_ssize_t xs = 1, last = 1, nleft, g, gap, off
    cdef bint ok
    cdef double val, vl, delta
    while xs <= n:
        nleft = _nleft_for(xs, span, n, span_ge_n)
        val = _eval_window_c(y, w, use_w, n, span, degree,
                             <double>xs, nleft, width, ws, &ok)
        out[xs - 1] = val if ok else y[xs - 1]
        last = xs
        xs += nj

    if nj != 1 and last != n:
        nleft = _nleft_for(last, span, n, span_ge_n)
        val = _eval_window_c(y, w, use_w, n, span, degree,
                             <double>n, nleft, width, ws, &ok)
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
    return np.asarray(out)


cpdef loess_subseries_nb(double[:, ::1] sub_y, Py_ssize_t span,
                         Py_ssize_t degree, Py_ssize_t jump,
                         double[:, ::1] sub_w, bint use_w):
    """Smooth ``g`` equal-length subseries; extend each to positions 0/k+1."""
    cdef Py_ssize_t g = sub_y.shape[0]
    cdef Py_ssize_t k = sub_y.shape[1]
    cdef double[:, ::1] smoothed = np.empty((g, k), dtype=np.float64)
    cdef double[::1] head = np.empty(g, dtype=np.float64)
    cdef double[::1] tail = np.empty(g, dtype=np.float64)
    cdef Py_ssize_t nj, width
    cdef bint span_ge_n
    _grid_width_nleft(k, span, jump, &nj, &width, &span_ge_n)
    cdef double[::1] ws = np.empty(width, dtype=np.float64)
    cdef double[::1] yrow = np.empty(k, dtype=np.float64)
    cdef double[::1] wrow = np.empty(k, dtype=np.float64)

    cdef Py_ssize_t row, j, nleft_t
    cdef bint ok
    cdef double val
    cdef double[::1] outv
    for row in range(g):
        for j in range(k):
            yrow[j] = sub_y[row, j]
            if use_w:
                wrow[j] = sub_w[row, j]
        outv = loess_smooth_nb(yrow, span, degree, jump, wrow, use_w)
        for j in range(k):
            smoothed[row, j] = outv[j]
        val = _eval_window_c(yrow, wrow, use_w, k, span, degree,
                             0.0, 1, width, ws, &ok)
        head[row] = val if ok else outv[0]
        nleft_t = k - span + 1
        if nleft_t < 1:
            nleft_t = 1
        val = _eval_window_c(yrow, wrow, use_w, k, span, degree,
                             <double>(k + 1), nleft_t, width, ws, &ok)
        tail[row] = val if ok else outv[k - 1]
    return np.asarray(smoothed), np.asarray(head), np.asarray(tail)


# ---------------------------------------------------------------------------
# Robustness (bisquare) + reference-exact partial sort
# ---------------------------------------------------------------------------

cdef void _psort_pair_c(double* a, Py_ssize_t n, Py_ssize_t first,
                        Py_ssize_t second) noexcept nogil:
    """Reference-exact partial sort; mutates ``a`` in place (see _stl_ref)."""
    cdef Py_ssize_t ind0 = first
    cdef Py_ssize_t ind1 = second
    if n < 2:
        return
    cdef Py_ssize_t seg_i[128]
    cdef Py_ssize_t seg_j[128]
    cdef Py_ssize_t seg_jl[128]
    cdef Py_ssize_t seg_ju[128]
    seg_i[0] = 0
    seg_j[0] = 0
    seg_jl[0] = 1
    seg_ju[0] = 2
    cdef Py_ssize_t jl = 1, ju = 2, i = 1, j = n, depth = 1
    cdef Py_ssize_t state = 0
    cdef Py_ssize_t k, mid_pos, low, pushed, pos
    cdef double pivot, swap, val
    cdef bint bail
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


cpdef tuple psort_pair_nb(double[::1] a, Py_ssize_t first, Py_ssize_t second):
    """Partial sort a in place; returns (a[first-1], a[second-1])."""
    cdef Py_ssize_t n = a.shape[0]
    if n < 2:
        return a[0], a[0]
    _psort_pair_c(&a[0], n, first, second)
    return a[first - 1], a[second - 1]


cpdef robustness_weights_nb(double[::1] y, double[::1] fit):
    """Bisquare robustness weights on the remainder y - fit."""
    cdef Py_ssize_t n = y.shape[0]
    cdef double[::1] r = np.empty(n, dtype=np.float64)
    cdef Py_ssize_t i
    cdef double d
    for i in range(n):
        d = y[i] - fit[i]
        r[i] = d if d >= 0.0 else -d
    cdef double[::1] work = np.empty(n, dtype=np.float64)
    for i in range(n):
        work[i] = r[i]
    _psort_pair_c(&work[0], n, n // 2 + 1, n - n // 2)
    cdef double lo = work[n // 2 + 1 - 1]
    cdef double hi = work[n - n // 2 - 1]
    cdef double cmad = 3.0 * (lo + hi)
    cdef double[::1] w = np.zeros(n, dtype=np.float64)
    cdef double c1, c9, ri, tmp, t
    if cmad <= 0.0:
        for i in range(n):
            if r[i] <= 0.0:
                w[i] = 1.0
        return np.asarray(w)
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
    return np.asarray(w)


# ---------------------------------------------------------------------------
# STL driver
# ---------------------------------------------------------------------------

cpdef moving_average_nb(double[::1] x, Py_ssize_t width):
    """Running mean of ``width`` consecutive values (sequential update)."""
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t m = n - width + 1
    cdef double[::1] out = np.empty(m, dtype=np.float64)
    cdef double v = 0.0
    cdef Py_ssize_t i, jx
    for i in range(width):
        v += x[i]
    out[0] = v / width
    for jx in range(1, m):
        v = v - x[jx - 1] + x[jx + width - 1]
        out[jx] = v / width
    return np.asarray(out)


cdef void _cycle_subseries_into(double[::1] detrended, Py_ssize_t period,
                                Py_ssize_t span, Py_ssize_t degree,
                                Py_ssize_t jump, double[::1] rob_w, bint use_w,
                                double[::1] ext, double[::1] yrow,
                                double[::1] wrow):
    """Smooth each cycle-subseries and write the one-period-extended result
    into ``ext`` (length n + 2*period), interleaved by cycle position."""
    cdef Py_ssize_t n = detrended.shape[0]
    cdef Py_ssize_t pos, k, t, nleft_t, idx, tt, nj, width
    cdef bint span_ge_n, ok
    cdef double head, tail, val
    cdef double[::1] smoothed
    cdef double[::1] ws
    for pos in range(period):
        k = 0
        t = pos
        while t < n:
            yrow[k] = detrended[t]
            if use_w:
                wrow[k] = rob_w[t]
            k += 1
            t += period
        smoothed = loess_smooth_nb(yrow[:k], span, degree, jump, wrow[:k], use_w)
        _grid_width_nleft(k, span, jump, &nj, &width, &span_ge_n)
        ws = np.empty(width, dtype=np.float64)
        val = _eval_window_c(yrow[:k], wrow[:k], use_w, k, span, degree,
                             0.0, 1, width, ws, &ok)
        head = val if ok else smoothed[0]
        nleft_t = k - span + 1
        if nleft_t < 1:
            nleft_t = 1
        val = _eval_window_c(yrow[:k], wrow[:k], use_w, k, span, degree,
                             <double>(k + 1), nleft_t, width, ws, &ok)
        tail = val if ok else smoothed[k - 1]
        idx = pos
        ext[idx] = head
        idx += period
        for tt in range(k):
            ext[idx] = smoothed[tt]
            idx += period
        ext[idx] = tail


cpdef stl_core_nb(double[::1] y, Py_ssize_t period, Py_ssize_t s_win,
                  Py_ssize_t s_deg, Py_ssize_t s_jump, Py_ssize_t t_win,
                  Py_ssize_t t_deg, Py_ssize_t t_jump, Py_ssize_t l_win,
                  Py_ssize_t l_deg, Py_ssize_t l_jump, Py_ssize_t n_inner,
                  Py_ssize_t n_outer, bint periodic):
    """Run the full STL inner/outer loop. Returns (seasonal, trend, weights)."""
    cdef Py_ssize_t n = y.shape[0]
    cdef double[::1] trend = np.zeros(n, dtype=np.float64)
    cdef double[::1] seasonal = np.zeros(n, dtype=np.float64)
    cdef double[::1] rob_w = np.ones(n, dtype=np.float64)
    cdef bint use_w = False
    cdef double[::1] ext = np.empty(n + 2 * period, dtype=np.float64)
    cdef double[::1] detr = np.empty(n, dtype=np.float64)
    cdef double[::1] deseas = np.empty(n, dtype=np.float64)
    cdef double[::1] yrow = np.empty(n, dtype=np.float64)
    cdef double[::1] wrow = np.empty(n, dtype=np.float64)
    cdef double[::1] ma1, ma2, ma3, low, fit
    cdef Py_ssize_t iteration, inner, i, pos, t, cnt
    cdef double acc, mean

    for iteration in range(n_outer + 1):
        for inner in range(n_inner):
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
            trend = loess_smooth_nb(deseas, t_win, t_deg, t_jump, rob_w, use_w)
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
    return np.asarray(seasonal), np.asarray(trend), np.asarray(rob_w)
