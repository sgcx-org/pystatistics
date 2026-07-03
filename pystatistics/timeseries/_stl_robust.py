"""
STL robustness weighting, matching R ``stats::stl`` bit-for-bit.

The outer STL loop downweights outlying observations with Tukey's
bisquare applied to the remainder, scaled by six times the median
absolute residual.  The reference implementation selects its two
"middle" order statistics through a partial quicksort whose behaviour
for even-length inputs deviates from a true partial sort (see
:func:`_psort_pair`); R inherits that behaviour, so exact parity
requires replicating the algorithm rather than substituting a correct
median.  Clean-room implementation from the algorithm (no
transliteration of R's GPL Fortran).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _psort_pair(a: NDArray, first: int, second: int) -> tuple[float, float]:
    """
    Reference-exact partial sort returning ``(a[first], a[second])``
    (1-based positions) as the STL robustness step selects them.

    The reference implementation picks its two "middle" order statistics
    with a partial quicksort (netlib ``psort``) whose requested positions
    arrive in the order ``(n//2 + 1, n - n//2)`` — *descending* whenever
    *n* is even.  psort's segment bookkeeping assumes ascending requests,
    so on some inputs one of the two returned positions holds an element
    that is **not** the true order statistic (empirically roughly 10% of
    random even-length vectors, growing with length; absent for n <= 4).  R inherits this behaviour, so exact parity
    requires reproducing the algorithm — median-of-three quicksort with
    insertion sort below length 11 and the exact stat-range pruning —
    rather than substituting a correct partial sort.  Reimplemented
    clean-room from the algorithm; positions handled 1-based as in the
    algorithm description, with 0-based array access.
    """
    n = len(a)
    ind = (first, second)
    if n < 2:
        return float(a[0]), float(a[0])
    # Segment stack entries: (i, j) bounds plus (jl, ju) request-range.
    seg_i: list[int] = [0]
    seg_j: list[int] = [0]
    seg_jl: list[int] = [1]
    seg_ju: list[int] = [2]
    jl, ju = 1, 2
    i, j, depth = 1, n, 1

    state = "top"
    while True:
        if state == "top":
            state = "partition" if i < j else "pop"
        elif state == "pop":
            depth -= 1
            if depth == 0:
                break
            i, j = seg_i[depth - 1], seg_j[depth - 1]
            jl, ju = seg_jl[depth - 1], seg_ju[depth - 1]
            if jl <= ju:
                state = "size_check"
        elif state == "size_check":
            state = "partition" if j - i > 10 else "small"
        elif state == "partition":
            # Median-of-three pivot on a[i], a[(i+j)//2], a[j].
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
            # Push the larger side; keep the request-range bookkeeping
            # exactly as the reference does (including its descending-
            # request quirk).
            if len(seg_i) < depth + 1:
                seg_i.append(0); seg_j.append(0)
                seg_jl.append(0); seg_ju.append(0)
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
                    if ind[ju - 1] > j:
                        ju -= 1
                    else:
                        break
                if bail:
                    state = "pop"
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
                    if ind[jl - 1] < i:
                        jl += 1
                    else:
                        break
                if bail:
                    state = "pop"
                    continue
                seg_ju[pushed - 1] = jl - 1
            state = "size_check"
        else:  # "small": insertion sort, needs the a[i-1] sentinel (i > 1)
            if i != 1:
                pos = i
                while True:
                    if pos == j:
                        state = "pop"
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
                state = "top"
    return float(a[first - 1]), float(a[second - 1])


def _robustness_weights(y: NDArray, fit: NDArray) -> NDArray:
    """
    Bisquare robustness weights on the remainder ``y - fit``.

    ``w_i = (1 - (r_i / cmad)^2)^2`` clamped to 1 below ``0.001*cmad``
    and to 0 above ``0.999*cmad``, where ``cmad`` is three times the sum
    of the two "middle" elements as the reference's partial sort returns
    them (nominally ``6 * median(|r|)``; see :func:`_psort_pair` for the
    even-length quirk R inherits and we replicate).  A zero ``cmad``
    (perfect fit) weights exact fits 1 and all else 0.
    """
    r = np.abs(y - fit)
    n = len(r)
    work = r.copy()
    lo, hi = _psort_pair(work, n // 2 + 1, n - n // 2)
    cmad = 3.0 * (lo + hi)
    w = np.zeros(n, dtype=np.float64)
    if cmad <= 0.0:
        w[r <= 0.0] = 1.0
        return w
    small = r <= 0.001 * cmad
    mid = ~small & (r <= 0.999 * cmad)
    w[small] = 1.0
    w[mid] = (1.0 - (r[mid] / cmad) ** 2) ** 2
    return w
