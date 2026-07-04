"""
STL robustness weighting, matching R ``stats::stl`` (numba-compiled).

The outer STL loop downweights outlying observations with Tukey's
bisquare applied to the remainder, scaled by six times the median
absolute residual.  The reference implementation selects its two
"middle" order statistics through a partial quicksort whose behaviour
for even-length inputs deviates from a true partial sort (see
:func:`psort_pair_nb`); R inherits that behaviour, so exact parity
requires replicating the algorithm rather than substituting a correct
median.  Clean-room implementation from the algorithm (no
transliteration of R's GPL Fortran).
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(cache=True, fastmath=False)
def psort_pair_nb(a, first, second):
    """
    Reference-exact partial sort returning ``(a[first-1], a[second-1])``
    (1-based positions) as the STL robustness step selects them.  Mutates
    ``a`` in place.

    The reference implementation picks its two "middle" order statistics
    with a partial quicksort (netlib ``psort``) whose requested positions
    arrive in the order ``(n//2 + 1, n - n//2)`` — *descending* whenever
    *n* is even.  psort's segment bookkeeping assumes ascending requests,
    so on some inputs one of the two returned positions holds an element
    that is **not** the true order statistic (empirically roughly 10% of
    random even-length vectors, growing with length; absent for n <= 4).
    R inherits this behaviour, so exact parity requires reproducing the
    algorithm — median-of-three quicksort with insertion sort below
    length 11 and the exact stat-range pruning — rather than substituting
    a correct partial sort.  Reimplemented clean-room from the algorithm;
    positions handled 1-based as in the algorithm description, with
    0-based array access.  The segment stack is a fixed-size array
    (recursion depth is bounded by ~2*log2(n)).
    """
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
    # States: 0 top, 1 pop, 2 size_check, 3 partition, 4 small.
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
        else:  # state == 4 (small): insertion sort, needs a[i-1] sentinel
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
    """
    Bisquare robustness weights on the remainder ``y - fit``.

    ``w_i = (1 - (r_i / cmad)^2)^2`` clamped to 1 below ``0.001*cmad``
    and to 0 above ``0.999*cmad``, where ``cmad`` is three times the sum
    of the two "middle" elements as the reference's partial sort returns
    them (nominally ``6 * median(|r|)``; see :func:`psort_pair_nb` for the
    even-length quirk R inherits and we replicate).  A zero ``cmad``
    (perfect fit) weights exact fits 1 and all else 0.
    """
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
            t = 1.0 - (ri / cmad) ** 2
            w[i] = t * t
    return w


# ---------------------------------------------------------------------------
# Python wrappers (stable public surface over the numba kernels)
# ---------------------------------------------------------------------------

def _psort_pair(a: NDArray, first: int, second: int) -> tuple[float, float]:
    """Wrapper over :func:`psort_pair_nb` (see it for the semantics)."""
    return psort_pair_nb(np.ascontiguousarray(a, dtype=np.float64), first, second)


def _robustness_weights(y: NDArray, fit: NDArray) -> NDArray:
    """Wrapper over :func:`robustness_weights_nb`."""
    return robustness_weights_nb(
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(fit, dtype=np.float64),
    )
