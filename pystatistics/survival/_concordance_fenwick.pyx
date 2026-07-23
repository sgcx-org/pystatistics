# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
"""
Compiled concordance counting (the hot Fenwick loop) — Cython port of the
former Numba ``@njit`` kernels.

Harrell's C is a comparable-pair count over an at-risk set that changes as an
event-time sweep advances — an inherently sequential Fenwick (binary-indexed
tree) loop that pure numpy cannot vectorise, called once per Cox fit. The
arithmetic is integer counting held in float64, so the result is exact and
bit-identical to the pure-numpy reference in ``_concordance_ref``.

Two kernels:
  * ``concordance_counts_simple``    — add-only sweep (no left truncation).
  * ``concordance_counts_truncated`` — activation/deactivation sweep for
    counting-process rows. Its two argsorts are done in Python (deterministic
    stable mergesort) and the Fenwick sweep runs under ``nogil``.

Contract (Rule 2): ``rank`` and ``event_rows`` are C-contiguous int64;
``t``/``e``/``entry``/``uet`` are C-contiguous float64. The caller
(``_cox._concordance_counts``) guarantees this.
"""

import numpy as np
from numpy cimport int64_t


cdef inline void _fenwick_add(double* tree, Py_ssize_t r, Py_ssize_t size,
                              double delta) noexcept nogil:
    while r <= size:
        tree[r] += delta
        r += r & (-r)


cdef inline double _fenwick_prefix(double* tree, Py_ssize_t r) noexcept nogil:
    cdef double s = 0.0
    while r > 0:
        s += tree[r]
        r -= r & (-r)
    return s


def concordance_counts_simple(const int64_t[::1] rank, const double[::1] t,
                              const double[::1] e, const double[::1] uet,
                              Py_ssize_t size):
    """Add-only concordance sweep (no left truncation).
    Returns (concordant, discordant, tied_risk)."""
    cdef Py_ssize_t n = rank.shape[0]
    cdef double[::1] tree = np.zeros(size + 1, dtype=np.float64)
    cdef double* treep = &tree[0]
    cdef const int64_t* rankp = &rank[0]
    cdef const double* tp = &t[0]
    cdef const double* ep = &e[0]

    cdef double concordant = 0.0, discordant = 0.0, tied_risk = 0.0
    cdef double tj, less, leq
    cdef Py_ssize_t seen = 0, i, lo, k, r

    with nogil:
        i = n - 1
        while i >= 0:
            tj = tp[i]
            lo = i
            while lo >= 0 and tp[lo] == tj:
                lo -= 1
            lo += 1
            for k in range(lo, i + 1):
                if ep[k] == 0.0:
                    _fenwick_add(treep, <Py_ssize_t>rankp[k], size, 1.0)
                    seen += 1
            for k in range(lo, i + 1):
                if ep[k] == 1.0:
                    r = <Py_ssize_t>rankp[k]
                    less = _fenwick_prefix(treep, r - 1)
                    leq = _fenwick_prefix(treep, r)
                    concordant += less
                    tied_risk += leq - less
                    discordant += seen - leq
            for k in range(lo, i + 1):
                if ep[k] == 1.0:
                    _fenwick_add(treep, <Py_ssize_t>rankp[k], size, 1.0)
                    seen += 1
            i = lo - 1
    return concordant, discordant, tied_risk


def concordance_counts_truncated(const int64_t[::1] rank, const double[::1] t,
                                 const double[::1] entry,
                                 const int64_t[::1] event_rows,
                                 const double[::1] uet, Py_ssize_t size):
    """Activation/deactivation concordance sweep for counting-process rows.
    Returns (concordant, discordant, tied_risk)."""
    cdef Py_ssize_t n = rank.shape[0]
    cdef Py_ssize_t m = uet.shape[0]
    cdef Py_ssize_t n_ev = event_rows.shape[0]
    cdef double[::1] tree = np.zeros(size + 1, dtype=np.float64)
    cdef unsigned char[::1] removed = np.zeros(n, dtype=np.uint8)
    # Stable argsorts done in Python (mergesort) — matches the reference; the
    # Fenwick sweep below is what needs to be fast and runs nogil.
    cdef int64_t[::1] stop_order = np.argsort(
        np.asarray(t), kind="mergesort").astype(np.int64)
    cdef int64_t[::1] entry_order = np.argsort(
        np.asarray(entry), kind="mergesort").astype(np.int64)

    cdef double* treep = &tree[0]
    cdef unsigned char* remp = &removed[0]
    cdef const int64_t* rankp = &rank[0]
    cdef const int64_t* evp = &event_rows[0]
    cdef const int64_t* sop = &stop_order[0]
    cdef const int64_t* eop = &entry_order[0]
    cdef const double* tp = &t[0]
    cdef const double* entp = &entry[0]
    cdef const double* uetp = &uet[0]

    cdef double concordant = 0.0, discordant = 0.0, tied_risk = 0.0
    cdef double tj, less, leq
    cdef Py_ssize_t active = 0, pa = 0, pr = 0, pe = 0, pe_start, g, idx, k, r

    with nogil:
        for g in range(m):
            tj = uetp[g]
            while pa < n and entp[eop[pa]] < tj:
                k = <Py_ssize_t>eop[pa]
                _fenwick_add(treep, <Py_ssize_t>rankp[k], size, 1.0)
                active += 1
                pa += 1
            while pr < n and tp[sop[pr]] < tj:
                k = <Py_ssize_t>sop[pr]
                if remp[k] == 0:
                    _fenwick_add(treep, <Py_ssize_t>rankp[k], size, -1.0)
                    active -= 1
                    remp[k] = 1
                pr += 1
            pe_start = pe
            while pe < n_ev and tp[evp[pe]] == tj:
                k = <Py_ssize_t>evp[pe]
                _fenwick_add(treep, <Py_ssize_t>rankp[k], size, -1.0)
                active -= 1
                remp[k] = 1
                pe += 1
            for idx in range(pe_start, pe):
                k = <Py_ssize_t>evp[idx]
                r = <Py_ssize_t>rankp[k]
                less = _fenwick_prefix(treep, r - 1)
                leq = _fenwick_prefix(treep, r)
                concordant += less
                tied_risk += leq - less
                discordant += active - leq
    return concordant, discordant, tied_risk
