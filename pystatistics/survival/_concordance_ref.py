"""Pure-Python/numpy reference for the concordance Fenwick kernels.

Bit-identity oracle for the compiled kernels in ``_concordance_fenwick`` (the
tests assert the compiled output equals these to the last bit). The arithmetic
is integer counting held in float64, so equality is exact by construction; the
oracle exists to pin that and to document the algorithm in readable form.

Line-for-line twins of the compiled kernels — do not "optimise" this file.
Test-only; never imported on the hot path.
"""

from __future__ import annotations

import numpy as np


def _fenwick_add(tree, r, size, delta):
    while r <= size:
        tree[r] += delta
        r += r & (-r)


def _fenwick_prefix(tree, r):
    s = 0.0
    while r > 0:
        s += tree[r]
        r -= r & (-r)
    return s


def concordance_counts_simple(rank, t, e, uet, size):
    n = rank.shape[0]
    tree = np.zeros(size + 1, dtype=np.float64)
    concordant = 0.0
    discordant = 0.0
    tied_risk = 0.0
    seen = 0
    i = n - 1
    while i >= 0:
        tj = t[i]
        lo = i
        while lo >= 0 and t[lo] == tj:
            lo -= 1
        lo += 1
        for k in range(lo, i + 1):
            if e[k] == 0.0:
                _fenwick_add(tree, rank[k], size, 1.0)
                seen += 1
        for k in range(lo, i + 1):
            if e[k] == 1.0:
                r = rank[k]
                less = _fenwick_prefix(tree, r - 1)
                leq = _fenwick_prefix(tree, r)
                concordant += less
                tied_risk += leq - less
                discordant += seen - leq
        for k in range(lo, i + 1):
            if e[k] == 1.0:
                _fenwick_add(tree, rank[k], size, 1.0)
                seen += 1
        i = lo - 1
    return concordant, discordant, tied_risk


def concordance_counts_truncated(rank, t, entry, event_rows, uet, size):
    n = rank.shape[0]
    tree = np.zeros(size + 1, dtype=np.float64)
    removed = np.zeros(n, dtype=np.bool_)
    stop_order = np.argsort(t, kind="mergesort")
    entry_order = np.argsort(entry, kind="mergesort")

    concordant = 0.0
    discordant = 0.0
    tied_risk = 0.0
    active = 0
    pa = 0
    pr = 0
    pe = 0
    m = uet.shape[0]
    for g in range(m):
        tj = uet[g]
        while pa < n and entry[entry_order[pa]] < tj:
            k = entry_order[pa]
            _fenwick_add(tree, rank[k], size, 1.0)
            active += 1
            pa += 1
        while pr < n and t[stop_order[pr]] < tj:
            k = stop_order[pr]
            if not removed[k]:
                _fenwick_add(tree, rank[k], size, -1.0)
                active -= 1
                removed[k] = True
            pr += 1
        pe_start = pe
        while pe < event_rows.shape[0] and t[event_rows[pe]] == tj:
            k = event_rows[pe]
            _fenwick_add(tree, rank[k], size, -1.0)
            active -= 1
            removed[k] = True
            pe += 1
        for idx in range(pe_start, pe):
            k = event_rows[idx]
            r = rank[k]
            less = _fenwick_prefix(tree, r - 1)
            leq = _fenwick_prefix(tree, r)
            concordant += less
            tied_risk += leq - less
            discordant += active - leq
    return concordant, discordant, tied_risk
