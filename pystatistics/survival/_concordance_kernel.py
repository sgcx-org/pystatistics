"""
Numba-compiled concordance counting (the hot Fenwick loop).

Harrell's C is a comparable-pair count over an at-risk set that changes as an
event-time sweep advances — an inherently sequential Fenwick (binary-indexed
tree) loop that pure numpy cannot vectorise, and that the Cox fit calls once
per fit over every event. Compiling it removes the per-row Python overhead that
otherwise dominates the fit at moderate n; the arithmetic is integer counting,
so ``fastmath=False`` and the result is bit-identical to the pure-Python
reference (:func:`._cox._concordance_counts`), which stays the blessed
definition. Same house pattern as the ETS / SARIMA / STL kernels.

Two kernels share the ranking/setup done by the caller:

* ``concordance_counts_simple`` — no left truncation: every subject is at risk
  from the start, so the sweep only ever ADDS rows to the tree (add-only), which
  is what the pre-truncation implementation did.
* ``concordance_counts_truncated`` — counting-process rows: a subject is at risk
  only on ``(entry, exit]``, so the sweep both activates (at entry) and
  deactivates (at exit) rows.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from numba import njit


@njit(cache=True, fastmath=False)
def _fenwick_add(tree, r, size, delta):
    while r <= size:
        tree[r] += delta
        r += r & (-r)


@njit(cache=True, fastmath=False)
def _fenwick_prefix(tree, r):
    s = 0.0
    while r > 0:
        s += tree[r]
        r -= r & (-r)
    return s


@njit(cache=True, fastmath=False)
def concordance_counts_simple(rank, t, e, uet, size):
    """Add-only concordance sweep (no left truncation).

    Processes descending time. A censored subject tied to an event time
    outlives the event (comparable); two events at the same time are a time tie
    (not comparable). Returns (concordant, discordant, tied_risk).
    """
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
        # Insert this block's censored subjects first — they are comparable
        # "later" subjects for the block's events.
        for k in range(lo, i + 1):
            if e[k] == 0.0:
                _fenwick_add(tree, rank[k], size, 1.0)
                seen += 1
        # Query the block's events against everything currently at risk.
        for k in range(lo, i + 1):
            if e[k] == 1.0:
                r = rank[k]
                less = _fenwick_prefix(tree, r - 1)
                leq = _fenwick_prefix(tree, r)
                concordant += less
                tied_risk += leq - less
                discordant += seen - leq
        # Then insert the block's events (later-time for earlier events).
        for k in range(lo, i + 1):
            if e[k] == 1.0:
                _fenwick_add(tree, rank[k], size, 1.0)
                seen += 1
        i = lo - 1
    return concordant, discordant, tied_risk


@njit(cache=True, fastmath=False)
def concordance_counts_truncated(rank, t, entry, event_rows, uet, size):
    """Activation/deactivation concordance sweep for counting-process rows.

    ``event_rows`` are the row indices of events, ascending by exit time; the
    tree holds rows currently at risk (entry < tj <= exit). Returns
    (concordant, discordant, tied_risk).
    """
    n = rank.shape[0]
    tree = np.zeros(size + 1, dtype=np.float64)
    removed = np.zeros(n, dtype=np.bool_)
    # Sort keys precomputed by the caller are avoided; do the argsorts here so
    # the kernel is self-contained on plain arrays.
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
