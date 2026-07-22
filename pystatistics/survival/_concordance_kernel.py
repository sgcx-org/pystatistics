"""
Concordance counting (the hot Fenwick loop) — public import surface.

Harrell's C is a comparable-pair count over an at-risk set that changes as an
event-time sweep advances — an inherently sequential Fenwick (binary-indexed
tree) loop that pure numpy cannot vectorise, and that the Cox fit calls once
per fit over every event. The arithmetic is integer counting held in float64,
so the compiled result is bit-identical to the pure-numpy reference
(:mod:`._concordance_ref`), which stays the blessed definition.

The kernels are compiled Cython (``_concordance_fenwick``); this module keeps
the historical import path (``concordance_counts_simple`` /
``concordance_counts_truncated``) stable for :mod:`._cox`.

Two kernels share the ranking/setup done by the caller:

* ``concordance_counts_simple`` — no left truncation: every subject is at risk
  from the start, so the sweep only ever ADDS rows to the tree (add-only).
* ``concordance_counts_truncated`` — counting-process rows: a subject is at risk
  only on ``(entry, exit]``, so the sweep both activates (at entry) and
  deactivates (at exit) rows.
"""

from __future__ import annotations

from ._concordance_fenwick import (  # noqa: F401
    concordance_counts_simple,
    concordance_counts_truncated,
)

__all__ = ["concordance_counts_simple", "concordance_counts_truncated"]
