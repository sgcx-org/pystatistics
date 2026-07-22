"""Bit-identity: compiled concordance Fenwick kernels == pure-numpy reference.

The counts are integers held in float64, so equality is exact by construction;
this pins it across random simple (add-only) and truncated (counting-process)
inputs, including tie-heavy configurations.
"""

from __future__ import annotations

import numpy as np
import pytest

from pystatistics.survival import _concordance_fenwick as cy
from pystatistics.survival import _concordance_ref as ref


def _make_simple(seed, n, n_ranks, tie_scale):
    rng = np.random.default_rng(seed)  # NON-DETERMINISTIC: fixed seed
    # Ranks 1..K (ties share a rank), integer-valued.
    rank = rng.integers(1, n_ranks + 1, size=n).astype(np.int64)
    size = int(n_ranks)
    # Times with deliberate ties (small integer grid when tie_scale small).
    t = np.sort(rng.integers(1, max(2, int(n * tie_scale)) + 1, size=n)).astype(np.float64)
    e = (rng.random(n) < 0.6).astype(np.float64)
    uet = np.unique(t[e == 1])
    return rank, t, e, uet, size


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("tie_scale", [0.2, 1.0, 3.0])
def test_simple_matches_reference_bit_for_bit(seed, tie_scale):
    rank, t, e, uet, size = _make_simple(seed, n=80, n_ranks=25, tie_scale=tie_scale)
    a = cy.concordance_counts_simple(rank, t, e, uet, size)
    b = ref.concordance_counts_simple(rank, t, e, uet, size)
    assert a == b, f"{a} vs {b}"


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("tie_scale", [0.2, 1.0, 3.0])
def test_truncated_matches_reference_bit_for_bit(seed, tie_scale):
    rng = np.random.default_rng(1000 + seed)
    n = 80
    n_ranks = 25
    rank = rng.integers(1, n_ranks + 1, size=n).astype(np.int64)
    size = int(n_ranks)
    grid = max(2, int(n * tie_scale)) + 1
    entry = rng.integers(0, grid, size=n).astype(np.float64)
    span = rng.integers(1, grid, size=n).astype(np.float64)
    t = entry + span  # exit strictly after entry
    e = (rng.random(n) < 0.6).astype(np.float64)
    uet = np.unique(t[e == 1])
    stop_order = np.argsort(t, kind="mergesort")
    event_rows = stop_order[e[stop_order] == 1].astype(np.int64)

    a = cy.concordance_counts_truncated(rank, t, entry, event_rows, uet, size)
    b = ref.concordance_counts_truncated(rank, t, entry, event_rows, uet, size)
    assert a == b, f"{a} vs {b}"
