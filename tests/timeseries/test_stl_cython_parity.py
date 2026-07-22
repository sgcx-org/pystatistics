"""Bit-identity: compiled STL/LOESS/robustness kernels == pure-Python reference.

The compiled kernels in ``_stl_kernels`` must reproduce the reference in
``_stl_ref`` (the verbatim original bodies) to the last bit. Covers loess
(degrees, jumps, weights), the moving-average cascade, the reference-exact
partial sort (including even-length inputs that trigger its quirk), the
bisquare robustness weights, and the full STL driver (robust and non-robust).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pystatistics.timeseries import _stl_kernels as cy
from pystatistics.timeseries import _stl_ref as ref


def _rng(seed):
    return np.random.default_rng(seed)  # NON-DETERMINISTIC: fixed seed


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("span", [3, 5, 7, 11])
@pytest.mark.parametrize("degree", [0, 1])
@pytest.mark.parametrize("jump", [1, 2, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_loess_smooth_matches_reference(seed, span, degree, jump, weighted):
    n = 40
    y = np.ascontiguousarray(_rng(seed).standard_normal(n))
    if weighted:
        w = np.ascontiguousarray(_rng(seed + 1).random(n))
        use_w = True
    else:
        w = np.empty(0, dtype=np.float64)
        use_w = False
    a = cy.loess_smooth_nb(y, span, degree, jump, w, use_w)
    b = ref.loess_smooth_nb(y, span, degree, jump, w, use_w)
    assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("width", [2, 3, 5, 12])
def test_moving_average_matches_reference(seed, width):
    x = np.ascontiguousarray(_rng(seed).standard_normal(50))
    assert_array_equal(np.asarray(cy.moving_average_nb(x, width)),
                       np.asarray(ref.moving_average_nb(x, width)))


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 9, 16, 17, 30, 31])
def test_psort_pair_matches_reference(seed, n):
    base = _rng(seed).standard_normal(n)
    first, second = n // 2 + 1, n - n // 2
    a1 = np.ascontiguousarray(base.copy())
    a2 = np.ascontiguousarray(base.copy())
    r1 = cy.psort_pair_nb(a1, first, second)
    r2 = ref.psort_pair_nb(a2, first, second)
    assert r1 == r2, f"n={n}: {r1} vs {r2}"
    assert_array_equal(a1, a2)  # in-place mutation identical too


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("n", [3, 4, 7, 8, 20, 21, 50])
def test_robustness_weights_matches_reference(seed, n):
    rng = _rng(seed)
    y = np.ascontiguousarray(rng.standard_normal(n))
    fit = np.ascontiguousarray(y + 0.1 * rng.standard_normal(n))
    # inject a couple of outliers to exercise the clamps
    fit[0] += 5.0
    assert_array_equal(np.asarray(cy.robustness_weights_nb(y, fit)),
                       np.asarray(ref.robustness_weights_nb(y, fit)))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("period", [4, 7, 12])
@pytest.mark.parametrize("n_outer,periodic", [(0, False), (2, False), (0, True)])
def test_stl_core_matches_reference(seed, period, n_outer, periodic):
    rng = _rng(seed)
    n = period * 8
    t = np.arange(n)
    y = np.ascontiguousarray(
        10 * np.sin(2 * np.pi * t / period)
        + 0.05 * t
        + rng.standard_normal(n)
    )
    # valid STL params (odd windows >= 3)
    s_win, t_win, l_win = 7, 15, period + 1 if period % 2 == 0 else period + 2
    args = (period, s_win, 0, 1, t_win, 1, 1, l_win, 1, 1, 2, n_outer, periodic)
    a = cy.stl_core_nb(y, *args)
    b = ref.stl_core_nb(y, *args)
    for ai, bi in zip(a, b):
        assert_array_equal(np.asarray(ai), np.asarray(bi))
