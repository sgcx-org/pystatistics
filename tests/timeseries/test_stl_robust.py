"""
Unit tests for STL robustness weighting (``timeseries._stl_robust``).

The bisquare weight formula and the reference partial-sort semantics are
pinned here in isolation; bit-exact agreement with R through full robust
STL runs (including the even-length partial-sort quirk R inherits) is
covered by the robust cases in ``test_stl_r_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pystatistics.timeseries._stl_robust import _psort_pair, _robustness_weights


class TestPsortPair:
    """The reference-exact partial sort."""

    def test_odd_length_returns_true_median_twice(self):
        """For odd n both requested positions are the median."""
        rng = np.random.default_rng(11)
        for n in (5, 9, 33, 101):
            values = rng.normal(0.0, 1.0, n)
            lo, hi = _psort_pair(values.copy(), n // 2 + 1, n - n // 2)
            med = float(np.sort(values)[n // 2])
            assert lo == med and hi == med, f"odd n={n}"

    def test_returned_values_are_array_elements(self):
        """Whatever the partial sort returns, both values come from the
        input array (the even-n quirk mis-selects, it never invents)."""
        rng = np.random.default_rng(12)
        for n in (6, 44, 144, 200):
            values = rng.normal(0.0, 1.0, n)
            lo, hi = _psort_pair(values.copy(), n // 2 + 1, n - n // 2)
            assert lo in values and hi in values, f"even n={n}"

    def test_deterministic(self):
        rng = np.random.default_rng(13)
        values = rng.normal(0.0, 1.0, 144)
        first = _psort_pair(values.copy(), 73, 72)
        second = _psort_pair(values.copy(), 73, 72)
        assert first == second

    def test_sorted_input_even_length(self):
        """On already-sorted input the pair is the true middle pair."""
        values = np.arange(20, dtype=float)
        lo, hi = _psort_pair(values.copy(), 11, 10)
        assert {lo, hi} == {9.0, 10.0}

    def test_tiny_arrays(self):
        lo, hi = _psort_pair(np.array([3.0]), 1, 1)
        assert lo == 3.0 and hi == 3.0
        lo, hi = _psort_pair(np.array([2.0, 1.0]), 2, 1)
        assert lo + hi == 3.0


class TestRobustnessWeights:
    """Bisquare weights on the remainder."""

    def test_zero_residuals_get_weight_one(self):
        y = np.arange(10, dtype=float)
        fit = y.copy()
        fit[3] += 5.0  # one large residual so cmad > 0
        w = _robustness_weights(y, fit)
        assert np.all(w[np.abs(y - fit) == 0.0] == 1.0)

    def test_large_outliers_get_weight_zero(self):
        rng = np.random.default_rng(14)
        y = rng.normal(0.0, 1.0, 101)
        fit = np.zeros(101)
        y[50] = 1e6
        w = _robustness_weights(y, fit)
        assert w[50] == 0.0

    def test_weights_in_unit_interval_and_decreasing(self):
        rng = np.random.default_rng(15)
        y = rng.normal(0.0, 1.0, 75)
        fit = np.zeros(75)
        w = _robustness_weights(y, fit)
        assert np.all((w >= 0.0) & (w <= 1.0))
        # Larger |residual| never gets a larger weight.
        order = np.argsort(np.abs(y - fit))
        assert np.all(np.diff(w[order]) <= 1e-12)

    def test_matches_bisquare_formula_odd_n(self):
        """For odd n (no partial-sort quirk) weights equal the exact
        bisquare with cmad = 6 * median(|r|)."""
        rng = np.random.default_rng(16)
        y = rng.normal(0.0, 2.0, 101)
        fit = np.zeros(101)
        r = np.abs(y - fit)
        cmad = 6.0 * np.median(r)
        expected = np.zeros(101)
        low = r <= 0.001 * cmad
        mid = ~low & (r <= 0.999 * cmad)
        expected[low] = 1.0
        expected[mid] = (1.0 - (r[mid] / cmad) ** 2) ** 2
        np.testing.assert_allclose(
            _robustness_weights(y, fit), expected, atol=1e-14, rtol=0.0)

    def test_perfect_fit_degenerate_cmad(self):
        """cmad == 0 (perfect fit): exact fits weigh 1, everything else 0."""
        y = np.ones(12)
        w = _robustness_weights(y, y.copy())
        np.testing.assert_array_equal(w, np.ones(12))
        fit = y.copy()
        y2 = y.copy()
        y2[0] += 1.0  # a single nonzero residual; median |r| still 0
        w2 = _robustness_weights(y2, fit)
        assert w2[0] == 0.0
        np.testing.assert_array_equal(w2[1:], np.ones(11))
