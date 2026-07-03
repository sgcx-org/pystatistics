"""
Unit tests for the R-faithful loess kernel (``timeseries._loess``).

Covers the mathematical invariants of the smoother (exact reproduction
of polynomial data at the fitted degree, interpolation exactness, weight
handling, fallbacks) plus edge and contract cases.  End-to-end numerical
parity with R is covered by ``test_stl_r_parity.py``; these tests pin
down the kernel's own behaviour in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pystatistics.timeseries._loess import (
    _estimate_windows,
    loess_smooth,
    loess_subseries_smooth,
)


class TestLoessSmoothNormal:
    """Normal cases: polynomial reproduction and weighting."""

    def test_degree1_reproduces_linear_data_exactly(self):
        """Weighted linear regression on exactly linear data is exact."""
        y = 3.0 + 0.5 * np.arange(50, dtype=float)
        out = loess_smooth(y, span=11, degree=1, jump=1)
        np.testing.assert_allclose(out, y, atol=1e-12, rtol=0.0)

    def test_degree0_reproduces_constant_data_exactly(self):
        y = np.full(30, 7.25)
        out = loess_smooth(y, span=7, degree=0, jump=1)
        np.testing.assert_allclose(out, y, atol=1e-14, rtol=0.0)

    def test_jump_interpolation_exact_on_linear_data(self):
        """Linear interpolation between exact linear estimates is exact."""
        y = -2.0 + 1.5 * np.arange(41, dtype=float)
        out = loess_smooth(y, span=9, degree=1, jump=4)
        np.testing.assert_allclose(out, y, atol=1e-11, rtol=0.0)

    def test_jump_equals_direct_at_evaluated_points(self):
        """Points on the jump grid equal the jump=1 estimates."""
        rng = np.random.default_rng(3)
        y = rng.normal(0.0, 1.0, 60)
        direct = loess_smooth(y, span=13, degree=1, jump=1)
        strided = loess_smooth(y, span=13, degree=1, jump=5)
        grid = np.arange(0, 60, 5)
        np.testing.assert_allclose(strided[grid], direct[grid],
                                   atol=1e-12, rtol=0.0)

    def test_smoothing_reduces_noise_variance(self):
        rng = np.random.default_rng(4)
        y = rng.normal(0.0, 1.0, 200)
        out = loess_smooth(y, span=41, degree=1, jump=1)
        assert np.var(out) < 0.5 * np.var(y)

    def test_external_weights_shift_estimate(self):
        """Zero-weighting a corrupted point removes its influence."""
        y = np.zeros(21)
        y[10] = 100.0
        w_uniform = np.ones(21)
        w_masked = w_uniform.copy()
        w_masked[10] = 0.0
        smooth_unif = loess_smooth(y, span=21, degree=0, jump=1,
                                   weights=w_uniform)
        smooth_mask = loess_smooth(y, span=21, degree=0, jump=1,
                                   weights=w_masked)
        assert abs(smooth_mask[10]) < abs(smooth_unif[10])
        np.testing.assert_allclose(smooth_mask, 0.0, atol=1e-12)


class TestLoessSmoothEdge:
    """Edge cases: tiny inputs, spans beyond the series, fallbacks."""

    def test_single_point_returned_unchanged(self):
        y = np.array([42.0])
        out = loess_smooth(y, span=7, degree=1, jump=1)
        np.testing.assert_array_equal(out, y)

    def test_two_points(self):
        y = np.array([1.0, 3.0])
        out = loess_smooth(y, span=3, degree=1, jump=1)
        assert out.shape == (2,)
        assert np.all(np.isfinite(out))

    def test_span_larger_than_series(self):
        """span >= n widens the bandwidth; smooth stays finite/sane."""
        y = 5.0 + 0.1 * np.arange(10, dtype=float)
        out = loess_smooth(y, span=101, degree=1, jump=1)
        assert np.all(np.isfinite(out))
        # Linear data must still be reproduced exactly by degree 1.
        np.testing.assert_allclose(out, y, atol=1e-10, rtol=0.0)

    def test_jump_larger_than_series_capped(self):
        y = np.arange(10, dtype=float)
        out = loess_smooth(y, span=5, degree=1, jump=100)
        assert out.shape == (10,)
        np.testing.assert_allclose(out, y, atol=1e-11, rtol=0.0)

    def test_all_zero_weights_fall_back_to_observations(self):
        """A window with zero total weight falls back to the raw value."""
        y = np.arange(9, dtype=float)
        weights = np.zeros(9)
        out = loess_smooth(y, span=3, degree=0, jump=1, weights=weights)
        np.testing.assert_array_equal(out, y)


class TestSubseriesSmooth:
    """The grouped cycle-subseries kernel."""

    def test_matches_per_series_smoothing(self):
        """Group evaluation equals smoothing each subseries alone."""
        rng = np.random.default_rng(7)
        sub_y = rng.normal(0.0, 1.0, (5, 20))
        grouped, head, tail = loess_subseries_smooth(
            sub_y, span=7, degree=1, jump=2)
        for row in range(5):
            single, h1, t1 = loess_subseries_smooth(
                sub_y[row:row + 1], span=7, degree=1, jump=2)
            np.testing.assert_allclose(grouped[row], single[0],
                                       atol=1e-14, rtol=0.0)
            np.testing.assert_allclose(head[row], h1[0], atol=1e-14)
            np.testing.assert_allclose(tail[row], t1[0], atol=1e-14)

    def test_extensions_extrapolate_linear_data(self):
        """Degree-1 extensions at 0 and k+1 continue an exact line."""
        k = 15
        line = 2.0 + 0.5 * np.arange(1, k + 1, dtype=float)
        smoothed, head, tail = loess_subseries_smooth(
            line[None, :], span=7, degree=1, jump=1)
        np.testing.assert_allclose(smoothed[0], line, atol=1e-11)
        np.testing.assert_allclose(head[0], 2.0, atol=1e-10)       # pos 0
        np.testing.assert_allclose(tail[0], 2.0 + 0.5 * (k + 1), atol=1e-10)

    def test_weights_respected_per_row(self):
        """Each row's weights only affect that row."""
        sub_y = np.zeros((2, 12))
        sub_y[0, 5] = 60.0
        sub_y[1, 5] = 60.0
        weights = np.ones((2, 12))
        weights[0, 5] = 0.0  # mask the spike only in row 0
        smoothed, _, _ = loess_subseries_smooth(
            sub_y, span=25, degree=0, jump=1, sub_weights=weights)
        assert abs(smoothed[0, 5]) < abs(smoothed[1, 5])


class TestEstimateWindowsContract:
    """Failure/degenerate cases of the vectorised core."""

    def test_zero_total_weight_flags_not_ok(self):
        yw = np.array([[1.0, 2.0, 3.0]])
        ww = np.zeros((1, 3))
        values, ok = _estimate_windows(
            yw, 3, 3, 0, np.array([2.0]), np.array([1], dtype=np.int64), ww)
        assert not ok[0]

    def test_degenerate_spread_falls_back_to_weighted_mean(self):
        """When all weight sits on one point, degree 1 skips the slope."""
        yw = np.array([[5.0, 9.0, 13.0]])
        ww = np.array([[0.0, 1.0, 0.0]])
        values, ok = _estimate_windows(
            yw, 3, 3, 1, np.array([2.0]), np.array([1], dtype=np.int64), ww)
        assert ok[0]
        # Weighted mean of the single carried point, no linear adjustment.
        np.testing.assert_allclose(values[0], 9.0, atol=1e-12)
