"""Tests for the boundary (singular) fit diagnostic (F1).

Mirrors lme4's isSingular(): a fit is singular when a random-effects
variance collapses to ~0 or an implied correlation reaches ±1, both of
which show up as a Cholesky-factor diagonal at its lower bound of 0.
"""

import warnings

import numpy as np
import pytest

from pystatistics.mixed import lmm
from pystatistics.mixed._random_effects import (
    parse_random_effects, is_singular_fit,
)


def _specs_intercept(n_groups=6, n_per=8):
    n = n_groups * n_per
    group = np.repeat(np.arange(n_groups), n_per)
    return parse_random_effects({"g": group}, {"g": ["1"]}, None, n)


def _specs_intercept_slope(n_groups=6, n_per=8):
    n = n_groups * n_per
    group = np.repeat(np.arange(n_groups), n_per)
    x = np.tile(np.arange(n_per, dtype=float), n_groups)
    return parse_random_effects({"g": group}, {"g": ["1", "x"]}, {"x": x}, n)


class TestIsSingularDetector:
    """Direct tests of the detector against lme4's rule."""

    def test_zero_diagonal_is_singular(self):
        """A scalar variance at the boundary (Cholesky diag ~0) is singular."""
        specs = _specs_intercept()
        assert is_singular_fit(np.array([1e-6]), specs) is True

    def test_healthy_diagonal_not_singular(self):
        """A well-separated variance is not singular."""
        specs = _specs_intercept()
        assert is_singular_fit(np.array([0.9]), specs) is False

    def test_correlation_boundary_is_singular(self):
        """Correlation→±1 collapses the *trailing* Cholesky diagonal: the
        block [[d0, 0], [off, d1]] with d1~0 is detected even though the
        first variance (d0) is healthy."""
        specs = _specs_intercept_slope()
        # theta layout per 2-term block: [d0, off, d1]
        theta = np.array([1.2, 0.7, 1e-7])  # d1 ~ 0 → corr ±1
        assert is_singular_fit(theta, specs) is True

    def test_offdiagonal_does_not_trigger(self):
        """A large off-diagonal alone (healthy diagonals) is not singular."""
        specs = _specs_intercept_slope()
        theta = np.array([1.0, 5.0, 0.8])  # big off-diag, both diags healthy
        assert is_singular_fit(theta, specs) is False

    def test_tolerance_boundary(self):
        """Detection respects the lme4 default tol=1e-4."""
        specs = _specs_intercept()
        assert is_singular_fit(np.array([5e-5]), specs) is True   # below tol
        assert is_singular_fit(np.array([5e-4]), specs) is False  # above tol


class TestSingularFitEndToEnd:
    def test_no_effect_data_is_singular_and_warns(self):
        """Pure-noise data (no real group effect) → variance→0 boundary fit,
        like lme4 on Dyestuff2. Must warn and set is_singular."""
        rng = np.random.default_rng(7)
        n_groups, n_per = 12, 10
        n = n_groups * n_per
        group = np.repeat(np.arange(n_groups), n_per)
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])
        y = 3.0 + 1.5 * x + rng.standard_normal(n)  # no group-level variance

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = lmm(y, X, groups={"g": group})

        assert result.is_singular is True
        assert any("singular" in str(w.message).lower() for w in caught)
        # The estimate is still the correct boundary MLE.
        py_var = result.var_components[0].variance
        assert py_var < 0.5

    def test_genuine_effect_not_singular(self):
        """Data with a real, large group effect is not a boundary fit."""
        rng = np.random.default_rng(11)
        n_groups, n_per = 12, 12
        n = n_groups * n_per
        group = np.repeat(np.arange(n_groups), n_per)
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])
        re = rng.standard_normal(n_groups) * 4.0  # strong group variance
        y = 3.0 + 1.5 * x + re[group] + rng.standard_normal(n)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = lmm(y, X, groups={"g": group})

        assert result.is_singular is False
        assert not any("singular" in str(w.message).lower() for w in caught)
