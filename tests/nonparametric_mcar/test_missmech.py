"""Tests for missmech_mcar_test (Jamshidian-Jalal-style permutation test)."""

import numpy as np
import pytest

from pystatistics.nonparametric_mcar import (
    NonparametricMCARResult,
    missmech_mcar_test,
)


def _mcar_matrix_with_patterns(n: int = 240, d: int = 4, seed: int = 0):
    """MCAR matrix where a few well-populated patterns arise (each column
    independently missing 25% of the time → 2^d patterns, some large)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    mask = rng.random((n, d)) < 0.25
    X[mask] = np.nan
    return X


def _mar_matrix_with_patterns(n: int = 240, d: int = 4, seed: int = 0):
    """MAR: col 1 missingness depends on col 0; col 2 missingness depends
    on col 3. Produces pattern-dependent observed-value distributions."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    mask1 = rng.random(n) < 1.0 / (1.0 + np.exp(-3.0 * X[:, 0]))
    X[mask1, 1] = np.nan
    mask2 = rng.random(n) < 1.0 / (1.0 + np.exp(-3.0 * X[:, 3]))
    X[mask2, 2] = np.nan
    return X


# ---------------------------------------------------------------------
# Normal cases
# ---------------------------------------------------------------------


def test_missmech_on_mcar_data_does_not_reject():
    X = _mcar_matrix_with_patterns(n=240, d=4, seed=0)
    r = missmech_mcar_test(X, n_permutations=99, seed=0, min_pattern_size=5)
    assert isinstance(r, NonparametricMCARResult)
    assert "Jamshidian-Jalal" in r.method
    assert r.statistic >= 0.0
    assert r.rejected is False


def test_missmech_rejects_mar_data():
    X = _mar_matrix_with_patterns(n=240, d=4, seed=0)
    r = missmech_mcar_test(X, n_permutations=99, seed=0, min_pattern_size=5)
    assert r.rejected is True


def test_missmech_reproducible_same_seed():
    X = _mcar_matrix_with_patterns(n=240, d=4, seed=1)
    r1 = missmech_mcar_test(X, n_permutations=19, seed=42, min_pattern_size=5)
    r2 = missmech_mcar_test(X, n_permutations=19, seed=42, min_pattern_size=5)
    assert r1.statistic == r2.statistic
    assert r1.p_value == r2.p_value


def test_missmech_reports_pattern_bookkeeping():
    X = _mcar_matrix_with_patterns(n=240, d=4, seed=0)
    r = missmech_mcar_test(X, n_permutations=19, seed=0, min_pattern_size=5)
    assert r.extra["n_patterns_used"] >= 2
    assert r.extra["n_patterns_used"] <= r.extra["n_patterns_total"]
    assert r.extra["n_rows_used"] <= X.shape[0]


# ---------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------


def test_raises_on_1d_input():
    with pytest.raises(ValueError, match="2D"):
        missmech_mcar_test(np.array([np.nan, 1.0, 2.0]))


def test_raises_on_small_n():
    X = _mcar_matrix_with_patterns(n=5, d=4)
    with pytest.raises(ValueError, match="at least 10 rows"):
        missmech_mcar_test(X, n_permutations=9)


def test_raises_on_single_column():
    X = np.full((30, 1), np.nan)
    X[::2] = 1.0
    with pytest.raises(ValueError, match="at least 2 columns"):
        missmech_mcar_test(X, n_permutations=9)


def test_raises_on_no_missingness():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    with pytest.raises(ValueError, match="missing cell"):
        missmech_mcar_test(X, n_permutations=9)


def test_raises_when_fewer_than_two_patterns_meet_min_size():
    """If we set min_pattern_size absurdly high, no patterns qualify → raise."""
    X = _mcar_matrix_with_patterns(n=30, d=4, seed=0)
    with pytest.raises(ValueError, match="at least 2 missingness patterns"):
        missmech_mcar_test(X, n_permutations=9, min_pattern_size=100)


def test_raises_on_invalid_n_neighbors():
    X = _mcar_matrix_with_patterns(n=30, d=4, seed=0)
    with pytest.raises(ValueError, match="n_neighbors"):
        missmech_mcar_test(X, n_permutations=9, n_neighbors=0)
