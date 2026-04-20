"""Tests for hsic_mcar_test.

Covers:
    Normal cases:
        - HSIC is non-negative and small under MCAR; does not reject.
        - HSIC rejects under strongly MAR-generated missingness.
        - Reproducible given the same seed.
    Edge cases:
        - Runs on a matrix with just one missing column.
        - Bandwidths reported in extra are positive.
    Failure cases:
        - Non-2D input raises.
        - Too few rows raises.
        - Too few columns raises.
        - A fully-observed matrix raises (test undefined).
"""

import numpy as np
import pytest

from pystatistics.nonparametric_mcar import (
    NonparametricMCARResult,
    hsic_mcar_test,
)


def _mcar_matrix(n: int = 60, d: int = 4, p_miss: float = 0.25, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    mask = rng.random((n, d)) < p_miss
    X[mask] = np.nan
    return X


def _mar_matrix(n: int = 120, d: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    drop_prob = 1.0 / (1.0 + np.exp(-3.0 * X[:, 0]))
    mask = rng.random(n) < drop_prob
    X[mask, 1] = np.nan
    return X


# ---------------------------------------------------------------------
# Normal cases
# ---------------------------------------------------------------------


def test_hsic_on_mcar_data_does_not_reject():
    X = _mcar_matrix(n=80, d=4, p_miss=0.25, seed=0)
    r = hsic_mcar_test(X, n_permutations=99, seed=0)
    assert isinstance(r, NonparametricMCARResult)
    assert "HSIC" in r.method
    assert r.statistic >= 0.0
    assert r.rejected is False


def test_hsic_rejects_mar_data():
    X = _mar_matrix(n=200, d=4, seed=0)
    r = hsic_mcar_test(X, n_permutations=99, seed=0)
    assert r.rejected is True


def test_hsic_reproducible_same_seed():
    X = _mcar_matrix(n=60, d=4, seed=1)
    r1 = hsic_mcar_test(X, n_permutations=19, seed=42)
    r2 = hsic_mcar_test(X, n_permutations=19, seed=42)
    assert r1.statistic == r2.statistic
    assert r1.p_value == r2.p_value


def test_hsic_reports_positive_bandwidths():
    X = _mcar_matrix(n=60, d=4, seed=0)
    r = hsic_mcar_test(X, n_permutations=19, seed=0)
    assert r.extra["bandwidth_X"] > 0.0
    assert r.extra["bandwidth_R"] > 0.0


# ---------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------


def test_raises_on_1d_input():
    with pytest.raises(ValueError, match="2D"):
        hsic_mcar_test(np.array([np.nan, 1.0, 2.0]))


def test_raises_on_small_n():
    X = _mcar_matrix(n=5, d=4)
    with pytest.raises(ValueError, match="at least 10 rows"):
        hsic_mcar_test(X, n_permutations=9)


def test_raises_on_single_column():
    X = np.full((30, 1), np.nan)
    X[::2] = 1.0
    with pytest.raises(ValueError, match="at least 2 columns"):
        hsic_mcar_test(X, n_permutations=9)


def test_raises_on_no_missingness():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    with pytest.raises(ValueError, match="missing cell"):
        hsic_mcar_test(X, n_permutations=9)


def test_raises_on_invalid_alpha():
    X = _mcar_matrix(n=30, d=3, seed=0)
    with pytest.raises(ValueError, match="alpha"):
        hsic_mcar_test(X, alpha=0.0, n_permutations=9)
