"""
Tests for rank-deficiency (collinearity) detection in MVN MLE.

Rank-deficient input — (near-)collinear variables — has no interior
maximum-likelihood estimate. The library must detect this and fail loudly
(or, under force=True, report it honestly) rather than returning a
meaningless fit with converged=True. These tests cover the detector itself
and its integration into mlest() across algorithms.
"""

import warnings

import numpy as np
import pytest

from pystatistics.core.exceptions import SingularMatrixError
from pystatistics.mvnmle import mlest, datasets
from pystatistics.mvnmle._degeneracy import (
    DEFAULT_COLLINEARITY_TOL,
    check_fitted_covariance,
    correlation_min_eigenvalue,
)


# ── Fixtures ─────────────────────────────────────────────────────────

def _collinear_data(seed: int = 0, n: int = 300) -> np.ndarray:
    """Data with an exactly duplicated column → rank-deficient covariance."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    X = np.column_stack([a, b, a.copy()])
    X[::15, 1] = np.nan
    return X


# ── correlation_min_eigenvalue ───────────────────────────────────────

class TestCorrelationMinEigenvalue:
    """The scale-invariant degeneracy measure."""

    def test_identity_is_one(self):
        assert correlation_min_eigenvalue(np.eye(3)) == pytest.approx(1.0)

    def test_scale_invariance(self):
        # Different variable scales must not change the measure.
        base = np.array([[1.0, 0.5], [0.5, 1.0]])
        scaled = np.diag([1e6, 1e-3]) @ base @ np.diag([1e6, 1e-3])
        assert correlation_min_eigenvalue(scaled) == pytest.approx(
            correlation_min_eigenvalue(base), rel=1e-6
        )

    def test_near_collinear_pair(self):
        r = 0.99999
        sigma = np.array([[1.0, r], [r, 1.0]])
        # min eigenvalue of [[1,r],[r,1]] is 1 - r
        assert correlation_min_eigenvalue(sigma) == pytest.approx(1 - r, rel=1e-3)

    def test_exact_duplicate_is_zero(self):
        sigma = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert correlation_min_eigenvalue(sigma) == pytest.approx(0.0, abs=1e-12)

    def test_non_finite_returns_zero(self):
        sigma = np.array([[1.0, np.nan], [np.nan, 1.0]])
        assert correlation_min_eigenvalue(sigma) == 0.0

    def test_non_positive_diagonal_returns_zero(self):
        sigma = np.array([[0.0, 0.0], [0.0, 1.0]])
        assert correlation_min_eigenvalue(sigma) == 0.0

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            correlation_min_eigenvalue(np.ones((2, 3)))


# ── check_fitted_covariance ──────────────────────────────────────────

class TestCheckFittedCovariance:
    """The raise / warn guard."""

    def test_full_rank_returns_none(self):
        sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        assert check_fitted_covariance(sigma) is None

    def test_degenerate_raises(self):
        sigma = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(SingularMatrixError, match="rank-deficient"):
            check_fitted_covariance(sigma)

    def test_degenerate_force_returns_warning(self):
        sigma = np.array([[1.0, 1.0], [1.0, 1.0]])
        msg = check_fitted_covariance(sigma, force=True)
        assert msg is not None
        assert "force=True" in msg

    def test_tol_controls_sensitivity(self):
        r = 0.999  # min eig = 1e-3
        sigma = np.array([[1.0, r], [r, 1.0]])
        # Default tol (1e-5) treats this as full-rank.
        assert check_fitted_covariance(sigma, tol=DEFAULT_COLLINEARITY_TOL) is None
        # A stricter tol flags it.
        with pytest.raises(SingularMatrixError):
            check_fitted_covariance(sigma, tol=1e-2)


# ── Integration through mlest ────────────────────────────────────────

class TestMlestDegeneracyGuard:
    """End-to-end behaviour across algorithms and the force escape hatch."""

    def test_direct_collinear_raises(self):
        X = _collinear_data()
        with pytest.raises(SingularMatrixError, match="collinear"):
            mlest(X, backend="cpu", algorithm="direct")

    def test_force_returns_non_converged_with_warning(self):
        X = _collinear_data()
        res = mlest(X, backend="cpu", algorithm="direct", force=True)
        assert res.converged is False
        assert any("force=True" in w for w in res._result.warnings)

    def test_em_collinear_raises(self):
        X = _collinear_data()
        with warnings.catch_warnings():
            # The EM ridge regularizer is noisy on degenerate input; the
            # guard still raises after EM completes.
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(SingularMatrixError):
                mlest(X, backend="cpu", algorithm="em")

    def test_collinearity_tol_can_relax_the_guard(self):
        X = _collinear_data()
        # A tolerance below the degenerate floor (~3e-6) lets the fit through
        # without raising.
        res = mlest(X, backend="cpu", algorithm="direct", collinearity_tol=1e-9)
        assert res is not None  # no SingularMatrixError

    def test_full_rank_data_unaffected(self):
        res = mlest(datasets.apple, backend="cpu", algorithm="direct")
        assert res.converged is True
        assert not any(
            "rank-deficient" in w for w in res._result.warnings
        )
