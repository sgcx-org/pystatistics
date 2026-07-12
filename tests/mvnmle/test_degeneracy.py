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
    check_observed_variances,
    correlation_min_eigenvalue,
)


def _constant_column_data(seed: int = 0, n: int = 400, const: float = 2.5) -> np.ndarray:
    """Full-rank numeric columns plus one constant (zero-variance) column."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    X = np.column_stack([X, np.full(n, const)])
    X[::13, 0] = np.nan
    return X


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


# ── check_observed_variances (constant-column input guard) ───────────

class TestConstantColumnGuard:
    """A (near-)constant column has zero variance and no interior MLE. The
    scale-invariant fitted-covariance guard cannot see it (it divides each
    variable by its own std, dividing the zero variance away), so it is caught
    at the input boundary."""

    def test_varying_data_returns_none(self):
        rng = np.random.default_rng(0)
        assert check_observed_variances(rng.normal(size=(200, 3))) is None

    def test_constant_column_raises(self):
        data = _constant_column_data()
        with pytest.raises(SingularMatrixError, match="constant"):
            check_observed_variances(data)

    def test_constant_column_force_returns_warning(self):
        data = _constant_column_data()
        msg = check_observed_variances(data, force=True)
        assert msg is not None and "force=True" in msg

    def test_small_but_real_variance_not_flagged(self):
        # A genuinely small-variance column that DOES vary is full-rank — its
        # range is comparable to its own magnitude, unlike a constant column.
        rng = np.random.default_rng(1)
        X = np.column_stack([rng.normal(size=400),
                             1e-6 * rng.normal(size=400) + 3.0])
        assert check_observed_variances(X) is None

    def test_scale_does_not_matter(self):
        # Constant at any magnitude is still constant (range ~0 at any scale).
        for c in (0.0, 1e-9, 2.5, 1e9):
            data = _constant_column_data(const=c)
            with pytest.raises(SingularMatrixError, match="constant"):
                check_observed_variances(data)

    def test_column_with_one_observed_value_flagged(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(200, 3))
        X[1:, 2] = np.nan  # only one observed value in column 2
        with pytest.raises(SingularMatrixError):
            check_observed_variances(X)


class TestMlestConstantColumn:
    """End-to-end: the default mlest(X) fails loud on a constant column instead
    of returning converged=True with a meaningless log-likelihood."""

    def test_default_call_raises(self):
        with pytest.raises(SingularMatrixError, match="constant"):
            mlest(_constant_column_data())

    def test_force_returns_non_converged_with_warning(self):
        res = mlest(_constant_column_data(), force=True)
        assert res.converged is False
        assert any("constant" in w for w in res._result.warnings)

    def test_small_variance_column_not_refused(self):
        # A genuinely small-variance column that varies must NOT be refused by
        # the constant-column guard — mlest returns a fit rather than raising
        # SingularMatrixError. (Its convergence flag may still reflect the
        # conditioning the small variance induces; that is the optimizer's call,
        # not the input guard's.)
        rng = np.random.default_rng(3)
        X = np.column_stack([rng.normal(size=500),
                             1e-6 * rng.normal(size=500) + 1.0])
        X[::11, 0] = np.nan
        res = mlest(X)  # must not raise
        assert res.muhat.shape == (2,)


# ── Integration through mlest ────────────────────────────────────────

class TestMlestDegeneracyGuard:
    """End-to-end behaviour across algorithms and the force escape hatch."""

    def test_direct_collinear_raises(self):
        X = _collinear_data()
        with pytest.raises(SingularMatrixError, match="collinear"):
            mlest(X, backend="cpu", method="direct")

    def test_force_returns_non_converged_with_warning(self):
        X = _collinear_data()
        res = mlest(X, backend="cpu", method="direct", force=True)
        assert res.converged is False
        assert any("force=True" in w for w in res._result.warnings)

    def test_em_collinear_raises(self):
        X = _collinear_data()
        with warnings.catch_warnings():
            # The EM ridge regularizer is noisy on degenerate input; the
            # guard still raises after EM completes.
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(SingularMatrixError):
                mlest(X, backend="cpu", method="em")

    def test_collinearity_tol_can_relax_the_guard(self):
        X = _collinear_data()
        # Demonstrated on the numpy inverse-Cholesky reference, whose degenerate
        # floor is well-characterized at ~3e-6: a tolerance below that floor lets
        # the fit through without raising. (The default forward-Cholesky path
        # drives the collinear direction to ~1e-13 — essentially machine zero and
        # platform-sensitive — so the relaxation window there is not a stable
        # thing to assert on. The default path's *raise* on collinear input is
        # covered by test_direct_collinear_raises.)
        res = mlest(X, solver="reference", method="direct",
                    collinearity_tol=1e-9)
        assert res is not None  # no SingularMatrixError

    def test_full_rank_data_unaffected(self):
        res = mlest(datasets.apple, backend="cpu", method="direct")
        assert res.converged is True
        assert not any(
            "rank-deficient" in w for w in res._result.warnings
        )
