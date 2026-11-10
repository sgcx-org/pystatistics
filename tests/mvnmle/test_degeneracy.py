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
        # Near-collinear but full-rank data: x3 = x1 + x2 + small noise gives a
        # fitted correlation matrix whose smallest eigenvalue (~5e-6) sits just
        # below the default threshold (1e-5) and far above machine zero, so it
        # is stable across platforms. (Exactly collinear input drives that
        # eigenvalue to ~1e-13, which is not a stable thing to assert on.)
        rng = np.random.default_rng(0)
        x1, x2 = rng.standard_normal(400), rng.standard_normal(400)
        X = np.column_stack(
            [x1, x2, x1 + x2 + 0.005 * rng.standard_normal(400)])
        with pytest.raises(SingularMatrixError, match="collinear"):
            mlest(X, backend="cpu", method="direct")
        res = mlest(X, backend="cpu", method="direct", collinearity_tol=1e-9)
        assert np.isfinite(res.loglik)  # no SingularMatrixError

    def test_full_rank_data_unaffected(self):
        res = mlest(datasets.apple, backend="cpu", method="direct")
        assert res.converged is True
        assert not any(
            "rank-deficient" in w for w in res._result.warnings
        )


class TestConstantGuardLargeOffset:
    """Constant-column guard must track fp resolution, not magnitude — 6.0.x.

    A genuinely-varying column with a large offset (1e8 + N(0,1e-3)) was wrongly
    rejected as constant because the old threshold (1e-10 * magnitude = 1e-2 at
    1e8) exceeded the genuine spread (~6e-3). The eps-based noise floor fixes it
    while still catching truly-constant columns at any scale.
    """

    def test_large_offset_varying_column_fits(self):
        import numpy as np
        from pystatistics.mvnmle import mlest
        rng = np.random.default_rng(777)
        col0 = 1e8 + rng.standard_normal(150) * 1e-3
        col1 = rng.standard_normal(150)
        X = np.column_stack([col0, col1])
        res = mlest(X)  # must NOT raise
        assert res.converged
        assert res.sigmahat[0, 0] == pytest.approx(np.var(col0), rel=1e-3)

    @pytest.mark.parametrize("const", [3.0, 1e8, 1e-6])
    def test_true_constant_still_rejected_at_any_scale(self, const):
        import numpy as np
        from pystatistics.mvnmle import mlest
        from pystatistics.core.exceptions import SingularMatrixError
        rng = np.random.default_rng(1)
        X = np.column_stack([np.full(150, const), rng.standard_normal(150)])
        with pytest.raises(SingularMatrixError):
            mlest(X)


class TestNonFiniteLoglik:
    """A fit whose log-likelihood cannot be evaluated at the reported optimum
    must not be reported as converged (it used to surface as a spurious,
    enormous log-likelihood with converged=True)."""

    def test_non_finite_loglik_flags_not_converged(self):
        from pystatistics.mvnmle.backends._direct import run_direct_solve

        class _Objective:
            """Quadratic bowl whose reported log-likelihood is not finite."""
            n_observed_scalars = 10

            def get_initial_parameters(self):
                return np.array([1.0, -1.0])

            def compute_objective(self, theta):
                return float(theta @ theta)

            def compute_gradient(self, theta):
                return 2.0 * theta

            def extract_parameters(self, theta):
                return theta[:1], np.eye(1), float('nan')

        res = run_direct_solve(
            _Objective, method='BFGS', tol=1e-8, max_iter=50,
            backend_name='stub', parameterization='cholesky')
        assert not res.params.converged
        assert np.isnan(res.params.loglik)
        assert any("not finite" in w for w in res.warnings)

    def test_scale_disparate_default_path_reports_finite_loglik(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((4, 4))
        L = np.linalg.cholesky(A @ A.T + 4 * np.eye(4))
        Z = rng.standard_normal((300, 4)) @ L.T
        Z[:, 0] *= 1e6
        Z[:, 3] *= 1e-6
        m = rng.random((300, 4)) < 0.10
        Z[m] = np.nan
        Z[np.all(np.isnan(Z), 1), 0] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = mlest(Z)
        assert np.isfinite(res.loglik)
        # Either a genuine optimum or an honest non-convergence report.
        assert res.converged or any(
            "did not converge" in w for w in res.warnings)


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available() or torch.backends.mps.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_available(), reason="no GPU (CUDA/MPS)")
class TestFp32NoSilentWrongGate:
    """fp32 GPU must not silently accept a fit fp64 refuses — 6.0.x (R12).

    On near-collinear data the fp32 optimizer stalls at a non-stationary point
    whose fp32 fitted covariance looks full-rank, so the post-fit guard passed
    and a fit ~1000 nats below the optimum was accepted with converged=True,
    while the fp64 CPU path refused. A scale-invariant collinearity screen on
    the RAW sample covariance now refuses it at GPU init (safe side).
    """

    def test_near_collinear_refused_on_gpu(self):
        import numpy as np
        from pystatistics.mvnmle import mlest
        from pystatistics.core.exceptions import NumericalError, SingularMatrixError
        rng = np.random.default_rng(20260723)
        p, n = 5, 200
        Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
        cov = (Q * np.logspace(0, -8, p)) @ Q.T
        X = rng.multivariate_normal(rng.standard_normal(p) * 2, cov, size=n)
        with pytest.raises(SingularMatrixError):
            mlest(X, backend="cpu")
        with pytest.raises((NumericalError, SingularMatrixError)):
            mlest(X, backend="gpu")

    def test_well_conditioned_still_fits_on_gpu(self):
        import numpy as np
        from pystatistics.mvnmle import mlest
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 4))
        C = A @ A.T + 4 * np.eye(4)
        Y = rng.multivariate_normal(np.zeros(4), C, size=500)
        mm = rng.random((500, 4)) < 0.1
        Y[mm] = np.nan
        Y[np.all(np.isnan(Y), 1), 0] = 0.0
        assert mlest(Y, backend="gpu").converged  # no false refusal


# ── Never-jointly-observed pair guard ────────────────────────────────

def _disjoint_pair_data(seed: int = 0, n: int = 200) -> np.ndarray:
    """Vars 1 and 2 are never observed in the same row (var 0 always seen).

    The likelihood is exactly flat in sigma[1, 2]: that entry is
    unidentified and any fitted value for it is a starting-point artifact.
    """
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.7], [0.5, 0.7, 1.0]])
    X = rng.multivariate_normal(np.zeros(3), cov, size=n)
    X[: n // 2, 1] = np.nan   # first half: var 1 missing
    X[n // 2:, 2] = np.nan    # second half: var 2 missing
    return X


class TestPairwiseObservationGuard:
    """Unidentified covariance entries must fail loud, not report converged."""

    def test_checker_flags_the_pair(self):
        from pystatistics.mvnmle._degeneracy import check_pairwise_observation
        with pytest.raises(SingularMatrixError, match=r"\(1, 2\)"):
            check_pairwise_observation(_disjoint_pair_data())

    def test_checker_passes_jointly_observed_data(self):
        from pystatistics.mvnmle._degeneracy import check_pairwise_observation
        rng = np.random.default_rng(1)
        X = rng.normal(size=(100, 4))
        X[::7, 2] = np.nan
        assert check_pairwise_observation(X) is None

    def test_mlest_direct_raises(self):
        with pytest.raises(SingularMatrixError, match="never observed"):
            mlest(_disjoint_pair_data())

    def test_mlest_em_raises(self):
        with pytest.raises(SingularMatrixError, match="never observed"):
            mlest(_disjoint_pair_data(), method='em')

    def test_force_returns_non_converged_with_warning(self):
        res = mlest(_disjoint_pair_data(), force=True)
        assert res.converged is False
        assert any("never observed" in w for w in res._result.warnings)

    def test_complete_data_unaffected(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(150, 3))
        X[::9, 0] = np.nan
        res = mlest(X)
        assert res.converged is True
