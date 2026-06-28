"""
Tests for the single-pass, rank-revealing QR least-squares solver
(`pystatistics.core.compute.linalg.qr.qr_solve`).

Covers normal (full-rank), edge (rank-deficient, ill-conditioned-but-full-rank),
and failure cases, plus a speed sanity check that the solver stays at or below
NumPy's own `lstsq` on a large design (a proxy for R parity; the dedicated R
benchmark lives in the dev/validation harness, not the unit suite).
"""

import time

import numpy as np
import pytest

from pystatistics.core.compute.linalg.qr import qr_solve, QRResult


def _coef_full(coef):
    """Aliased (NaN) coefficients count as 0 for fitted-value comparison."""
    return np.where(np.isnan(coef), 0.0, coef)


def _rss(X, y, coef):
    r = y - X @ _coef_full(coef)
    return float(r @ r)


# ---------------------------------------------------------------------------
# Normal case: full-rank
# ---------------------------------------------------------------------------

class TestFullRank:
    def test_matches_numpy_lstsq(self):
        rng = np.random.default_rng(0)
        n, p = 500, 8
        X = rng.standard_normal((n, p))
        X[:, 0] = 1.0
        beta = rng.standard_normal(p)
        y = X @ beta + rng.standard_normal(n) * 0.1

        coef, qr = qr_solve(X, y)
        ref, *_ = np.linalg.lstsq(X, y, rcond=None)

        assert qr.rank == p
        assert qr.Q is None
        np.testing.assert_allclose(coef, ref, rtol=1e-10, atol=1e-10)
        assert not np.isnan(coef).any()

    def test_pivot_is_identity_when_full_rank(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 5))
        y = rng.standard_normal(200)
        _, qr = qr_solve(X, y)
        np.testing.assert_array_equal(qr.pivot, np.arange(5))
        assert qr.R.shape == (5, 5)

    def test_R_factor_reproduces_normal_equations(self):
        # R[:p,:p] from the solver must satisfy RᵀR = XᵀX (the Gram matrix),
        # which is what the SE code relies on.
        rng = np.random.default_rng(2)
        X = rng.standard_normal((300, 6))
        y = rng.standard_normal(300)
        _, qr = qr_solve(X, y)
        R = qr.R[:qr.rank, :qr.rank]
        np.testing.assert_allclose(R.T @ R, X.T @ X, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# Edge case: rank-deficient (exact collinearity)
# ---------------------------------------------------------------------------

class TestRankDeficient:
    def test_detects_rank_and_aliases_dependent_column(self):
        rng = np.random.default_rng(3)
        n, p = 400, 5
        X = rng.standard_normal((n, p))
        X[:, 0] = 1.0
        X[:, 4] = 2.0 * X[:, 1] - 3.0 * X[:, 2]  # exact dependency on col 4
        y = rng.standard_normal(n)

        coef, qr = qr_solve(X, y)

        assert qr.rank == p - 1
        # the dependent column is aliased to NaN; the rest are finite
        assert np.isnan(coef[4])
        assert np.isfinite(coef[[0, 1, 2, 3]]).all()

    def test_dependent_column_in_the_middle(self):
        # The aliased column need not be last: a middle dependent column must be
        # detected and the surrounding independent columns solved.
        rng = np.random.default_rng(4)
        n, p = 400, 6
        X = rng.standard_normal((n, p))
        X[:, 0] = 1.0
        X[:, 3] = X[:, 1] + X[:, 2]  # col 3 dependent
        y = rng.standard_normal(n)

        coef, qr = qr_solve(X, y)
        assert qr.rank == p - 1
        assert np.isnan(coef[3])
        # pivot: independent block first, dependent column last
        assert qr.pivot[-1] == 3
        np.testing.assert_array_equal(np.sort(qr.pivot), np.arange(p))

    def test_fit_equivalent_to_dropping_dependent_column(self):
        # The fitted values / RSS must equal the fit on the reduced design.
        rng = np.random.default_rng(5)
        n, p = 600, 5
        X = rng.standard_normal((n, p))
        X[:, 0] = 1.0
        X[:, 4] = X[:, 0] - X[:, 1]
        y = rng.standard_normal(n)

        coef, qr = qr_solve(X, y)
        rss_full = _rss(X, y, coef)

        X_reduced = X[:, [0, 1, 2, 3]]
        ref, *_ = np.linalg.lstsq(X_reduced, y, rcond=None)
        rss_reduced = float(((y - X_reduced @ ref) ** 2).sum())

        assert qr.rank == 4
        np.testing.assert_allclose(rss_full, rss_reduced, rtol=1e-8)

    def test_se_layout_independent_block_leads(self):
        # R[:rank,:rank] must be the factor of the independent columns and
        # pivot[:rank] their original indices (the contract the SE code uses).
        rng = np.random.default_rng(6)
        n, p = 500, 5
        X = rng.standard_normal((n, p))
        X[:, 0] = 1.0
        X[:, 2] = X[:, 1]  # col 2 dependent
        y = rng.standard_normal(n)

        _, qr = qr_solve(X, y)
        rank = qr.rank
        ind = qr.pivot[:rank]
        R_block = qr.R[:rank, :rank]
        Xi = X[:, ind]
        np.testing.assert_allclose(R_block.T @ R_block, Xi.T @ Xi,
                                   rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# Edge case: ill-conditioned but full rank (must NOT be flagged deficient)
# ---------------------------------------------------------------------------

class TestIllConditioned:
    def test_near_collinear_stays_full_rank(self):
        rng = np.random.default_rng(7)
        n, p = 800, 4
        X = rng.standard_normal((n, p))
        X[:, 0] = 1.0
        # col 3 nearly equals col 1 but with a small independent perturbation —
        # full rank, ill-conditioned, must not be aliased.
        X[:, 3] = X[:, 1] + 1e-4 * rng.standard_normal(n)
        beta = rng.standard_normal(p)
        y = X @ beta + rng.standard_normal(n) * 1e-3

        coef, qr = qr_solve(X, y)
        assert qr.rank == p
        assert np.isfinite(coef).all()
        ref, *_ = np.linalg.lstsq(X, y, rcond=None)
        np.testing.assert_allclose(coef, ref, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Failure cases (Bible Rule 1: fail loud)
# ---------------------------------------------------------------------------

class TestFailureCases:
    def test_x_not_2d_raises(self):
        with pytest.raises(ValueError, match="X must be 2-D"):
            qr_solve(np.ones(10), np.ones(10))

    def test_y_not_1d_raises(self):
        with pytest.raises(ValueError, match="y must be 1-D"):
            qr_solve(np.ones((10, 3)), np.ones((10, 1)))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="rows but y has"):
            qr_solve(np.ones((10, 3)), np.ones(9))


# ---------------------------------------------------------------------------
# Degenerate shape: n < p (fallback path)
# ---------------------------------------------------------------------------

class TestUnderdetermined:
    def test_n_less_than_p_uses_pivoted_fallback(self):
        rng = np.random.default_rng(8)
        X = rng.standard_normal((4, 7))
        y = rng.standard_normal(4)
        coef, qr = qr_solve(X, y)
        # rank capped at n; fit reproduces y (underdetermined → zero residual)
        assert qr.rank == 4
        np.testing.assert_allclose(X @ _coef_full(coef), y, atol=1e-8)


# ---------------------------------------------------------------------------
# Speed sanity (proxy for R parity; not a hard CI gate on absolute ms)
# ---------------------------------------------------------------------------

class TestSpeed:
    def test_not_slower_than_numpy_lstsq_on_large_design(self):
        rng = np.random.default_rng(9)
        n, p = 200_000, 40
        X = rng.standard_normal((n, p))
        X[:, 0] = 1.0
        beta = rng.standard_normal(p)
        y = X @ beta + rng.standard_normal(n) * 0.5

        # warmup
        qr_solve(X, y)
        np.linalg.lstsq(X, y, rcond=None)

        def best(fn, k=3):
            t = []
            for _ in range(k):
                t0 = time.perf_counter()
                fn()
                t.append(time.perf_counter() - t0)
            return min(t)

        t_qr = best(lambda: qr_solve(X, y))
        t_np = best(lambda: np.linalg.lstsq(X, y, rcond=None))

        # qr_solve avoids forming Q and the SVD lstsq does, so it should be at
        # least competitive. Generous 1.5x margin to avoid CI flakiness.
        assert t_qr <= 1.5 * t_np, (
            f"qr_solve {t_qr*1e3:.0f}ms vs lstsq {t_np*1e3:.0f}ms"
        )
