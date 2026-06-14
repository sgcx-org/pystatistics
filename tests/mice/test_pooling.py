"""Tests for Rubin's-rules pooling."""

import numpy as np
import pytest

from pystatistics.core.exceptions import ValidationError
from pystatistics.mice import datasets, mice
from pystatistics.mice.pooling import pool


class TestRubinHandComputed:
    def test_two_imputations_scalar(self):
        # Q = [1, 3], U = [0.5, 0.5], m = 2.
        # Qbar = 2; Ubar = 0.5; B = var([1,3], ddof=1) = 2.0
        # T = 0.5 + 1.5 * 2 = 3.5
        res = pool([1.0, 3.0], [0.5, 0.5])
        assert res.estimate == pytest.approx(2.0)
        assert res.within == pytest.approx(0.5)
        assert res.between == pytest.approx(2.0)
        assert res.total == pytest.approx(3.5)
        assert res.se == pytest.approx(np.sqrt(3.5))
        assert res.riv == pytest.approx(6.0)
        assert res.lambda_ == pytest.approx(3.0 / 3.5)
        # df_old = (m-1)/lambda^2 = 1 / (3/3.5)^2
        assert res.df == pytest.approx(1.0 / (3.0 / 3.5) ** 2)

    def test_estimate_is_mean(self):
        q = [2.0, 4.0, 6.0, 8.0]
        res = pool(q, [1.0, 1.0, 1.0, 1.0])
        assert res.estimate == pytest.approx(np.mean(q))

    def test_identical_estimates_zero_between(self):
        res = pool([5.0, 5.0, 5.0], [2.0, 2.0, 2.0])
        assert res.between == pytest.approx(0.0)
        assert res.total == pytest.approx(2.0)  # T = Ubar when B = 0
        assert res.riv == pytest.approx(0.0)
        assert res.lambda_ == pytest.approx(0.0)


class TestCI:
    def test_ci_brackets_estimate(self):
        res = pool([1.0, 2.0, 3.0], [0.4, 0.5, 0.6])
        assert res.ci_low < res.estimate < res.ci_high

    def test_alpha_widens_ci(self):
        narrow = pool([1.0, 2.0, 3.0], [0.4, 0.5, 0.6], alpha=0.10)
        wide = pool([1.0, 2.0, 3.0], [0.4, 0.5, 0.6], alpha=0.01)
        assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)


class TestBarnardRubinDf:
    def test_finite_dfcom_reduces_df(self):
        # With finite complete-data df, the pooled df must not exceed dfcom and
        # is smaller than the dfcom->inf (classic Rubin) value.
        q, u = [1.0, 3.0, 2.0], [0.5, 0.5, 0.5]
        inf_df = pool(q, u).df
        finite = pool(q, u, dfcom=10.0)
        assert finite.df < inf_df
        assert finite.df <= 10.0

    def test_large_dfcom_approaches_classic(self):
        q, u = [1.0, 3.0, 2.0], [0.5, 0.5, 0.5]
        classic = pool(q, u).df
        big = pool(q, u, dfcom=1e8).df
        assert big == pytest.approx(classic, rel=1e-3)


class TestVectorPooling:
    def test_pool_multiple_parameters(self):
        # Pool 2 parameters across 3 imputations simultaneously.
        Q = np.array([[1.0, 10.0], [2.0, 12.0], [3.0, 11.0]])
        U = np.full((3, 2), 0.5)
        res = pool(Q, U)
        assert np.asarray(res.estimate).shape == (2,)
        np.testing.assert_allclose(res.estimate, [2.0, 11.0])

    def test_vector_matches_scalar(self):
        # Pooling 2 params together == pooling each alone.
        Q = np.array([[1.0, 10.0], [2.0, 12.0], [3.0, 11.0]])
        U = np.full((3, 2), 0.5)
        vec = pool(Q, U)
        s0 = pool(Q[:, 0], U[:, 0])
        s1 = pool(Q[:, 1], U[:, 1])
        np.testing.assert_allclose(vec.se, [s0.se, s1.se])
        np.testing.assert_allclose(vec.df, [s0.df, s1.df])


class TestSingleImputation:
    def test_m1_degenerate(self):
        res = pool([4.0], [2.0])
        assert res.estimate == pytest.approx(4.0)
        assert res.between == pytest.approx(0.0)
        assert res.total == pytest.approx(2.0)
        assert res.m == 1


class TestPoolingValidation:
    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValidationError):
            pool([1.0, 2.0], [0.5, 0.5, 0.5])

    def test_negative_variance_rejected(self):
        with pytest.raises(ValidationError):
            pool([1.0, 2.0], [0.5, -0.1])

    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 2.0])
    def test_bad_alpha_rejected(self, bad_alpha):
        with pytest.raises(ValidationError):
            pool([1.0, 2.0], [0.5, 0.5], alpha=bad_alpha)

    def test_bad_dfcom_rejected(self):
        with pytest.raises(ValidationError):
            pool([1.0, 2.0], [0.5, 0.5], dfcom=-5.0)

    def test_3d_rejected(self):
        with pytest.raises(ValidationError):
            pool(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))


class TestEndToEndWithMice:
    def test_pool_regression_slope(self):
        # Fit OLS slope of col 0 on col 1 in each completed dataset, then pool.
        complete = datasets.make_gaussian_complete(120, seed=11)
        miss = datasets.make_mcar(complete, 0.2, seed=12)
        sol = mice(miss, m=8, maxit=8, method="norm", seed=13)

        slopes, variances = [], []
        for d in sol.completed_datasets():
            x = np.column_stack([np.ones(d.shape[0]), d[:, 1]])
            y = d[:, 0]
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            resid = y - x @ beta
            sigma2 = resid @ resid / (d.shape[0] - 2)
            cov = sigma2 * np.linalg.inv(x.T @ x)
            slopes.append(beta[1])
            variances.append(cov[1, 1])

        res = pool(slopes, variances, dfcom=120 - 2)
        # True slope = cov(0,1)/var(1) = 0.6 / 1.0 = 0.6 for the default cov.
        assert res.ci_low < 0.6 < res.ci_high
        assert 0.0 <= res.fmi <= 1.0
        assert res.m == 8
