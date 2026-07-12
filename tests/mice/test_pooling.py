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
        assert res.standard_errors == pytest.approx(np.sqrt(3.5))
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
        assert res.ci_lower < res.estimate < res.ci_upper

    def test_higher_conf_level_widens_ci(self):
        narrow = pool([1.0, 2.0, 3.0], [0.4, 0.5, 0.6], conf_level=0.90)
        wide = pool([1.0, 2.0, 3.0], [0.4, 0.5, 0.6], conf_level=0.99)
        assert (wide.ci_upper - wide.ci_lower) > (narrow.ci_upper - narrow.ci_lower)

    def test_conf_int_property_shape_and_values(self):
        # Scalar pooling -> (1, 2); rows are [lower, upper].
        res = pool([1.0, 2.0, 3.0], [0.4, 0.5, 0.6])
        ci = res.conf_int
        assert ci.shape == (1, 2)
        assert ci[0, 0] == pytest.approx(res.ci_lower)
        assert ci[0, 1] == pytest.approx(res.ci_upper)

    def test_conf_level_095_matches_old_alpha_005(self):
        # The conf_level convention (0.95) must reproduce the CI bounds the old
        # alpha=0.05 significance-level convention produced: both are 95% CIs.
        from scipy import stats

        q, u = [1.0, 2.0, 3.0], [0.4, 0.5, 0.6]
        res = pool(q, u, conf_level=0.95)
        # Recompute the old alpha=0.05 bounds directly: tcrit = t.ppf(1 - 0.05/2, df).
        tcrit = stats.t.ppf(1.0 - 0.05 / 2.0, res.df)
        expected_lo = res.estimate - tcrit * res.standard_errors
        expected_hi = res.estimate + tcrit * res.standard_errors
        assert res.ci_lower == pytest.approx(expected_lo)
        assert res.ci_upper == pytest.approx(expected_hi)


class TestBarnardRubinDf:
    def test_finite_dfcom_reduces_df(self):
        # With finite complete-data df, the pooled df must not exceed dfcom and
        # is smaller than the dfcom->inf (classic Rubin) value.
        q, u = [1.0, 3.0, 2.0], [0.5, 0.5, 0.5]
        inf_df = pool(q, u).df
        finite = pool(q, u, df_complete=10.0)
        assert finite.df < inf_df
        assert finite.df <= 10.0

    def test_large_dfcom_approaches_classic(self):
        q, u = [1.0, 3.0, 2.0], [0.5, 0.5, 0.5]
        classic = pool(q, u).df
        big = pool(q, u, df_complete=1e8).df
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
        np.testing.assert_allclose(
            vec.standard_errors, [s0.standard_errors, s1.standard_errors]
        )
        np.testing.assert_allclose(vec.df, [s0.df, s1.df])


class TestSingleImputation:
    def test_m1_degenerate(self):
        res = pool([4.0], [2.0])
        assert res.estimate == pytest.approx(4.0)
        assert res.between == pytest.approx(0.0)
        assert res.total == pytest.approx(2.0)
        assert res.n_imputations == 1


class TestPoolingValidation:
    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValidationError):
            pool([1.0, 2.0], [0.5, 0.5, 0.5])

    def test_negative_variance_rejected(self):
        with pytest.raises(ValidationError):
            pool([1.0, 2.0], [0.5, -0.1])

    @pytest.mark.parametrize("bad_conf_level", [0.0, 1.0, -0.1, 2.0])
    def test_bad_conf_level_rejected(self, bad_conf_level):
        with pytest.raises(ValidationError):
            pool([1.0, 2.0], [0.5, 0.5], conf_level=bad_conf_level)

    def test_bad_df_complete_rejected(self):
        with pytest.raises(ValidationError):
            pool([1.0, 2.0], [0.5, 0.5], df_complete=-5.0)

    def test_3d_rejected(self):
        with pytest.raises(ValidationError):
            pool(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))


class TestEndToEndWithMice:
    def test_pool_regression_slope(self):
        # Fit OLS slope of col 0 on col 1 in each completed dataset, then pool.
        complete = datasets.make_gaussian_complete(120, seed=11)
        miss = datasets.make_mcar(complete, 0.2, seed=12)
        sol = mice(miss, n_imputations=8, max_iter=8, method="norm", seed=13)

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

        res = pool(slopes, variances, df_complete=120 - 2)
        # True slope = cov(0,1)/var(1) = 0.6 / 1.0 = 0.6 for the default cov.
        assert res.ci_lower < 0.6 < res.ci_upper
        assert 0.0 <= res.fmi <= 1.0
        assert res.n_imputations == 8
