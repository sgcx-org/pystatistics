"""
Tests for jackknife influence values.

Used by BCa confidence intervals. Validates against known results.
"""

import numpy as np
import pytest

from pystatistics.montecarlo import boot
from pystatistics.montecarlo._influence import jackknife_influence


def mean_stat(data, indices):
    """Bootstrap statistic: sample mean."""
    return np.array([np.mean(data[indices])])


class TestJackknifeInfluence:
    """Tests for jackknife influence values."""

    def test_influence_shape(self):
        """Influence values have shape (n,)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = boot(data, mean_stat, n_resamples=100, seed=42)
        L = jackknife_influence(result, stat_index=0)
        assert L.shape == (5,)

    def test_influence_sum_zero(self):
        """Influence values should sum to approximately zero."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = boot(data, mean_stat, n_resamples=100, seed=42)
        L = jackknife_influence(result, stat_index=0)

        # Sum of influence values should be close to zero
        assert abs(np.sum(L)) < 1e-10

    def test_influence_mean_known(self):
        """For the mean, influence values are proportional to (x_i - x_bar)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        n = len(data)
        result = boot(data, mean_stat, n_resamples=100, seed=42)
        L = jackknife_influence(result, stat_index=0)

        # For the mean: theta_{-i} = (n*x_bar - x_i) / (n-1)
        # mean_jack = mean of theta_{-i} = x_bar
        # L_i = (n-1) * (x_bar - theta_{-i}) = x_i - x_bar
        x_bar = np.mean(data)
        expected_L = data - x_bar

        np.testing.assert_allclose(L, expected_L, rtol=1e-10)

    def test_influence_symmetric_data(self):
        """Symmetric data gives antisymmetric influence values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = boot(data, mean_stat, n_resamples=100, seed=42)
        L = jackknife_influence(result, stat_index=0)

        # L[0] should equal -L[4], L[1] should equal -L[3]
        assert L[0] == pytest.approx(-L[4], rel=1e-10)
        assert L[1] == pytest.approx(-L[3], rel=1e-10)

    def test_acceleration_parameter(self):
        """BCa acceleration parameter from influence values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = boot(data, mean_stat, n_resamples=100, seed=42)
        L = jackknife_influence(result, stat_index=0)

        # Acceleration: a = sum(L^3) / (6 * sum(L^2)^1.5)
        a = np.sum(L ** 3) / (6.0 * np.sum(L ** 2) ** 1.5)

        # For symmetric data, a should be close to 0
        assert abs(a) < 0.01

    def test_skewed_data_nonzero_acceleration(self):
        """Skewed data gives nonzero acceleration parameter."""
        # Exponential-like data (right-skewed)
        data = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
        result = boot(data, mean_stat, n_resamples=100, seed=42)
        L = jackknife_influence(result, stat_index=0)

        a = np.sum(L ** 3) / (6.0 * np.sum(L ** 2) ** 1.5)

        # Should be nonzero for skewed data
        assert abs(a) > 0.01


class TestRegressionInfluenceNonOrdinary:
    """A7: regression empinf now applies to balanced + stratified bootstrap
    (was jackknife fallback); parametric still (correctly) falls back."""

    @staticmethod
    def _var_stat(data, indices):
        return np.array([np.var(data[indices], ddof=1)])

    @staticmethod
    def _accel(L):
        s2 = np.sum(L ** 2)
        return np.sum(L ** 3) / (6 * s2 ** 1.5) if s2 > 0 else 0.0

    def _data(self):
        rng = np.random.default_rng(1)
        return rng.gamma(2.0, 1.0, 40)

    def test_balanced_uses_regression_influence(self):
        from pystatistics.montecarlo._influence import regression_influence
        x = self._data()
        b = boot(x, self._var_stat, n_resamples=3000, method="balanced",
                 seed=7, statistic_type="index")
        L = regression_influence(b)
        assert L is not None                      # no longer falls back
        assert L.shape == (len(x),)
        assert np.all(np.isfinite(L))
        # centred to (approximately) sum zero — the balanced frequency matrix is
        # near-degenerate, so pinv leaves a small numerical residual.
        assert abs(np.sum(L)) < 1e-3

    def test_balanced_accel_matches_ordinary(self):
        """The balanced reg-empinf estimates the same influence as the ordinary
        one (both converge to the true empirical influence)."""
        from pystatistics.montecarlo._influence import regression_influence
        x = self._data()
        bo = boot(x, self._var_stat, n_resamples=4000, method="ordinary",
                  seed=7, statistic_type="index")
        bb = boot(x, self._var_stat, n_resamples=4000, method="balanced",
                  seed=7, statistic_type="index")
        a_ord = self._accel(regression_influence(bo))
        a_bal = self._accel(regression_influence(bb))
        assert abs(a_ord - a_bal) < 5e-3

    def test_stratified_uses_regression_influence_centred_within_strata(self):
        from pystatistics.montecarlo._influence import regression_influence
        x = self._data()
        strata = np.repeat([0, 1], 20)
        b = boot(x, self._var_stat, n_resamples=3000, method="ordinary",
                 seed=7, statistic_type="index", strata=strata)
        L = regression_influence(b)
        assert L is not None
        # centred within each stratum (R's empinf for a stratified boot object);
        # the near-degenerate frequency matrix leaves a small pinv residual.
        assert abs(np.sum(L[strata == 0])) < 1e-3
        assert abs(np.sum(L[strata == 1])) < 1e-3

    def test_parametric_falls_back_to_jackknife(self):
        """Parametric bootstrap has no resample frequencies → regression empinf
        does not apply (documented B); the caller uses the jackknife."""
        from pystatistics.montecarlo._influence import regression_influence
        x = self._data()

        def ran_gen(data, mle, rng):
            return rng.normal(mle[0], mle[1], len(data))

        b = boot(x, lambda d: np.array([np.var(d, ddof=1)]),
                 n_resamples=500, method="parametric", seed=7,
                 ran_gen=ran_gen, mle=(float(x.mean()), float(x.std())))
        assert regression_influence(b) is None
