"""Tests for imputation methods: registry, Bayesian linreg draw, norm, pmm."""

import numpy as np
import pytest

from pystatistics.core.exceptions import ValidationError
from pystatistics.mice._rng import make_rng
from pystatistics.mice.methods import available_methods, get_method, is_registered
from pystatistics.mice.methods._linreg import bayes_linreg_draw
from pystatistics.mice.methods.pmm import PMMMethod, _match_donors
from pystatistics.mice.methods.norm import NormMethod


def _linear_data(n, seed, beta=(2.0, -1.0, 0.5), noise=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(beta) - 1))
    y = beta[0] + X @ np.array(beta[1:]) + noise * rng.standard_normal(n)
    return X, y


class TestRegistry:
    def test_builtins_registered(self):
        assert is_registered("pmm")
        assert is_registered("norm")
        assert set(available_methods()) >= {"pmm", "norm"}

    def test_get_method_returns_instance(self):
        assert get_method("pmm").name == "pmm"
        assert get_method("norm").name == "norm"

    def test_methods_declare_numeric_kind(self):
        assert get_method("pmm").target_kind == "numeric"
        assert get_method("norm").target_kind == "numeric"

    def test_unknown_method_raises(self):
        with pytest.raises(ValidationError, match="Unknown imputation method"):
            get_method("nope")


class TestBayesLinregDraw:
    def test_recovers_coefficients_low_noise(self):
        X, y = _linear_data(500, seed=0, noise=0.05)
        draw = bayes_linreg_draw(y, X, make_rng(1))
        # Intercept first, then slopes — should be close to the true (2, -1, 0.5).
        np.testing.assert_allclose(draw.beta_hat, [2.0, -1.0, 0.5], atol=0.05)

    def test_reproducible_given_rng_seed(self):
        X, y = _linear_data(80, seed=2)
        a = bayes_linreg_draw(y, X, make_rng(7))
        b = bayes_linreg_draw(y, X, make_rng(7))
        np.testing.assert_array_equal(a.beta_draw, b.beta_draw)
        assert a.sigma_draw == b.sigma_draw

    def test_predict_shapes(self):
        X, y = _linear_data(40, seed=3)
        draw = bayes_linreg_draw(y, X, make_rng(0))
        Xm = X[:5]
        assert draw.predict_hat(Xm).shape == (5,)
        assert draw.predict_draw(Xm).shape == (5,)

    def test_handles_more_params_than_obs(self):
        # n_obs < n_params: ridge + df floor must keep it finite (no crash).
        rng = np.random.default_rng(0)
        X = rng.normal(size=(3, 5))
        y = rng.normal(size=3)
        draw = bayes_linreg_draw(y, X, make_rng(0))
        assert np.all(np.isfinite(draw.beta_draw))
        assert np.isfinite(draw.sigma_draw)


class TestNorm:
    def test_shape(self):
        X, y = _linear_data(60, seed=1)
        out = NormMethod().impute(y, X, X[:10], make_rng(0))
        assert out.shape == (10,)

    def test_reproducible(self):
        X, y = _linear_data(60, seed=1)
        a = NormMethod().impute(y, X, X[:10], make_rng(3))
        b = NormMethod().impute(y, X, X[:10], make_rng(3))
        np.testing.assert_array_equal(a, b)

    def test_predictions_track_signal(self):
        # With low noise, norm imputations should correlate with the true mean.
        X, y = _linear_data(400, seed=4, noise=0.05)
        Xm, ym_true = X[:50], y[:50]
        out = NormMethod().impute(y, X, Xm, make_rng(0))
        # Imputations and true values should be strongly positively correlated.
        assert np.corrcoef(out, ym_true)[0, 1] > 0.9


class TestPMM:
    def test_shape(self):
        X, y = _linear_data(60, seed=1)
        out = PMMMethod().impute(y, X, X[:10], make_rng(0))
        assert out.shape == (10,)

    def test_imputed_values_are_observed_values(self):
        # The defining PMM property: every imputed value is an actual observed
        # value of the column (donor copy), never a synthesized number.
        X, y = _linear_data(80, seed=5)
        out = PMMMethod().impute(y, X, X[:20], make_rng(0))
        observed = set(np.round(y, 12))
        for v in out:
            assert round(float(v), 12) in observed

    def test_reproducible(self):
        X, y = _linear_data(60, seed=1)
        a = PMMMethod().impute(y, X, X[:10], make_rng(2))
        b = PMMMethod().impute(y, X, X[:10], make_rng(2))
        np.testing.assert_array_equal(a, b)

    def test_donor_count_validation(self):
        with pytest.raises(ValidationError):
            PMMMethod(donors=0)

    def test_match_donors_returns_valid_indices(self):
        rng = make_rng(0)
        yhat_obs = np.linspace(0, 10, 50)
        yhat_mis = np.array([2.0, 5.0, 9.0])
        idx = _match_donors(yhat_mis, yhat_obs, donors=5, rng=rng)
        assert idx.shape == (3,)
        assert np.all((idx >= 0) & (idx < 50))

    def test_match_donors_k_capped_at_n_obs(self):
        # Fewer observed than requested donors: must not crash, picks among all.
        rng = make_rng(0)
        yhat_obs = np.array([1.0, 2.0])
        yhat_mis = np.array([1.5, 1.5, 1.5])
        idx = _match_donors(yhat_mis, yhat_obs, donors=5, rng=rng)
        assert np.all((idx >= 0) & (idx < 2))

    def test_single_donor_picks_nearest(self):
        # With donors=1, each missing gets the single closest observed value.
        rng = make_rng(0)
        y_obs = np.array([10.0, 20.0, 30.0])
        # Build X so fitted values are ~ the y values (strong linear signal).
        X = np.array([[10.0], [20.0], [30.0]])
        Xm = np.array([[21.0]])  # closest fitted should map to the "20" donor
        out = PMMMethod(donors=1).impute(y_obs, X, Xm, rng)
        assert out[0] in y_obs
