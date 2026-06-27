"""
Tests for categorical imputation: logreg, polyreg, polr, encoding, and
mixed-type integration through mice().
"""

import warnings

import numpy as np
import pytest

from pystatistics.core.exceptions import ValidationError
from pystatistics.mice import datasets, mice, MICEDesign
from pystatistics.mice._encode import (
    build_predictor_matrix,
    codes_to_indices,
    indices_to_codes,
)
from pystatistics.mice._rng import make_rng
from pystatistics.mice.methods import get_method
from pystatistics.mice.methods.logreg import LogregMethod, _fit_logistic
from pystatistics.mice.methods.polyreg import PolyregMethod
from pystatistics.mice.methods.polr import PolrMethod


# --------------------------------------------------------------------- helpers
def _make_mixed(n, seed):
    """Numeric predictor + binary, 3-level nominal, 4-level ordinal targets."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.4 + 1.3 * x)))
    binary = (rng.random(n) < p).astype(float)
    eta = np.column_stack([0.9 * x, -0.7 * x, np.zeros(n)])
    pr = np.exp(eta)
    pr /= pr.sum(1, keepdims=True)
    nominal = np.array([rng.choice(3, p=pr[i]) for i in range(n)], dtype=float)
    ordinal = np.clip(np.round(1.5 + 0.9 * x + rng.normal(scale=0.6, size=n)), 0, 3)
    data = np.column_stack([x, binary, nominal, ordinal])
    kinds = ["numeric", "binary", "categorical", "ordered"]
    return data, kinds


# ----------------------------------------------------------------- registration
class TestRegistration:
    def test_categorical_methods_registered(self):
        assert get_method("logreg").target_kind == "binary"
        assert get_method("polyreg").target_kind == "categorical"
        assert get_method("polr").target_kind == "ordered"


# --------------------------------------------------------------------- encoding
class TestEncoding:
    def test_code_index_roundtrip(self):
        levels = np.array([0.0, 2.0, 5.0])
        codes = np.array([5.0, 0.0, 2.0, 5.0])
        idx = codes_to_indices(codes, levels)
        np.testing.assert_array_equal(idx, [2, 0, 1, 2])
        np.testing.assert_array_equal(indices_to_codes(idx, levels), codes)

    def test_categorical_predictor_is_dummy_encoded(self):
        # Column 1 categorical with 3 levels -> 2 dummy columns; numeric col 0
        # passes through. Predict for target column 2 (also numeric here).
        data = np.array(
            [[1.0, 0.0, 9.0], [2.0, 1.0, 9.0], [3.0, 2.0, 9.0], [4.0, 0.0, 9.0]]
        )
        design = MICEDesign.from_array(
            np.where(np.arange(12).reshape(4, 3) == 11, np.nan, data),
            column_kinds=["numeric", "categorical", "numeric"],
        )
        X = build_predictor_matrix(design.data, 2, design)
        # numeric col0 (1 col) + categorical col1 (3 levels -> 2 dummies) = 3 cols
        assert X.shape == (4, 3)
        # Row with level 0 -> both dummies zero (reference).
        assert X[0, 1] == 0.0 and X[0, 2] == 0.0

    def test_numeric_only_uses_fast_path(self):
        # All-numeric design returns a plain slice (no dummy expansion).
        d = MICEDesign.from_array(datasets.EXAMPLE)
        X = build_predictor_matrix(d.data, 0, d)
        assert X.shape == (12, 2)


# ----------------------------------------------------------------------- logreg
class TestLogreg:
    def _xy(self, n, seed):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 2))
        p = 1.0 / (1.0 + np.exp(-(0.5 + 1.5 * X[:, 0] - 1.0 * X[:, 1])))
        y = (rng.random(n) < p).astype(float)
        return X, y

    def test_returns_binary_indices(self):
        X, y = self._xy(200, 0)
        out = LogregMethod().impute(y, X, X[:30], make_rng(0))
        assert set(np.unique(out)).issubset({0, 1})
        assert out.shape == (30,)

    def test_deterministic(self):
        X, y = self._xy(200, 1)
        a = LogregMethod().impute(y, X, X[:30], make_rng(3))
        b = LogregMethod().impute(y, X, X[:30], make_rng(3))
        np.testing.assert_array_equal(a, b)

    def test_tracks_signal(self):
        # Strong signal: imputed 1-rate should be higher where true p is higher.
        X, y = self._xy(800, 2)
        order = np.argsort(X[:, 0])
        low, high = order[:100], order[-100:]
        rng = make_rng(0)
        out_low = LogregMethod().impute(y, X, X[low], rng)
        out_high = LogregMethod().impute(y, X, X[high], rng)
        assert out_high.mean() > out_low.mean()

    def test_fit_logistic_recovers_sign(self):
        X, y = self._xy(1000, 3)
        beta, cov = _fit_logistic(y, X)
        assert beta[1] > 0   # positive effect of X[:,0]
        assert beta[2] < 0   # negative effect of X[:,1]
        assert cov.shape == (3, 3)

    def test_separable_data_does_not_crash(self):
        # Perfectly separable -> unpenalised MLE diverges; ridge must keep finite.
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        out = LogregMethod().impute(y, X, X, make_rng(0))
        assert set(np.unique(out)).issubset({0, 1})


# ---------------------------------------------------------------------- polyreg
class TestPolyreg:
    def _xy(self, n, seed, K=3):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 2))
        eta = np.column_stack(
            [X @ rng.normal(size=2) for _ in range(K - 1)] + [np.zeros(n)]
        )
        pr = np.exp(eta)
        pr /= pr.sum(1, keepdims=True)
        y = np.array([rng.choice(K, p=pr[i]) for i in range(n)], dtype=float)
        return X, y

    def test_returns_valid_indices(self):
        X, y = self._xy(300, 0)
        out = PolyregMethod().impute(y, X, X[:40], make_rng(0))
        assert set(np.unique(out)).issubset({0, 1, 2})
        assert out.shape == (40,)

    def test_deterministic(self):
        X, y = self._xy(300, 1)
        a = PolyregMethod().impute(y, X, X[:40], make_rng(5))
        b = PolyregMethod().impute(y, X, X[:40], make_rng(5))
        np.testing.assert_array_equal(a, b)

    def test_marginal_proportions_reasonable(self):
        # Imputed category proportions should be in the right ballpark of the
        # observed proportions (same DGP).
        X, y = self._xy(800, 2)
        out = PolyregMethod().impute(y, X, X, make_rng(0))
        obs_prop = np.bincount(y.astype(int), minlength=3) / len(y)
        imp_prop = np.bincount(out, minlength=3) / len(out)
        np.testing.assert_allclose(imp_prop, obs_prop, atol=0.12)


# ------------------------------------------------------------------------- polr
class TestPolr:
    def _xy(self, n, seed, K=4):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 2))
        latent = X[:, 0] * 1.0 - X[:, 1] * 0.5 + rng.logistic(size=n)
        cuts = np.quantile(latent, np.linspace(0, 1, K + 1)[1:-1])
        y = np.digitize(latent, cuts).astype(float)
        return X, y

    def test_returns_valid_indices(self):
        X, y = self._xy(300, 0)
        out = PolrMethod().impute(y, X, X[:40], make_rng(0))
        assert set(np.unique(out)).issubset({0, 1, 2, 3})
        assert out.shape == (40,)

    def test_deterministic(self):
        X, y = self._xy(300, 1)
        a = PolrMethod().impute(y, X, X[:40], make_rng(2))
        b = PolrMethod().impute(y, X, X[:40], make_rng(2))
        np.testing.assert_array_equal(a, b)

    def test_respects_ordinal_signal(self):
        # Higher X[:,0] -> higher ordinal category on average.
        X, y = self._xy(800, 2)
        order = np.argsort(X[:, 0])
        low, high = order[:150], order[-150:]
        rng = make_rng(0)
        out_low = PolrMethod().impute(y, X, X[low], rng)
        out_high = PolrMethod().impute(y, X, X[high], rng)
        assert out_high.mean() > out_low.mean()

    @staticmethod
    def _separated(n=4000, seed=0, p=3):
        """Quasi-complete separation: sparse extreme categories perfectly
        ordered by continuous predictors (issue #7's in-MICE failure mode)."""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        z = X @ np.array([2.0, -1.5, 1.0])
        thr = np.quantile(z, [0.55, 0.99, 0.998])
        y = np.digitize(z, thr).astype(float)
        return X, y, z

    def test_separated_fit_does_not_fall_back_to_marginal(self):
        """The ridge keeps the conditional fit usable, so the predictor-blind
        marginal-draw fallback (which silently degraded polr on exactly these
        real-data fits) no longer fires."""
        X, y, z = self._separated()
        rng = np.random.default_rng(1)
        mis = rng.random(X.shape[0]) < 0.2
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            PolrMethod().impute(y[~mis], X[~mis], X[mis], make_rng(0))
        assert not any("marginal draw" in str(w.message) for w in caught)

    def test_separated_fit_stays_predictor_aware(self):
        """On separated sweep data imputations still track the predictor — the
        whole point of not degrading to a marginal draw."""
        X, y, z = self._separated()
        rng = np.random.default_rng(1)
        mis = rng.random(X.shape[0]) < 0.2
        out = PolrMethod().impute(y[~mis], X[~mis], X[mis], make_rng(0))
        zmis = z[mis]
        lo = out[zmis < np.quantile(zmis, 0.25)].mean()
        hi = out[zmis > np.quantile(zmis, 0.75)].mean()
        assert hi > lo


# ----------------------------------------------------------- mixed integration
class TestMixedIntegration:
    def test_mixed_type_runs_and_valid_codes(self):
        data, kinds = _make_mixed(400, 0)
        miss = datasets.make_mcar(data, 0.2, seed=1)
        design = MICEDesign.from_array(miss, column_kinds=kinds)
        sol = mice(design, n_imputations=4, max_iter=5, seed=7)
        for d in sol.completed_datasets():
            assert not np.isnan(d).any()
            for j in (1, 2, 3):
                allowed = set(design.levels_for(j).tolist())
                assert set(np.unique(d[:, j])).issubset(allowed)

    def test_auto_methods_by_kind(self):
        data, kinds = _make_mixed(200, 0)
        miss = datasets.make_mcar(data, 0.2, seed=1)
        d = MICEDesign.from_array(miss, column_kinds=kinds)
        assert d.methods == ("pmm", "logreg", "polyreg", "polr")

    def test_deterministic_mixed(self):
        data, kinds = _make_mixed(300, 0)
        miss = datasets.make_mcar(data, 0.2, seed=1)
        design = MICEDesign.from_array(miss, column_kinds=kinds)
        a = mice(design, n_imputations=3, max_iter=4, seed=11)
        b = mice(design, n_imputations=3, max_iter=4, seed=11)
        for da, db in zip(a.completed_datasets(), b.completed_datasets()):
            np.testing.assert_array_equal(da, db)

    def test_observed_categorical_preserved(self):
        data, kinds = _make_mixed(300, 0)
        miss = datasets.make_mcar(data, 0.2, seed=1)
        design = MICEDesign.from_array(miss, column_kinds=kinds)
        sol = mice(design, n_imputations=2, max_iter=3, seed=0)
        observed = ~np.isnan(miss)
        for d in sol.completed_datasets():
            np.testing.assert_array_equal(d[observed], miss[observed])


# ---------------------------------------------------------- design validation
class TestCategoricalDesignValidation:
    def test_binary_requires_two_levels(self):
        data = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [np.nan, 1.0]])
        with pytest.raises(ValidationError, match="binary.*exactly 2|exactly 2"):
            MICEDesign.from_array(data, column_kinds=["binary", "numeric"])

    def test_non_integer_categorical_rejected(self):
        data = np.array([[0.5, 1.0], [1.0, 2.0], [0.0, np.nan]])
        with pytest.raises(ValidationError, match="integer category codes"):
            MICEDesign.from_array(data, column_kinds=["categorical", "numeric"])

    def test_levels_detected(self):
        data = np.array([[0.0, 1.0], [2.0, 2.0], [5.0, np.nan], [2.0, 3.0]])
        d = MICEDesign.from_array(data, column_kinds=["categorical", "numeric"])
        np.testing.assert_array_equal(d.levels_for(0), [0.0, 2.0, 5.0])
        assert d.levels_for(1) is None

    def test_method_kind_mismatch_rejected(self):
        data, kinds = _make_mixed(100, 0)
        miss = datasets.make_mcar(data, 0.2, seed=1)
        # pmm is numeric-only; forcing it on the binary column must fail.
        with pytest.raises(ValidationError, match="imputes"):
            MICEDesign.from_array(miss, column_kinds=kinds, methods={1: "pmm"})
