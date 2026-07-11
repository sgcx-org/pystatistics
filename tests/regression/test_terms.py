"""
Tests for the structured term spec: categorical predictors + interactions.

Covers the design-matrix construction (build_terms_design), reference-level
selection, all three interaction kinds, label/array alignment through fit(),
the Cox no-intercept path, and the fail-loud contract for bare non-numeric
columns.
"""

import numpy as np
import pandas as pd
import pytest

from pystatistics import DataSource
from pystatistics.regression import Design, fit, C
from pystatistics.regression.terms import build_terms_design
from pystatistics.survival import coxph


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 240
    age = rng.normal(50, 10, n)
    sex = rng.choice(["F", "M"], n)
    treatment = rng.choice(["A", "B", "C"], n)
    y = (
        1.0
        + 0.05 * age
        + 1.5 * (sex == "M")
        + 2.0 * (treatment == "B")
        - 1.0 * (treatment == "C")
        + 0.8 * ((sex == "M") & (treatment == "B"))
        + rng.normal(0, 0.4, n)
    )
    return pd.DataFrame({"age": age, "sex": sex, "treatment": treatment, "y": y})


@pytest.fixture
def ds(df):
    return DataSource.from_dataframe(df)


# ── encoding & reference level ────────────────────────────────────────────

class TestCategoricalEncoding:
    def test_default_reference_is_first_sorted_level(self, ds):
        X, names, _assign, _tn = build_terms_design(ds, [C("treatment")], intercept=True)
        # treatment levels A,B,C → baseline A → columns for B, C
        assert names == ["(Intercept)", "treatment[B]", "treatment[C]"]

    def test_explicit_reference_level(self, ds):
        X, names, _assign, _tn = build_terms_design(ds, [C("treatment", ref="C")], intercept=True)
        assert names == ["(Intercept)", "treatment[A]", "treatment[B]"]

    def test_k_minus_1_columns(self, ds):
        X, names, _assign, _tn = build_terms_design(ds, [C("treatment")], intercept=False)
        assert X.shape[1] == 2  # 3 levels → 2 indicators

    def test_indicator_values_are_zero_one(self, ds):
        X, names, _assign, _tn = build_terms_design(ds, [C("sex", ref="F")], intercept=False)
        assert set(np.unique(X)) <= {0.0, 1.0}

    def test_bad_reference_raises(self, ds):
        with pytest.raises(ValueError, match="reference level"):
            build_terms_design(ds, [C("sex", ref="Q")], intercept=False)


# ── interactions ──────────────────────────────────────────────────────────

class TestInteractions:
    def test_numeric_x_numeric(self, ds):
        X, names, _assign, _tn = build_terms_design(ds, [("age", "age")], intercept=False)
        assert names == ["age:age"]
        np.testing.assert_allclose(X[:, 0], ds["age"] ** 2)

    def test_numeric_x_categorical(self, ds):
        X, names, _assign, _tn = build_terms_design(ds, [("age", C("sex", ref="F"))], intercept=False)
        assert names == ["age:sex[M]"]
        expected = ds["age"] * (ds["sex"] == "M")
        np.testing.assert_allclose(X[:, 0], expected)

    def test_categorical_x_categorical_label_order(self, ds):
        X, names, _assign, _tn = build_terms_design(
            ds, [(C("treatment", ref="A"), C("sex", ref="F"))], intercept=False
        )
        assert names == ["treatment[B]:sex[M]", "treatment[C]:sex[M]"]

    def test_cat_x_cat_column_order_matches_r(self):
        # R's model.matrix varies the FIRST factor fastest: with a-levels
        # (y, z) and b-levels (q, r) the interaction columns are
        # a[y]:b[q], a[z]:b[q], a[y]:b[r], a[z]:b[r].
        df = pd.DataFrame({
            "a": ["x", "y", "z", "x", "y", "z", "x", "y", "z"],
            "b": ["p", "q", "r", "q", "r", "p", "r", "p", "q"],
        })
        ds = DataSource.from_dataframe(df)
        X, names, _assign, _tn = build_terms_design(
            ds, [(C("a", ref="x"), C("b", ref="p"))], intercept=False
        )
        assert names == [
            "a[y]:b[q]", "a[z]:b[q]", "a[y]:b[r]", "a[z]:b[r]",
        ]

    def test_interaction_requires_two_elements(self, ds):
        with pytest.raises(ValueError, match="at least 2"):
            build_terms_design(ds, [("age",)], intercept=False)


# ── fail-loud contract ────────────────────────────────────────────────────

class TestFailLoud:
    def test_bare_categorical_column_raises(self, ds):
        with pytest.raises(ValueError, match="not numeric"):
            build_terms_design(ds, ["sex"], intercept=False)

    def test_empty_terms_raises(self, ds):
        with pytest.raises(ValueError, match="non-empty"):
            build_terms_design(ds, [], intercept=False)

    def test_bad_element_type_raises(self, ds):
        with pytest.raises(TypeError, match="column name"):
            build_terms_design(ds, [123], intercept=False)


# ── end-to-end through fit() / coxph() ────────────────────────────────────

class TestFitIntegration:
    def test_ols_recovers_coefficients_and_labels(self, ds):
        terms = [
            "age",
            C("sex", ref="F"),
            C("treatment", ref="A"),
            (C("treatment", ref="A"), C("sex", ref="F")),
        ]
        res = fit(Design.from_datasource(ds, y="y", terms=terms))
        coef = res.coef
        assert list(coef) == [
            "(Intercept)", "age", "sex[M]", "treatment[B]", "treatment[C]",
            "treatment[B]:sex[M]", "treatment[C]:sex[M]",
        ]
        assert coef["age"] == pytest.approx(0.05, abs=0.02)
        assert coef["sex[M]"] == pytest.approx(1.5, abs=0.3)
        assert coef["treatment[B]"] == pytest.approx(2.0, abs=0.3)

    def test_design_carries_names(self, ds):
        d = Design.from_datasource(ds, y="y", terms=["age", C("sex", ref="F")])
        assert d.names == ("(Intercept)", "age", "sex[M]")

    def test_explicit_names_override_design_names(self, ds):
        d = Design.from_datasource(ds, y="y", terms=["age", C("sex", ref="F")])
        res = fit(d, names=["b0", "b_age", "b_sex"])
        assert list(res.coef) == ["b0", "b_age", "b_sex"]

    def test_glm_binomial_terms(self, df):
        rng = np.random.default_rng(1)
        yb = (rng.random(len(df)) < 0.5).astype(float)
        ds = DataSource.from_dataframe(df.assign(yb=yb))
        res = fit(
            Design.from_datasource(ds, y="yb", terms=["age", C("sex", ref="F")]),
            family="binomial",
        )
        assert list(res.coef) == ["(Intercept)", "age", "sex[M]"]

    def test_cox_has_no_intercept(self, df):
        rng = np.random.default_rng(2)
        n = len(df)
        time = rng.exponential(2.0, n)
        event = (rng.random(n) < 0.6).astype(float)
        ds = DataSource.from_dataframe(df)
        res = coxph(time, event, ds, terms=["age", C("sex", ref="F")])
        assert list(res.coef) == ["age", "sex[M]"]

    def test_cox_terms_and_names_mutually_exclusive(self, df):
        ds = DataSource.from_dataframe(df)
        n = len(df)
        with pytest.raises(ValueError, match="either terms or names"):
            coxph(
                np.ones(n), np.ones(n), ds,
                terms=["age"], names=["age"],
            )
