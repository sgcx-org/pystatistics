"""
Tests for VA-6a: regression diagnostics (hat values, Cook's distance,
standardized residuals) and analysis-of-deviance tables (anova / drop1).

References validated vs R (`hatvalues`, `cooks.distance`, `rstandard`,
`anova.glm`/`anova.lm`, `drop1`) on committed fixtures.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pystatistics.regression import fit, anova, drop1, Design
from pystatistics.regression.terms import C
from pystatistics.core.exceptions import ValidationError

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(scope="module")
def refs():
    return json.loads((_FIXTURES / "regression_diagnostics_anova.json").read_text())


@pytest.fixture(scope="module")
def va5():
    d = json.loads((_FIXTURES / "glm_links_families.json").read_text())
    x1 = np.asarray(d["x1"], float)
    x2 = np.asarray(d["x2"], float)
    X = np.column_stack([np.ones(len(x1)), x1, x2])
    return X, d


class TestDiagnostics:
    def test_glm_poisson_diagnostics(self, va5, refs):
        X, d = va5
        m = fit(X, np.asarray(d["yc"], float), family="poisson")
        r = refs["glm_poisson_diag"]
        assert np.allclose(m.hat_values[:5], r["hat"], atol=1e-6)
        assert np.allclose(m.cooks_distance[:5], r["cook"], atol=1e-6)
        assert np.allclose(m.residuals_standardized[:5], r["rstd"], atol=1e-6)
        # leverage sums to the rank.
        assert abs(np.sum(m.hat_values) - m.rank) < 1e-5

    def test_ols_diagnostics(self, va5, refs):
        X, d = va5
        m = fit(X, np.asarray(d["yig"], float))
        r = refs["ols_diag"]
        assert np.allclose(m.hat_values[:5], r["hat"], atol=1e-6)
        assert np.allclose(m.cooks_distance[:5], r["cook"], atol=1e-6)
        assert np.allclose(m.residuals_standardized[:5], r["rstd"], atol=1e-6)
        assert abs(np.sum(m.hat_values) - m._result.params.rank) < 1e-5


def _match_anova(rows, ref, has_dev=True):
    assert len(rows) == len(ref)
    for row, (term, df, dev, rdf, rdev, p) in zip(rows, ref):
        assert row.resid_df == rdf
        assert abs(row.resid_deviance - rdev) < 1e-3
        if df is None:
            assert row.df is None
        else:
            assert row.df == df
            if has_dev:
                assert abs(row.deviance - dev) < 1e-3
            if p is not None:
                assert abs(row.p_value - p) < max(1e-4, p * 1e-2)


class TestAnova:
    def test_poisson_sequential_chisq(self, va5, refs):
        X, d = va5
        m = fit(X, np.asarray(d["yc"], float), family="poisson",
                names=["(Intercept)", "x1", "x2"])
        _match_anova(anova(m, test="Chisq").rows, refs["poisson_anova"])

    def test_ols_sequential_f(self, va5, refs):
        X, d = va5
        m = fit(X, np.asarray(d["yig"], float), names=["(Intercept)", "x1", "x2"])
        rows = [r for r in anova(m).rows if r.df is not None]  # drop NULL row
        for row, (term, df, ss, f, p) in zip(rows, refs["ols_anova"]):
            assert row.df == df
            assert abs(row.deviance - ss) < 1e-4     # SS for a linear model
            assert abs(row.statistic - f) < 1e-3
            assert abs(row.p_value - p) < 1e-3

    def test_nested_comparison(self, va5, refs):
        X, d = va5
        yc = np.asarray(d["yc"], float)
        m1 = fit(X[:, :2], yc, family="poisson")
        m2 = fit(X, yc, family="poisson")
        rows = anova(m1, m2, test="Chisq").rows
        for row, (rdf, rdev, df, dev, p) in zip(rows, refs["nested"]):
            assert row.resid_df == rdf
            assert abs(row.resid_deviance - rdev) < 1e-3
            if df is not None:
                assert row.df == df
                assert abs(row.deviance - dev) < 1e-3
                assert abs(row.p_value - p) < 1e-4

    def test_factor_term_grouped(self, refs):
        """A k-level factor is one anova term with df = k-1 (matches R)."""
        f = refs["fac"]
        y = np.asarray(f["y"], float)
        d = Design.from_datasource(
            {"y": y, "x": np.asarray(f["x"], float), "grp": np.asarray(f["grp"])},
            terms=["x", C("grp")], y="y",
        )
        m = fit(d, family="binomial")
        _match_anova(anova(m, test="Chisq").rows, refs["fac_anova"])
        # grp spans 2 dummy columns but is a single df=2 term.
        grp_row = [r for r in anova(m, test="Chisq").rows if r.term == "grp"][0]
        assert grp_row.df == 2


class TestDrop1:
    def test_poisson_drop1_chisq(self, va5, refs):
        X, d = va5
        m = fit(X, np.asarray(d["yc"], float), family="poisson",
                names=["(Intercept)", "x1", "x2"])
        rows = drop1(m, test="Chisq").rows
        for row, (term, df, rdev, aic, lrt, p) in zip(rows, refs["poisson_drop1"]):
            assert abs(row.resid_deviance - rdev) < 1e-3
            assert abs(row.aic - aic) < 1e-2
            if df is not None:
                assert row.df == df
                assert abs(row.statistic - lrt) < 1e-3
                assert abs(row.p_value - p) < 1e-4


class TestFailLoud:
    def test_unknown_test_raises(self, va5):
        X, d = va5
        m = fit(X, np.asarray(d["yc"], float), family="poisson")
        with pytest.raises(ValidationError, match="Unknown test"):
            anova(m, test="wald")

    def test_anova_no_models_raises(self):
        with pytest.raises(ValidationError, match="at least one"):
            anova()
