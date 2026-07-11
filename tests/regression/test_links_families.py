"""
Tests for VA-5: additional GLM links and families.

Links: cloglog, cauchit, sqrt, 1/mu^2 (inverse-squared).
Families: inverse.gaussian, quasipoisson, quasibinomial.

Validated vs R `glm` on a committed fixture (n=120): coefficients, standard
errors, dispersion, deviance, and AIC. The quasi families have no proper
likelihood, so their AIC is NaN (matching R's `quasi*()$aic`).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pystatistics.regression import fit
from pystatistics.regression.families import (
    resolve_family, _resolve_link, Binomial, Poisson, IdentityLink,
)
from pystatistics.core.exceptions import ValidationError

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(scope="module")
def glm_data():
    return json.loads((_FIXTURES / "glm_links_families.json").read_text())


def _design(d):
    x1 = np.asarray(d["x1"], float)
    x2 = np.asarray(d["x2"], float)
    return np.column_stack([np.ones(len(x1)), x1, x2])


def _family_for(name, link):
    if link is not None:
        if name == "binomial":
            return Binomial(link=link)
        if name == "poisson":
            return Poisson(link=link)
    return resolve_family(name)


class TestNewLinks:
    @pytest.mark.parametrize("name", ["cloglog", "cauchit", "sqrt", "1/mu^2"])
    def test_link_resolves(self, name):
        assert _resolve_link(name, IdentityLink()).name == name

    @pytest.mark.parametrize("name", ["cloglog", "cauchit"])
    def test_linkinv_link_roundtrip(self, name):
        link = _resolve_link(name, IdentityLink())
        eta = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        assert np.allclose(link.link(link.linkinv(eta)), eta, atol=1e-8)

    def test_sqrt_and_inverse_squared_roundtrip(self):
        for name in ("sqrt", "1/mu^2"):
            link = _resolve_link(name, IdentityLink())
            mu = np.array([0.5, 1.0, 2.0, 5.0])
            assert np.allclose(link.linkinv(link.link(mu)), mu, atol=1e-8)


class TestNewFamiliesVsR:
    @pytest.mark.parametrize("tag", [
        "binomial_cloglog", "binomial_cauchit", "poisson_sqrt",
        "inverse_gaussian", "quasipoisson", "quasibinomial",
    ])
    def test_matches_r_glm(self, glm_data, tag):
        ref = glm_data["refs"][tag]
        X = _design(glm_data)
        y = np.asarray(glm_data[ref["y"]], float)
        fam = _family_for(ref["family"], ref["link"])
        sol = fit(X, y, family=fam, names=["(int)", "x1", "x2"])
        assert np.allclose(sol.coefficients, ref["coef"], atol=1e-4)
        assert np.allclose(sol.standard_errors, ref["se"], atol=1e-4)
        assert abs(sol.dispersion - ref["disp"]) < 1e-4
        assert abs(sol.deviance - ref["dev"]) < 1e-3
        if ref["aic"] is None:
            assert np.isnan(sol.aic)      # quasi families: AIC undefined
        else:
            assert abs(sol.aic - ref["aic"]) < 1e-2

    def test_quasipoisson_coef_equals_poisson(self, glm_data):
        """Quasi-Poisson has the same fit as Poisson; only SEs (dispersion) differ."""
        X = _design(glm_data)
        y = np.asarray(glm_data["yc"], float)
        qp = fit(X, y, family=resolve_family("quasipoisson"))
        po = fit(X, y, family=resolve_family("poisson"))
        assert np.allclose(qp.coefficients, po.coefficients, atol=1e-8)
        # quasi SE = poisson SE * sqrt(dispersion)
        assert np.allclose(
            qp.standard_errors,
            po.standard_errors * np.sqrt(qp.dispersion),
            rtol=1e-6,
        )

    def test_quasi_uses_t_inference(self, glm_data):
        """Estimated dispersion → t-distribution p-values (like R), not normal."""
        from scipy import stats
        X = _design(glm_data)
        y = np.asarray(glm_data["yb"], float)
        qb = fit(X, y, family=resolve_family("quasibinomial"))
        stat = np.asarray(qb.z_values)         # Wald statistic (A3 accessor)
        df = len(y) - X.shape[1]
        p_t = 2.0 * stats.t.sf(np.abs(stat), df)
        p_z = 2.0 * stats.norm.sf(np.abs(stat))
        # p-values follow the t-distribution (estimated dispersion), not normal.
        assert np.allclose(qb.p_values, p_t, atol=1e-8)
        assert not np.allclose(qb.p_values, p_z, atol=1e-4)


class TestUnknownStillFailsLoud:
    def test_unknown_link_raises(self):
        with pytest.raises(ValidationError, match="Unknown link"):
            _resolve_link("logloglog", IdentityLink())

    def test_unknown_family_raises(self):
        with pytest.raises(ValidationError, match="Unknown family"):
            resolve_family("quasigaussian")
