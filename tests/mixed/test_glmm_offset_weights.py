"""
Tests for A4: glmm offset, prior weights, and aggregated-binomial response.

Validated vs lme4::glmer on the classic ``cbpp`` aggregated-binomial example and
a Poisson model with a ``log(exposure)`` offset. Agreement is at the mixed
module's established Laplace two-tier tolerance (glmm ≈ glmer(nAGQ=1)), not
machine precision — both are Laplace approximations.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pystatistics.mixed import glmm
from pystatistics.core.exceptions import ValidationError

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(scope="module")
def ref():
    return json.loads((_FIXTURES / "glmm_offset_weights_glmer.json").read_text())


def _cbpp_design(cb):
    X = np.column_stack([
        np.ones(len(cb["incidence"])),
        np.asarray(cb["period2"], float),
        np.asarray(cb["period3"], float),
        np.asarray(cb["period4"], float),
    ])
    herd = np.asarray(cb["herd"], int)
    inc = np.asarray(cb["incidence"], float)
    size = np.asarray(cb["size"], float)
    return X, herd, inc, size


class TestAggregatedBinomial:
    def test_cbind_response_matches_glmer(self, ref):
        cb = ref["cbpp"]
        X, herd, inc, size = _cbpp_design(cb)
        y = np.column_stack([inc, size - inc])   # cbind(incidence, size-incidence)
        sol = glmm(y, X, groups={"herd": herd}, family="binomial")
        assert np.allclose(sol.coefficients, cb["fixef"], atol=5e-3)
        assert np.allclose(sol.standard_errors, cb["se"], atol=2e-2)
        assert abs(sol.var_components[0].variance - cb["varRE"]) < 5e-3

    def test_proportion_plus_weights_equals_cbind(self, ref):
        """Passing proportions + weights=trials gives the same fit as cbind."""
        cb = ref["cbpp"]
        X, herd, inc, size = _cbpp_design(cb)
        y2 = np.column_stack([inc, size - inc])
        prop = inc / size
        a = glmm(y2, X, groups={"herd": herd}, family="binomial")
        b = glmm(prop, X, groups={"herd": herd}, family="binomial", weights=size)
        assert np.allclose(a.coefficients, b.coefficients, atol=1e-8)
        assert np.allclose(a.standard_errors, b.standard_errors, atol=1e-8)

    def test_cbind_requires_binomial(self, ref):
        cb = ref["cbpp"]
        X, herd, inc, size = _cbpp_design(cb)
        y = np.column_stack([inc, size - inc])
        with pytest.raises(ValidationError, match="binomial"):
            glmm(y, X, groups={"herd": herd}, family="poisson")


class TestOffset:
    def test_poisson_offset_matches_glmer(self, ref):
        po = ref["pois_offset"]
        yp = np.asarray(po["yp"], float)
        X = np.column_stack([np.ones(len(yp)), np.asarray(po["x"], float)])
        g = np.asarray(po["g"], int)
        sol = glmm(yp, X, groups={"g": g}, family="poisson",
                   offset=np.asarray(po["logexpo"], float))
        assert np.allclose(sol.coefficients, po["fixef"], atol=5e-3)
        assert np.allclose(sol.standard_errors, po["se"], atol=1e-2)
        assert abs(sol.var_components[0].variance - po["varRE"]) < 5e-3

    def test_offset_changes_fit(self, ref):
        """An offset materially changes the fit vs omitting it."""
        po = ref["pois_offset"]
        yp = np.asarray(po["yp"], float)
        X = np.column_stack([np.ones(len(yp)), np.asarray(po["x"], float)])
        g = np.asarray(po["g"], int)
        with_off = glmm(yp, X, groups={"g": g}, family="poisson",
                        offset=np.asarray(po["logexpo"], float))
        no_off = glmm(yp, X, groups={"g": g}, family="poisson")
        assert not np.allclose(with_off.coefficients, no_off.coefficients, atol=1e-2)

    def test_offset_wrong_length_raises(self, ref):
        po = ref["pois_offset"]
        yp = np.asarray(po["yp"], float)
        X = np.column_stack([np.ones(len(yp)), np.asarray(po["x"], float)])
        g = np.asarray(po["g"], int)
        with pytest.raises(ValidationError, match="offset"):
            glmm(yp, X, groups={"g": g}, family="poisson", offset=np.zeros(3))
