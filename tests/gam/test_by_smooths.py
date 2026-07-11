"""
Tests for VA-2: gam continuous ``by=`` (varying-coefficient) smooths.

``s(x, by=z)`` fits the varying-coefficient term ``z * f(x)`` — the smooth keeps
its full basis (no centering) and each row is scaled by the by-variable — exactly
as mgcv's ``s(x, by=z)`` for a continuous ``by``. Validated vs ``mgcv::gam``
(total EDF, fitted values). Native factor-``by`` auto-expansion (a separate smooth
per factor level) is not yet provided; it can be built with one continuous-``by``
term per level-indicator.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pystatistics.gam import gam, s
from pystatistics.core.exceptions import ValidationError

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(scope="module")
def by_data():
    return json.loads((_FIXTURES / "gam_by_mgcv.json").read_text())


class TestContinuousBy:
    def test_matches_mgcv(self, by_data):
        x = np.array(by_data["x"])
        z = np.array(by_data["z"])
        y = np.array(by_data["y1"])
        r = by_data["cont"]
        sol = gam(y, smooths=[s("x", by="z")], smooth_data={"x": x, "z": z},
                  method="REML")
        assert abs(sol.total_edf - r["edf"]) < 0.1
        mgcv_fit = np.array(r["fitted"])
        assert np.max(np.abs(sol.fitted_values - mgcv_fit)) < 0.1

    def test_keeps_full_basis(self, by_data):
        """A by-smooth is not centered — it has k (not k-1) columns."""
        x = np.array(by_data["x"])
        z = np.array(by_data["z"])
        y = np.array(by_data["y1"])
        sol = gam(y, smooths=[s("x", k=10, by="z")],
                  smooth_data={"x": x, "z": z}, method="REML")
        # 1 intercept + 10 by-smooth columns
        assert sol.coefficients.shape[0] == 11

    def test_term_is_z_times_fx(self, by_data):
        """Doubling the by-variable doubles the smooth's contribution."""
        x = np.array(by_data["x"])
        z = np.array(by_data["z"])
        y = np.array(by_data["y1"])
        sol = gam(y, smooths=[s("x", by="z")], smooth_data={"x": x, "z": z},
                  method="REML", sp=[1.0])
        sol2 = gam(y, smooths=[s("x", by="z2")],
                   smooth_data={"x": x, "z2": 2.0 * z}, method="REML", sp=[1.0])
        # the smooth part (fitted minus intercept) scales with the by-variable
        # up to the refit's intercept; check the design column scaling directly.
        from pystatistics.gam._basis import build_design
        Xp = np.ones((len(y), 1))
        Xa1, _ = build_design(Xp, {"x": x, "z": z}, [s("x", by="z")])
        Xa2, _ = build_design(Xp, {"x": x, "z2": 2 * z}, [s("x", by="z2")])
        assert np.allclose(Xa2[:, 1:], 2.0 * Xa1[:, 1:])


class TestByValidation:
    def test_missing_by_variable_raises(self, by_data):
        x = np.array(by_data["x"])
        y = np.array(by_data["y1"])
        with pytest.raises(ValidationError, match="missing variable 'z'"):
            gam(y, smooths=[s("x", by="z")], smooth_data={"x": x},
                method="REML")

    def test_by_must_be_string(self):
        with pytest.raises(ValidationError, match="by must be"):
            s("x", by=123)

    def test_level_indicator_is_smooth_on_subset(self, by_data):
        """A 0/1 indicator by-variable restricts the smooth to that level's
        observations — the building block for a hand-rolled factor-by."""
        x = np.array(by_data["x"])
        g = np.array(by_data["g"])
        y = np.array(by_data["y2"])
        ind_a = (g == "a").astype(float)
        sol = gam(y, smooths=[s("x", by="ia")],
                  smooth_data={"x": x, "ia": ind_a}, method="REML")
        assert sol.converged
        # the smooth contributes nothing where the indicator is 0
        contrib = sol.fitted_values - sol.coefficients[0]
        assert np.allclose(contrib[g == "b"], 0.0, atol=1e-8)
