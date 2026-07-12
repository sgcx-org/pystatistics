"""
Tests for factor ``by=`` smooths and the fail-loud guard on un-annotated
factor-looking ``by`` columns (VA2-F1).

``s(x, by=g, by_type='factor')`` fits a separate smooth per level of an
integer-coded grouping variable and injects the per-level group means, exactly
as ``mgcv``'s ``s(x, by=factor(g))``. Validated against a frozen mgcv fixture
(total EDF and fitted values agree to ~1e-8).

A factor-looking ``by`` column with ``by_type`` unset is REJECTED rather than
silently multiplied into a meaningless continuous varying coefficient, and a
binary 0/1 ``by`` is deliberately NOT flagged (it is a valid continuous
single-subgroup smooth).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from pystatistics.gam import gam, s
from pystatistics.gam._factor_by import looks_like_factor, factor_levels
from pystatistics.core.exceptions import ValidationError

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(scope="module")
def by_data():
    return json.loads((_FIXTURES / "gam_by_mgcv.json").read_text())


def _synth_three_level(seed: int = 0):
    """A deterministic 3-level dataset: y = g-specific smooth of x + noise."""
    rng = np.random.default_rng(seed)
    n = 300
    x = np.linspace(0.0, 1.0, n)
    g = np.tile(np.array([0, 1, 2]), n // 3).astype(float)
    shape = {0: np.sin(2 * np.pi * x), 1: np.cos(3 * x), 2: x ** 2}
    y = np.array([shape[int(gi)][i] for i, gi in enumerate(g)])
    y = y + 0.3 * g + rng.normal(scale=0.1, size=n)
    return x, g, y


# --------------------------------------------------------------------------
# mgcv parity
# --------------------------------------------------------------------------
class TestFactorByMatchesMgcv:
    def test_edf_and_fitted_match_mgcv(self, by_data):
        """s(x, by=factor(g)) reproduces mgcv's fac fit to fp arithmetic."""
        x = np.array(by_data["x"])
        g = np.array(by_data["g"])
        y = np.array(by_data["y2"])
        fac = by_data["fac"]
        gcode = (g == "b").astype(float)  # a -> 0, b -> 1
        sol = gam(y, smooths=[s("x", by="gcode", by_type="factor")],
                  smooth_data={"x": x, "gcode": gcode}, method="REML")
        assert abs(sol.total_edf - fac["edf"]) < 1e-4
        assert np.max(np.abs(sol.fitted_values - np.array(fac["fitted"]))) < 1e-5

    def test_structure_two_levels(self, by_data):
        """Two levels -> two smooths, two smoothing params, k coefs each."""
        x = np.array(by_data["x"])
        g = np.array(by_data["g"])
        y = np.array(by_data["y2"])
        gcode = (g == "b").astype(float)
        sol = gam(y, smooths=[s("x", k=10, by="gcode", by_type="factor")],
                  smooth_data={"x": x, "gcode": gcode}, method="REML")
        assert len(sol.smooth_terms) == 2
        assert len(sol.params.lambdas) == 2
        # per-level intercept (contrast) + global intercept + 2 x (k-1) smooth
        # cols = 2 + 2*9 = 20
        assert sol.coefficients.shape[0] == 20
        names = [si.term_name for si in sol.smooth_terms]
        assert names == ["s(x):gcode=0", "s(x):gcode=1"]

    def test_two_smooths_share_one_factor(self):
        """s(x, by=g) + s(w, by=g) with the SAME factor injects the group main
        effect once (no false-positive collision)."""
        x, g, y = _synth_three_level()
        w = np.linspace(-1.0, 1.0, len(x))
        sol = gam(
            y,
            smooths=[s("x", k=8, by="g", by_type="factor"),
                     s("w", k=8, by="g", by_type="factor")],
            smooth_data={"x": x, "w": w, "g": g}, method="REML",
        )
        assert sol.converged
        assert len(sol.smooth_terms) == 6          # 2 smooths x 3 levels
        assert len(sol.params.lambdas) == 6
        names = [si.term_name for si in sol.smooth_terms]
        assert names.count("s(x):g=0") == 1 and names.count("s(w):g=2") == 1

    def test_three_levels_structure(self):
        x, g, y = _synth_three_level()
        sol = gam(y, smooths=[s("x", k=10, by="g", by_type="factor")],
                  smooth_data={"x": x, "g": g}, method="REML")
        assert len(sol.smooth_terms) == 3
        assert len(sol.params.lambdas) == 3
        # 3 x k coefficients total (mgcv's 30 for k=10): global intercept + 2
        # contrasts + 3*(k-1) = 1 + 2 + 27 = 30
        assert sol.coefficients.shape[0] == 30
        assert sol.converged

# --------------------------------------------------------------------------
# fail-loud guard on un-annotated factor-looking by
# --------------------------------------------------------------------------
class TestFactorGuard:
    @pytest.mark.parametrize("codes", [
        [0, 1, 2],        # 0-based (pandas category codes)
        [1, 2, 3],        # 1-based (R as.integer(factor(...)))
        [0, 1, 2, 3, 4],
    ])
    def test_unannotated_factor_looking_by_raises(self, codes):
        n = 300
        rng = np.random.default_rng(1)
        x = np.linspace(0, 1, n)
        z = np.tile(np.array(codes, dtype=float), n)[:n]
        y = rng.normal(size=n)
        with pytest.raises(ValidationError, match="looks categorical"):
            gam(y, smooths=[s("x", by="z")],
                smooth_data={"x": x, "z": z}, method="REML")

    def test_binary_01_not_flagged(self, by_data):
        """A 0/1 by-variable is a valid continuous single-subgroup smooth and
        must NOT trip the guard."""
        x = np.array(by_data["x"])
        g = np.array(by_data["g"])
        y = np.array(by_data["y2"])
        z01 = (g == "b").astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no guard warning either
            sol = gam(y, smooths=[s("x", by="z01")],
                      smooth_data={"x": x, "z01": z01}, method="REML")
        assert sol.converged

    def test_continuous_by_bypasses_guard(self):
        """Explicit by_type='continuous' fits the {0,1,2} column as a varying
        coefficient without raising."""
        n = 300
        rng = np.random.default_rng(2)
        x = np.linspace(0, 1, n)
        z = np.tile(np.array([0.0, 1.0, 2.0]), n)[:n]
        y = z * np.sin(2 * np.pi * x) + rng.normal(scale=0.1, size=n)
        sol = gam(y, smooths=[s("x", by="z", by_type="continuous")],
                  smooth_data={"x": x, "z": z}, method="REML")
        assert sol.converged

    def test_genuine_continuous_not_flagged(self):
        """A real continuous by-variable (many distinct non-integer values) is
        never flagged."""
        rng = np.random.default_rng(3)
        x = np.linspace(0, 1, 200)
        z = rng.normal(size=200)
        assert not looks_like_factor(z)


# --------------------------------------------------------------------------
# validation / edge cases
# --------------------------------------------------------------------------
class TestFactorValidation:
    def test_by_type_requires_by(self):
        with pytest.raises(ValidationError, match="only meaningful with a by"):
            s("x", by_type="factor")

    def test_invalid_by_type(self):
        with pytest.raises(ValidationError, match="by_type must be one of"):
            s("x", by="g", by_type="nonsense")

    def test_tp_default_for_factor(self):
        assert s("x", by="g", by_type="factor").bs == "tp"
        assert s("x", by="g", by_type="factor", bs="cr").bs == "cr"
        assert s("x", by="z", by_type="continuous").bs == "cr"

    def test_non_integer_factor_codes_raise(self):
        x = np.linspace(0, 1, 30)
        z = np.tile(np.array([0.0, 1.5, 2.0]), 10)
        y = np.zeros(30)
        with pytest.raises(ValidationError, match="integer-coded levels"):
            gam(y, smooths=[s("x", by="z", by_type="factor")],
                smooth_data={"x": x, "z": z}, method="REML")

    def test_single_level_factor_raises(self):
        x = np.linspace(0, 1, 30)
        z = np.zeros(30)
        y = np.zeros(30)
        with pytest.raises(ValidationError, match="at least 2 levels"):
            gam(y, smooths=[s("x", by="z", by_type="factor")],
                smooth_data={"x": x, "z": z}, method="REML")

    def test_group_already_in_X_raises(self):
        """Auto-injected group main effect must fail loud if the user also
        encoded the grouping variable in X."""
        x, g, y = _synth_three_level()
        i1 = (g == 1).astype(float)
        i2 = (g == 2).astype(float)
        # X already carries the group contrasts -> collision.
        with pytest.raises(ValidationError, match="collides with a column"):
            gam(y, X=np.column_stack([np.ones_like(x), i1, i2]),
                names=["icpt", "g1", "g2"],
                smooths=[s("x", by="g", by_type="factor")],
                smooth_data={"x": x, "g": g}, method="REML")

    def test_factor_requires_intercept(self):
        """No intercept in X -> factor-by cannot anchor its treatment
        contrasts."""
        x, g, y = _synth_three_level()
        # X is a single non-constant column, no intercept.
        with pytest.raises(ValidationError, match="no intercept"):
            gam(y, X=x.reshape(-1, 1), names=["xlin"],
                smooths=[s("x", by="g", by_type="factor")],
                smooth_data={"x": x, "g": g}, method="REML")


class TestDetectionHelpers:
    def test_looks_like_factor(self):
        assert looks_like_factor(np.array([0, 1, 2, 0, 1, 2.0]))
        assert looks_like_factor(np.array([1, 2, 3, 1, 2, 3.0]))
        assert not looks_like_factor(np.array([0.0, 1.0]))          # binary
        assert not looks_like_factor(np.array([0, 2, 4.0]))         # gappy
        assert not looks_like_factor(np.array([0.5, 1.5, 2.5]))     # non-int

    def test_factor_levels(self):
        lv = factor_levels(np.array([2, 0, 1, 2, 0.0]), "g")
        assert lv.tolist() == [0, 1, 2]
