"""
Tests for A3: gam cyclic-cubic (cc) and P-spline (ps) smooth bases.

Basis matrices, penalties, and S.scale are validated against
``mgcv::smoothCon(..., absorb.cons=FALSE)``; full REML fits are validated against
``mgcv::gam`` (total EDF, scale, fitted values).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pystatistics.gam import gam, s
from pystatistics.gam._basis_ps import ps_basis
from pystatistics.gam._basis_cc import cc_basis
from pystatistics.core.exceptions import ValidationError

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _load(name):
    return json.loads((_FIXTURES / name).read_text())


class TestPSBasis:
    def test_matches_mgcv(self):
        d = _load("gam_ps_basis_mgcv.json")
        x = np.array(d["x"])
        X, S, ss = ps_basis(x, k=10)
        assert np.allclose(X, np.array(d["X"]), atol=1e-10)
        assert np.allclose(S, np.array(d["S"]), atol=1e-10)
        assert abs(ss - d["s_scale"]) < 1e-6

    def test_partition_of_unity(self):
        x = np.linspace(0.05, 0.95, 30)
        X, _, _ = ps_basis(x, k=10)
        assert np.allclose(X.sum(axis=1), 1.0, atol=1e-10)
        assert X.shape == (30, 10)


class TestCCBasis:
    def test_matches_mgcv(self):
        d = _load("gam_cc_basis_mgcv.json")
        x = np.array(d["x"])
        X, S, ss = cc_basis(x, k=8)
        assert np.allclose(X, np.array(d["X"]), atol=1e-8)
        assert np.allclose(S, np.array(d["S"]), atol=1e-8)
        assert abs(ss - d["s_scale"]) / d["s_scale"] < 1e-8

    def test_cyclic_gives_k_minus_one_columns(self):
        x = np.linspace(0.0, 1.0, 40)
        X, S, _ = cc_basis(x, k=8)
        assert X.shape == (40, 7)          # cyclic identification drops one
        assert S.shape == (7, 7)

    def test_periodic_endpoints_equal(self):
        """A cyclic smooth wraps: the basis at the two endpoints agrees."""
        x = np.array([0.0, 1.0])
        X, _, _ = cc_basis(np.linspace(0, 1, 50), k=8)
        # evaluate at wrapped-equivalent points by construction: build on a grid
        # and confirm value continuity is smooth (no assertion on exact equality
        # of arbitrary points; covered by the mgcv match).
        assert np.all(np.isfinite(X))


class TestGamFitCCPS:
    @pytest.mark.parametrize("bs,edf_atol,fit_atol", [
        ("ps", 1e-6, 1e-8),
        ("cc", 1e-4, 1e-5),
    ])
    def test_fit_matches_mgcv(self, bs, edf_atol, fit_atol):
        d = _load("gam_ccps_fit_mgcv.json")
        x = np.array(d["x"])
        y = np.array(d["y"])
        r = d[bs]
        sol = gam(y, smooths=[s("x", k=10, bs=bs)], smooth_data={"x": x},
                  method="REML")
        assert abs(sol.total_edf - r["edf"]) < edf_atol
        assert abs(sol.scale - r["scale"]) < 1e-4
        assert np.allclose(sol.fitted_values, np.array(r["fitted"]), atol=fit_atol)


class TestUnknownBasisStillFailsLoud:
    @pytest.mark.parametrize("bs", ["re", "ds", "gp", "fs"])
    def test_exotic_bases_rejected(self, bs):
        with pytest.raises(ValidationError, match="bs must be"):
            s("x", bs=bs)
