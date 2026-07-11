"""Full tensor / multivariate GAM fits validated against mgcv::gam.

Function-space checks (total EDF, scale, fitted values, per-margin
smoothing parameters) for ``te()``, ``ti()`` and isotropic ``s(x, z)``
against ``mgcv::gam`` — the module's two-tier contract. The per-margin
``sp`` match is the load-bearing check on the joint multi-lambda REML/GCV
machinery.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pystatistics.core.exceptions import ValidationError
from pystatistics.gam import gam, s, te, ti

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(scope="module")
def tensor_data():
    d = json.loads((_FIXTURES / "gam_tensor_mgcv.json").read_text())
    d["_np"] = {"x": np.array(d["x"]), "z": np.array(d["z"])}
    d["_y"] = np.array(d["y"])
    return d


def _check(sol, ref, *, sp_atol=None):
    assert sol.total_edf == pytest.approx(ref["edf_total"], abs=1e-3)
    assert sol.scale == pytest.approx(ref["scale"], rel=1e-4)
    assert np.allclose(sol.fitted_values, np.array(ref["fitted"]), atol=1e-4)
    if sp_atol is not None:
        assert np.allclose(sorted(sol.lambdas), sorted(ref["sp"]), atol=sp_atol)


class TestTensorFit:
    def test_te_reml(self, tensor_data):
        sol = gam(tensor_data["_y"],
                  smooths=[te("x", "z", bs="cr", k=[5, 4])],
                  smooth_data=tensor_data["_np"], method="REML")
        _check(sol, tensor_data["te_fit"], sp_atol=1e-2)

    def test_te_gcv(self, tensor_data):
        sol = gam(tensor_data["_y"],
                  smooths=[te("x", "z", bs="cr", k=[5, 4])],
                  smooth_data=tensor_data["_np"], method="GCV")
        _check(sol, tensor_data["te_gcv"], sp_atol=1e-2)

    def test_ti_functional_anova(self, tensor_data):
        sol = gam(tensor_data["_y"],
                  smooths=[ti("x", k=5), ti("z", k=5), ti("x", "z", k=[5, 5])],
                  smooth_data=tensor_data["_np"], method="REML")
        _check(sol, tensor_data["ti_fit"])
        # ti(x) + ti(z) each own one sp; ti(x,z) owns two -> four total.
        assert len(sol.lambdas) == 4

    def test_te_mixed_cc_cr_margins(self, tensor_data):
        sol = gam(tensor_data["_y"],
                  smooths=[te("x", "z", bs=["cc", "cr"], k=[6, 5])],
                  smooth_data=tensor_data["_np"], method="REML")
        _check(sol, tensor_data["te_mixed"])

    def test_te_poisson_glm(self, tensor_data):
        # Exercises the joint multi-lambda GLM (Newton-weight) REML gradient.
        # One margin is penalised onto its null space (sp on the flat lambda
        # -> inf plateau, not identifiable), so validate EDF + fitted, which
        # pin the fit; the identified margin's sp still matches mgcv.
        sol = gam(np.array(tensor_data["ycount"]),
                  smooths=[te("x", "z", bs="cr", k=[5, 4])],
                  smooth_data=tensor_data["_np"],
                  family="poisson", method="REML")
        ref = tensor_data["te_poisson"]
        assert sol.total_edf == pytest.approx(ref["edf_total"], abs=1e-2)
        assert np.allclose(sol.fitted_values, np.array(ref["fitted"]),
                           atol=1e-3)
        assert min(sol.lambdas) == pytest.approx(min(ref["sp"]), rel=1e-2)

    def test_isotropic_multivariate(self, tensor_data):
        sol = gam(tensor_data["_y"],
                  smooths=[s("x", "z", k=20)],
                  smooth_data=tensor_data["_np"], method="REML")
        _check(sol, tensor_data["iso_fit"])
        assert len(sol.lambdas) == 1  # one isotropic penalty

    def test_te_default_invocation(self, tensor_data):
        # R15: the bare default call must produce a sensible fit.
        sol = gam(tensor_data["_y"], smooths=[te("x", "z")],
                  smooth_data=tensor_data["_np"])
        assert sol.total_edf > 1.0
        assert len(sol.lambdas) == 2
        si = sol.smooth_terms[0]
        assert si.term_name == "te(x,z)"
        assert si.basis_type == "te"
        assert len(si.lambdas) == 2


class TestTensorReporting:
    def test_per_margin_sp_reported(self, tensor_data):
        sol = gam(tensor_data["_y"],
                  smooths=[te("x", "z", bs="cr", k=[5, 4])],
                  smooth_data=tensor_data["_np"], method="REML")
        si = sol.smooth_terms[0]
        assert len(si.lambdas) == 2 and len(si.s_scales) == 2
        assert all(lam > 0 for lam in si.lambdas)

    def test_summary_renders(self, tensor_data):
        sol = gam(tensor_data["_y"], smooths=[te("x", "z")],
                  smooth_data=tensor_data["_np"])
        text = sol.summary()
        assert "te(x,z)" in text


class TestTensorValidation:
    def test_te_single_var_is_ordinary_smooth(self):
        # mgcv te(x)/ti(x) reduce to a centred 1-D smooth.
        from pystatistics.gam import SmoothTerm
        assert isinstance(te("x"), SmoothTerm)
        assert isinstance(ti("x"), SmoothTerm)

    def test_bad_margin_count_k(self):
        with pytest.raises(ValidationError):
            te("x", "z", k=[5, 5, 5])  # 3 ks, 2 margins

    def test_isotropic_rejects_non_tp(self):
        with pytest.raises(ValidationError):
            s("x", "z", bs="cr")

    def test_sp_length_counts_margins(self, tensor_data):
        with pytest.raises(ValidationError, match="smoothing parameters"):
            gam(tensor_data["_y"], smooths=[te("x", "z")],
                smooth_data=tensor_data["_np"], sp=[1.0])  # needs 2
