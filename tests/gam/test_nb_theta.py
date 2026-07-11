"""
Tests for VA-3: usable gam ``family='nb'`` with estimated dispersion theta.

Previously ``gam(family='nb')`` failed loud ("Cannot compute deviance without
theta"). It now estimates theta by minimising the profiled REML criterion over
theta (mgcv's ``nb()`` approach), and fits.

Validation notes:
- The estimated theta matches ``mgcv::gam(family=nb())`` to ~1% (the residual is
  mgcv's extended-family REML theta-normalisation).
- The smooth fit itself carries gam's *existing* GLM-family tolerance vs mgcv
  (the "two-tier contract" — a plain Poisson gam shows the same ~0.2 EDF gap);
  that gap is the domain of the GLM-family analytic sp-gradient work, not of
  theta estimation. So the fit is checked for self-consistency and closeness,
  not machine-precision mgcv parity.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pystatistics.gam import gam, s
from pystatistics.regression.families import NegativeBinomial

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(scope="module")
def nb_data():
    d = json.loads((_FIXTURES / "gam_nb_mgcv.json").read_text())
    return np.array(d["x"]), np.array(d["y"]), d


class TestNbTheta:
    def test_nb_fits_and_estimates_theta(self, nb_data):
        x, y, d = nb_data
        sol = gam(y, smooths=[s("x", k=10)], smooth_data={"x": x},
                  family="nb", method="REML")
        assert sol._result.params.family_name == "negative.binomial"
        theta = sol._result.info["nb_theta"]
        # theta close to mgcv's nb() estimate (method-sensitive dispersion).
        assert abs(theta - d["mgcv_theta"]) / d["mgcv_theta"] < 0.05

    def test_fit_close_to_mgcv(self, nb_data):
        """Fitted mean is close to mgcv (within gam's GLM-family tolerance)."""
        x, y, d = nb_data
        sol = gam(y, smooths=[s("x", k=10)], smooth_data={"x": x},
                  family="nb", method="REML")
        mgcv_fit = np.array(d["mgcv_fitted"])
        rel = np.max(np.abs(sol.fitted_values - mgcv_fit) / mgcv_fit)
        assert rel < 0.10                       # a few percent, like poisson gam
        assert abs(sol.total_edf - d["mgcv_edf"]) < 0.5

    def test_fixed_theta_still_works(self, nb_data):
        """An explicit theta bypasses estimation and fits directly."""
        x, y, _ = nb_data
        sol = gam(y, smooths=[s("x", k=10)], smooth_data={"x": x},
                  family=NegativeBinomial(theta=3.0), method="REML")
        assert sol.converged
        assert np.all(np.isfinite(sol.fitted_values))

    def test_theta_recovers_dispersion_direction(self):
        """More overdispersion ⇒ smaller estimated theta."""
        rng = np.random.default_rng(3)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        mu = np.exp(0.5 + np.sin(2 * np.pi * x))
        y_od = rng.negative_binomial(1.0, 1.0 / (1.0 + mu)).astype(float)   # theta~1
        y_ld = rng.negative_binomial(10.0, 10.0 / (10.0 + mu)).astype(float)  # theta~10
        t_od = gam(y_od, smooths=[s("x", k=8)], smooth_data={"x": x},
                   family="nb", method="REML")._result.info["nb_theta"]
        t_ld = gam(y_ld, smooths=[s("x", k=8)], smooth_data={"x": x},
                   family="nb", method="REML")._result.info["nb_theta"]
        assert t_od < t_ld
