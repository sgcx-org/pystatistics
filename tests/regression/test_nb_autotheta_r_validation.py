"""R-reference validation for auto-θ negative binomial with weights / offset.

Each case in ``tests/fixtures/nb_autotheta_cases.json`` was fit in R with
``MASS::glm.nb`` (which estimates the dispersion θ by profile likelihood) and its
outputs stored in ``nb_autotheta_r_results.json`` (regenerate with
``tests/fixtures/run_r_nb_autotheta_validation.R``). These tests fit the same
data with ``fit(family='negative-binomial', ...)`` — which runs the same
alternating θ-estimation loop — and assert parity.

The estimated θ, coefficients, standard errors, deviance, AIC and BIC are all
checked. AIC/BIC count θ as an estimated parameter, matching glm.nb. Null
deviance is not compared: glm.nb's null model uses R's intercept-true convention
(weighted-mean μ), whereas PyStatistics follows the glm.fit(intercept=FALSE)
convention used across the library — null-deviance parity is covered by the
fixed-θ fixtures.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pystatistics.regression import fit

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
CASES_PATH = FIXTURES_DIR / "nb_autotheta_cases.json"
RESULTS_PATH = FIXTURES_DIR / "nb_autotheta_r_results.json"

pytestmark = pytest.mark.skipif(
    not (CASES_PATH.exists() and RESULTS_PATH.exists()),
    reason="NB auto-θ fixtures not generated",
)

CASES = json.loads(CASES_PATH.read_text()) if CASES_PATH.exists() else {}
RESULTS = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
CASE_NAMES = sorted(CASES.keys())


def _fit_case(name):
    cs = CASES[name]
    X = np.array(cs["X"], dtype=np.float64)
    y = np.array(cs["y"], dtype=np.float64)
    weights = None if cs["weights"] is None else np.array(cs["weights"], dtype=np.float64)
    offset = None if cs["offset"] is None else np.array(cs["offset"], dtype=np.float64)
    return fit(X, y, family="negative-binomial", weights=weights, offset=offset,
               tol=1e-12, max_iter=300)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_theta(name):
    res = _fit_case(name)
    assert res.info.get("theta_estimated") is True
    np.testing.assert_allclose(
        res.info["theta"], RESULTS[name]["theta"], rtol=1e-4,
        err_msg=f"estimated theta differs from glm.nb ({name})")


@pytest.mark.parametrize("name", CASE_NAMES)
def test_coefficients(name):
    res = _fit_case(name)
    np.testing.assert_allclose(
        res.coefficients, np.array(RESULTS[name]["coefficients"]),
        rtol=1e-5, atol=1e-7, err_msg=f"coefficients differ from glm.nb ({name})")


@pytest.mark.parametrize("name", CASE_NAMES)
def test_standard_errors(name):
    res = _fit_case(name)
    np.testing.assert_allclose(
        res.standard_errors, np.array(RESULTS[name]["standard_errors"]),
        rtol=1e-4, atol=1e-6, err_msg=f"standard errors differ from glm.nb ({name})")


@pytest.mark.parametrize("name", CASE_NAMES)
def test_deviance(name):
    res = _fit_case(name)
    np.testing.assert_allclose(
        res.deviance, RESULTS[name]["deviance"], rtol=1e-5,
        err_msg=f"deviance differs from glm.nb ({name})")


@pytest.mark.parametrize("name", CASE_NAMES)
def test_aic(name):
    # glm.nb counts the estimated theta as a parameter (+2 over the fixed-θ AIC).
    res = _fit_case(name)
    np.testing.assert_allclose(
        res.aic, RESULTS[name]["aic"], rtol=1e-5,
        err_msg=f"AIC differs from glm.nb ({name})")


@pytest.mark.parametrize("name", CASE_NAMES)
def test_bic(name):
    res = _fit_case(name)
    np.testing.assert_allclose(
        res.bic, RESULTS[name]["bic"], rtol=1e-5,
        err_msg=f"BIC differs from glm.nb ({name})")
