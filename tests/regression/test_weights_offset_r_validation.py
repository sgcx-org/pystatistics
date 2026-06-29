"""R-reference validation for prior weights (``weights=``) and offset (``offset=``).

Each case in ``tests/fixtures/weights_offset_cases.json`` was fit in R with
``glm.fit(X, y, weights=, offset=, intercept=FALSE)`` and its outputs stored in
``weights_offset_r_results.json`` (regenerate with
``tests/fixtures/run_r_weights_offset_validation.R``). These tests fit the same
data with PyStatistics and assert parity to CPU-vs-R round-off tolerance.

Gaussian cases are checked through the OLS/WLS path (``family=None``); the other
families through their GLM family object. AIC is asserted for every non-gaussian
family against R's ``glm.fit`` AIC: Gamma matches ``Gamma()$aic`` (dispersion =
``dev/sum(wt)`` plus a ``+2`` dispersion-parameter penalty) and negative
binomial matches ``MASS::negative.binomial(theta)$aic`` (no extra penalty for a
known theta). Gaussian AIC is not reported on the OLS ``LinearSolution`` and is
skipped here.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pystatistics.regression import fit
from pystatistics.regression.families import (
    GammaFamily, NegativeBinomial, LogLink, InverseLink,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
CASES_PATH = FIXTURES_DIR / "weights_offset_cases.json"
RESULTS_PATH = FIXTURES_DIR / "weights_offset_r_results.json"

pytestmark = pytest.mark.skipif(
    not (CASES_PATH.exists() and RESULTS_PATH.exists()),
    reason="weights/offset fixtures not generated",
)


def _load():
    cases = json.loads(CASES_PATH.read_text())
    results = json.loads(RESULTS_PATH.read_text())
    return cases, results


CASES, RESULTS = _load() if CASES_PATH.exists() and RESULTS_PATH.exists() else ({}, {})
CASE_NAMES = sorted(CASES.keys())

# AIC parity is asserted for every GLM family. Gaussian is listed so the
# parametrization covers it, but its assertion is skipped (no AIC on the OLS
# LinearSolution); the remaining families assert round-off parity with R.
AIC_CHECKED = {"gaussian", "poisson", "binomial", "Gamma", "negative.binomial"}


def _build_family(family, link, theta):
    if family == "gaussian":
        return None  # OLS / WLS path
    if family == "binomial":
        return "binomial"
    if family == "poisson":
        return "poisson"
    if family == "Gamma":
        return GammaFamily(link=LogLink() if link == "log" else InverseLink())
    if family == "negative.binomial":
        return NegativeBinomial(theta=theta, link="log")
    raise ValueError(f"unknown family {family}")


def _fit_case(name):
    cs = CASES[name]
    X = np.array(cs["X"], dtype=np.float64)
    y = np.array(cs["y"], dtype=np.float64)
    weights = None if cs["weights"] is None else np.array(cs["weights"], dtype=np.float64)
    offset = None if cs["offset"] is None else np.array(cs["offset"], dtype=np.float64)
    fam = _build_family(cs["family"], cs["link"], cs.get("theta"))
    return fit(X, y, family=fam, weights=weights, offset=offset,
               tol=1e-12, max_iter=200), cs


@pytest.mark.parametrize("name", CASE_NAMES)
def test_coefficients(name):
    res, _ = _fit_case(name)
    r = RESULTS[name]
    np.testing.assert_allclose(
        res.coefficients, np.array(r["coefficients"]),
        rtol=1e-6, atol=1e-8, err_msg=f"coefficients differ from R ({name})")


@pytest.mark.parametrize("name", CASE_NAMES)
def test_standard_errors(name):
    res, _ = _fit_case(name)
    r = RESULTS[name]
    np.testing.assert_allclose(
        res.standard_errors, np.array(r["standard_errors"]),
        rtol=1e-5, atol=1e-7, err_msg=f"standard errors differ from R ({name})")


@pytest.mark.parametrize("name", CASE_NAMES)
def test_fitted_values(name):
    res, _ = _fit_case(name)
    r = RESULTS[name]
    np.testing.assert_allclose(
        res.fitted_values, np.array(r["fitted_values"]),
        rtol=1e-6, atol=1e-8, err_msg=f"fitted values differ from R ({name})")


@pytest.mark.parametrize("name", CASE_NAMES)
def test_deviance(name):
    res, cs = _fit_case(name)
    r = RESULTS[name]
    # Gaussian via the OLS path: deviance == weighted RSS.
    dev = res.rss if cs["family"] == "gaussian" else res.deviance
    np.testing.assert_allclose(
        dev, r["deviance"], rtol=1e-6, err_msg=f"deviance differs from R ({name})")


@pytest.mark.parametrize("name", [n for n in CASE_NAMES if CASES[n]["family"] != "gaussian"])
def test_null_deviance(name):
    res, cs = _fit_case(name)
    r = RESULTS[name]
    null_r = float(r["null_deviance"])  # R serializes NaN as the string "NaN"
    if not np.isfinite(null_r):
        # Inverse-link Gamma: linkinv(0) is infinite, so the intercept-free
        # null model is undefined — R returns NaN; we don't assert on it.
        pytest.skip("R null deviance is undefined (inverse-link null model)")
    np.testing.assert_allclose(
        res.null_deviance, null_r, rtol=1e-6,
        err_msg=f"null deviance differs from R ({name})")


@pytest.mark.parametrize(
    "name",
    [n for n in CASE_NAMES
     if CASES[n]["family"] == "gaussian" and CASES[n]["offset"] is None],
)
def test_weighted_r_squared(name):
    """OLS/WLS R² matches R's weighted lm() to round-off (no-offset cases).

    With an offset the residual is not weighted-orthogonal to the offset, so R's
    own ``summary.lm`` R² is idiosyncratic (and documented as not meaningful with
    an offset). We report the principled mss/(mss+rss) decomposition there, which
    coincides with R only when there is no offset — so strict parity is asserted
    for the no-offset cases only."""
    res, _ = _fit_case(name)
    r = RESULTS[name]
    np.testing.assert_allclose(
        res.r_squared, float(r["r_squared"]), rtol=1e-6,
        err_msg=f"weighted R² differs from R's lm() ({name})")


@pytest.mark.parametrize("name", [n for n in CASE_NAMES if CASES[n]["family"] in AIC_CHECKED])
def test_aic(name):
    res, cs = _fit_case(name)
    if cs["family"] == "gaussian":
        pytest.skip("AIC not reported on the OLS LinearSolution")
    r = RESULTS[name]
    np.testing.assert_allclose(
        res.aic, r["aic"], rtol=1e-6, err_msg=f"AIC differs from R ({name})")
