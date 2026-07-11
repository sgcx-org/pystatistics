"""Stratified Cox PH validated against R survival::coxph(... + strata()).

Reference values in ``fixtures_stratified/*_ref.json`` were produced by
``fixtures_stratified/generate_stratified_fixtures.R`` under R 4.5.2 /
survival 3.8.3 (Efron and Breslow ties). Each ``*_data.csv`` holds the exact
inputs both engines fit. Tolerances mirror the unstratified R-validation suite
(``test_r_validation.py``): coefficients rtol=1e-4, SE rtol=1e-3, loglik
rel=1e-4; concordance now matches R's convention exactly (abs=1e-6).

Cases: S1 basic 2-strata; S2 degenerate strata (all-censored stratum + singleton
event stratum); S3 heavy integer-time ties, 3 strata (Efron != Breslow); S4
shared-beta monotone likelihood (separation) — R warns "coefficient may be
infinite"; S6 one-level strata == the plain unstratified fit.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.survival import coxph

_FX = Path(__file__).parent / "fixtures_stratified"


def _load(name: str, covariates: list[str], timecol: str, eventcol: str,
          stratacol: str):
    """Read a fixture CSV into (time, event, X, strata) numpy arrays."""
    rows = list(csv.DictReader((_FX / f"{name}_data.csv").open()))
    time = np.array([float(r[timecol]) for r in rows])
    event = np.array([float(r[eventcol]) for r in rows])
    X = np.array([[float(r[c]) for c in covariates] for r in rows])
    strata = np.array([r[stratacol] for r in rows])
    ref = json.load((_FX / f"{name}_ref.json").open())
    return time, event, X, strata, ref


def _assert_matches(sol, r):
    assert_allclose(sol.coefficients, np.atleast_1d(r["coefficients"]),
                    rtol=1e-4, err_msg="coefficients")
    assert_allclose(sol.standard_errors, np.atleast_1d(r["se"]),
                    rtol=1e-3, err_msg="standard errors")
    assert_allclose(sol.loglik[1], r["loglik_model"], rtol=1e-4,
                    err_msg="model loglik")
    assert_allclose(sol.loglik[0], r["loglik_null"], rtol=1e-4,
                    err_msg="null loglik")
    if "concordance" in r:
        assert_allclose(sol.concordance, r["concordance"], atol=1e-6,
                        err_msg="concordance")


@pytest.mark.parametrize("ties", ["efron", "breslow"])
@pytest.mark.parametrize("name,cov,tcol,ecol", [
    ("s1_basic", ["x1", "x2"], "time", "event"),
    ("s2_degenerate", ["x1"], "time", "event"),
    ("s3_heavyties", ["x1", "x2"], "time", "event"),
])
def test_stratified_matches_r(name, cov, tcol, ecol, ties):
    time, event, X, strata, ref = _load(name, cov, tcol, ecol, "g")
    sol = coxph(time, event, X, strata=strata, ties=ties, names=cov)
    _assert_matches(sol, ref[ties])
    assert sol.n_strata == len(np.unique(strata))


def test_efron_breslow_differ_on_heavy_ties():
    """With heavy ties, Efron and Breslow give materially different fits;
    each must match its own R reference."""
    time, event, X, strata, ref = _load(
        "s3_heavyties", ["x1", "x2"], "time", "event", "g")
    ef = coxph(time, event, X, strata=strata, ties="efron")
    br = coxph(time, event, X, strata=strata, ties="breslow")
    # sanity: the two methods genuinely disagree here
    assert abs(ef.coefficients[1] - br.coefficients[1]) > 0.05
    _assert_matches(ef, ref["efron"])
    _assert_matches(br, ref["breslow"])


def test_degenerate_strata_match_r():
    """Stratum with zero events + singleton stratum contribute nothing and the
    remaining strata still match R exactly."""
    time, event, X, strata, ref = _load(
        "s2_degenerate", ["x1"], "time", "event", "g")
    sol = coxph(time, event, X, strata=strata, ties="efron", names=["x1"])
    _assert_matches(sol, ref["efron"])
    assert sol.n_strata == 4


def test_one_level_strata_equals_plain_fit():
    """strata with a single level reduces exactly to the unstratified fit."""
    time, event, X, strata, ref = _load(
        "s6_onestratum", ["x1"], "time", "event", "g")
    strat = coxph(time, event, X, strata=strata, ties="efron")
    plain = coxph(time, event, X, ties="efron")
    assert_allclose(strat.coefficients, plain.coefficients, rtol=1e-10)
    assert_allclose(strat.loglik[1], plain.loglik[1], rtol=1e-10)
    assert strat.n_strata == 1
    _assert_matches(strat, ref["efron_strata"])


def test_separation_matches_r_divergence_and_warning():
    """Shared-beta monotone likelihood: R drives the coefficient to a large
    value and warns 'coefficient may be infinite'. pystatistics must reproduce
    both the value (loglik) and the warning (RIGOR R10)."""
    time, event, X, strata, ref = _load(
        "s4_nearsep", ["x1"], "time", "event", "g")
    sol = coxph(time, event, X, strata=strata, ties="efron", names=["x1"])
    r = ref["efron"]
    # The coefficient itself is essentially infinite (R gets ~21.9); compare the
    # stable quantity — the plateaued log-likelihood — not the runaway coef.
    assert_allclose(sol.loglik[1], r["loglik_model"], rtol=1e-5)
    assert abs(sol.coefficients[0]) > 10.0
    assert any("infinite" in w for w in sol.warnings)
