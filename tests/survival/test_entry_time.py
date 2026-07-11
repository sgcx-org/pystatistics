"""Entry-time surfaces validated against R (survival 3.8.3, R 4.5.2).

Two public spellings share one validated entry field (CONVENTIONS A8):
- ``kaplan_meier(entry=...)`` — scalar delayed entry (left truncation),
  vs ``survfit(Surv(entry, time, event))``.
- ``coxph(start=...)`` — counting-process rows ``(start, time]`` enabling
  time-varying covariates, vs ``coxph(Surv(start, stop, event) ~ x)``.

References in ``fixtures_entry/`` from ``generate_entry_fixtures.R``.
E1 left-trunc KM; E2 left-trunc stratified KM; E3 time-varying-covariate Cox
(multi-spell subjects) + cox_zph on that fit; E4 counting-process + strata +
heavy ties; E5 degenerate interval (R NA-drops with a warning — PyStatistics
refuses loudly, a documented stricter deviation); E6 left-truncated Cox.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.core.exceptions import ValidationError
from pystatistics.survival import cox_zph, coxph, kaplan_meier

_FX = Path(__file__).parent / "fixtures_entry"


def _load(name: str):
    rows = list(csv.DictReader((_FX / f"{name}_data.csv").open()))
    ref = json.load((_FX / f"{name}_ref.json").open())
    return rows, ref


def _cols(rows, *names):
    return tuple(np.array([float(row[c]) for row in rows]) for c in names)


# ---------- KM: left truncation ------------------------------------------------

def test_left_truncated_km_matches_r():
    rows, ref = _load("e1_lt_km")
    entry, time, event = _cols(rows, "entry", "time", "event")
    sol = kaplan_meier(time, event, entry=entry)
    r = ref["km"]
    assert_allclose(sol.time, np.array(r["time"]), rtol=1e-12)
    assert_allclose(sol.survival, np.array(r["surv"]), rtol=1e-10)
    assert_allclose(sol.n_risk, np.array(r["n_risk"]), rtol=0)
    assert_allclose(sol.n_events, np.array(r["n_event"]), rtol=0)
    rse = np.array(r["std_err"]); v = rse >= 0
    assert_allclose(sol.se[v], rse[v], rtol=1e-6)
    rlo = np.array(r["lower"]); v = rlo >= 0
    assert_allclose(sol.ci_lower[v], rlo[v], rtol=1e-6)


def test_left_truncated_stratified_km_matches_r():
    rows, ref = _load("e2_lt_km_strata")
    entry, time, event = _cols(rows, "entry", "time", "event")
    strata = np.array([row["g"] for row in rows])
    sol = kaplan_meier(time, event, strata=strata, entry=entry)
    for lab in ref["km"]["strata"]:
        rc = ref["km"]["curves"][lab]
        km = sol[lab]
        assert_allclose(km.survival, np.array(rc["surv"]), rtol=1e-10,
                        err_msg=f"stratum {lab}")
        assert_allclose(km.n_risk, np.array(rc["n_risk"]), rtol=0,
                        err_msg=f"stratum {lab}")


# ---------- Cox: counting process ----------------------------------------------

def _assert_cox(sol, r):
    assert_allclose(sol.coefficients, np.atleast_1d(r["coefficients"]),
                    rtol=1e-6, err_msg="coefficients")
    assert_allclose(sol.standard_errors, np.atleast_1d(r["se"]), rtol=1e-6,
                    err_msg="se")
    assert_allclose(sol.loglik[1], r["loglik_model"], rtol=1e-8)
    assert_allclose(sol.concordance, r["concordance"], atol=1e-9)


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_time_varying_covariate_cox_matches_r(ties):
    """Subjects split into (start, stop] spells with a covariate that switches
    value mid-follow-up — the canonical time-dependent-covariate encoding."""
    rows, ref = _load("e3_tvc_cox")
    start, stop, event, x1, x2 = _cols(rows, "start", "stop", "event",
                                       "x1", "x2")
    sol = coxph(stop, event, np.column_stack([x1, x2]), start=start,
                ties=ties)
    _assert_cox(sol, ref[ties])


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_counting_process_stratified_heavy_ties_matches_r(ties):
    rows, ref = _load("e4_tvc_strata")
    start, stop, event, x1 = _cols(rows, "start", "stop", "event", "x1")
    strata = np.array([row["g"] for row in rows])
    sol = coxph(stop, event, x1.reshape(-1, 1), start=start, strata=strata,
                ties=ties)
    _assert_cox(sol, ref[ties])


def test_left_truncated_cox_matches_r():
    rows, ref = _load("e6_lt_cox")
    start, stop, event, x1, x2 = _cols(rows, "start", "stop", "event",
                                       "x1", "x2")
    sol = coxph(stop, event, np.column_stack([x1, x2]), start=start)
    _assert_cox(sol, ref["efron"])


def test_zph_on_counting_process_fit_matches_r():
    rows, _ = _load("e3_tvc_cox")
    start, stop, event, x1, x2 = _cols(rows, "start", "stop", "event",
                                       "x1", "x2")
    fit = coxph(stop, event, np.column_stack([x1, x2]), start=start,
                names=["x1", "x2"])
    rz = json.load((_FX / "e3_zph_ref.json").open())
    z = cox_zph(fit, transform="km")
    assert_allclose(z.chisq, np.array(rz["chisq"]), rtol=1e-10, atol=1e-12)
    assert_allclose(z.p_values, np.array(rz["p"]), rtol=1e-8, atol=1e-14)


# ---------- degenerate intervals (E5) -------------------------------------------

def test_degenerate_interval_fails_loud():
    """R turns entry >= stop into NA with a warning ("Stop time must be >
    start time, NA created") and silently drops the row. PyStatistics refuses
    loudly instead — a documented, deliberately stricter deviation (A6)."""
    with pytest.raises(ValidationError, match="strictly less"):
        coxph([1.0, 2.0], [1, 1], [[0.5], [-0.5]], start=[0.0, 2.0])
    with pytest.raises(ValidationError, match="strictly less"):
        kaplan_meier([1.0, 2.0], [1, 1], entry=[0.0, 2.0])


def test_entry_none_unchanged():
    """entry/start=None reproduces the plain fit exactly (no risk-set change)."""
    rows, _ = _load("e6_lt_cox")
    start, stop, event, x1, x2 = _cols(rows, "start", "stop", "event",
                                       "x1", "x2")
    X = np.column_stack([x1, x2])
    plain = coxph(stop, event, X)
    zeroed = coxph(stop, event, X, start=np.zeros(len(stop)))
    # start=0 for non-negative positive times is at-risk-from-0: identical fit.
    assert_allclose(plain.coefficients, zeroed.coefficients, rtol=1e-12)
    assert_allclose(plain.loglik[1], zeroed.loglik[1], rtol=1e-12)
    assert_allclose(plain.concordance, zeroed.concordance, rtol=1e-12)
