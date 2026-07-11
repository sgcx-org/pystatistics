"""Stratified Kaplan-Meier validated against R survival::survfit(Surv~g).

Reference values in ``fixtures_km_strata/*_ref.json`` come from
``generate_km_strata_fixtures.R`` (R 4.5.2 / survival 3.8.3), extracted from
``summary(survfit(...))`` — one row per distinct EVENT time, ``std.err`` on the
survival scale — matching the pystatistics KM convention. Tolerances mirror the
unstratified KM suite: survival/time/n_risk rtol=1e-10, se rtol=1e-6, CI rtol=1e-6
(compared only where R reports a finite value).

Cases: K1 two strata + ties/censoring; K2 three strata incl. an all-censored
stratum (empty curve) + log-log CI; K3 heavy integer-time ties; K5 single-level
strata == the unstratified fit.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.survival import kaplan_meier, StratifiedKMSolution

_FX = Path(__file__).parent / "fixtures_km_strata"


def _load(name: str):
    rows = list(csv.DictReader((_FX / f"{name}_data.csv").open()))
    time = np.array([float(r["time"]) for r in rows])
    event = np.array([float(r["event"]) for r in rows])
    strata = np.array([r["g"] for r in rows])
    ref = json.load((_FX / f"{name}_ref.json").open())
    return time, event, strata, ref


def _match_key(label: str, sol):
    """R labels are strings; the solution may key strata by int."""
    if label in sol.curves:
        return label
    if label.lstrip("-").isdigit():
        return int(label)
    return label


def _assert_curve(km, rc):
    r_time = np.array(rc["time"], dtype=float)
    assert_allclose(km.time, r_time, rtol=1e-10, err_msg="time")
    assert_allclose(km.survival, np.array(rc["surv"]), rtol=1e-10,
                    err_msg="survival")
    assert_allclose(km.n_risk, np.array(rc["n_risk"]), rtol=1e-10,
                    err_msg="n_risk")
    assert_allclose(km.n_events, np.array(rc["n_event"]), rtol=1e-10,
                    err_msg="n_event")
    rse = np.array(rc["std_err"]); valid = rse >= 0
    if valid.any():
        assert_allclose(km.se[valid], rse[valid], rtol=1e-6, err_msg="se")
    rlo = np.array(rc["lower"]); vlo = rlo >= 0
    if vlo.any():
        assert_allclose(km.ci_lower[vlo], rlo[vlo], rtol=1e-6, err_msg="ci_lower")
    rup = np.array(rc["upper"]); vup = rup >= 0
    if vup.any():
        assert_allclose(km.ci_upper[vup], rup[vup], rtol=1e-6, err_msg="ci_upper")


@pytest.mark.parametrize("name,conf_type", [
    ("k1_basic", "log"),
    ("k3_heavyties", "log"),
    ("k2_censored", "log-log"),
])
def test_stratified_km_matches_r(name, conf_type):
    time, event, strata, ref = _load(name)
    sol = kaplan_meier(time, event, strata=strata, conf_type=conf_type)
    assert isinstance(sol, StratifiedKMSolution)
    assert sol.n_strata == len(ref["strata"])
    for label in ref["strata"]:
        rc = ref["curves"][label]
        km = sol[_match_key(label, sol)]
        if len(rc["time"]) == 0:
            assert len(km.time) == 0  # all-censored stratum: empty curve
            continue
        _assert_curve(km, rc)


def test_all_censored_stratum_is_empty():
    """A stratum with no events yields an empty curve (as R's summary does),
    not an error."""
    time, event, strata, ref = _load("k2_censored")
    sol = kaplan_meier(time, event, strata=strata, conf_type="log-log")
    km3 = sol[_match_key("3", sol)]
    assert len(km3.time) == 0
    assert km3.n_events_total == 0


def test_one_level_strata_equals_plain():
    """Single-level strata reduces exactly to the unstratified curve."""
    time, event, strata, ref = _load("k5_onelevel")
    sol = kaplan_meier(time, event, strata=strata)
    plain = kaplan_meier(time, event)
    km = sol[_match_key("1", sol)]
    assert sol.n_strata == 1
    assert_allclose(km.survival, plain.survival, rtol=1e-12)
    assert_allclose(km.time, plain.time, rtol=1e-12)
    # ... and both equal R's unstratified survfit
    assert_allclose(km.survival, np.array(ref["plain"]["surv"]), rtol=1e-10)


def test_solution_accessors():
    time, event, strata, ref = _load("k1_basic")
    sol = kaplan_meier(time, event, strata=strata)
    assert set(sol.strata) == {"A", "B"}
    assert sol["A"] is sol.stratum("A")
    assert sol.curves["A"].n_observations > 0
    assert isinstance(sol.summary(), str)
    assert isinstance(sol.warnings, tuple)
