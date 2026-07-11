"""cox_zph validated against R survival::cox.zph (survival >= 3.0 score test).

References in ``fixtures_zph/*_ref.json`` from ``generate_zph_fixtures.R``
(R 4.5.2 / survival 3.8.3). The chi-square table, p-values, scaled Schoenfeld
residual matrix (``y``), transformed times (``x``), and ``var`` all match R to
machine precision, so tolerances here are tight (1e-10).

Cases: Z1 plain fit x 4 transforms + residuals; Z2 stratified fit; Z3 heavy
ties under efron AND breslow (also pins the raw Schoenfeld residual tie
convention via reconstruction from the scaled matrix).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.core.exceptions import ValidationError
from pystatistics.survival import cox_zph, coxph

_FX = Path(__file__).parent / "fixtures_zph"


def _load(name: str, cov: list[str]):
    rows = list(csv.DictReader((_FX / f"{name}_data.csv").open()))
    time = np.array([float(r["time"]) for r in rows])
    event = np.array([float(r["event"]) for r in rows])
    X = np.array([[float(r[c]) for c in cov] for r in rows])
    strata = (np.array([r["g"] for r in rows])
              if "g" in rows[0] else None)
    ref = json.load((_FX / f"{name}_ref.json").open())
    return time, event, X, strata, ref


def _assert_table(z, r):
    assert_allclose(z.chisq, np.array(r["chisq"]), rtol=1e-10, atol=1e-12,
                    err_msg="chisq")
    assert_allclose(z.df, np.array(r["df"]), rtol=0, atol=0, err_msg="df")
    assert_allclose(z.p_values, np.array(r["p"]), rtol=1e-8, atol=1e-14,
                    err_msg="p")
    assert list(z.row_names) == r["rows"]


@pytest.mark.parametrize("transform", ["km", "identity", "rank", "log"])
def test_zph_plain_fit_matches_r(transform):
    time, event, X, _, ref = _load("z1", ["x1", "x2"])
    fit = coxph(time, event, X, names=["x1", "x2"])
    z = cox_zph(fit, transform=transform)
    _assert_table(z, ref[transform])


def test_zph_residuals_match_r():
    """Scaled Schoenfeld residual matrix, transformed times, and var."""
    time, event, X, _, ref = _load("z1", ["x1", "x2"])
    fit = coxph(time, event, X, names=["x1", "x2"])
    z = cox_zph(fit, transform="km")
    assert_allclose(z.residuals, np.array(ref["km_y"]), rtol=1e-10,
                    atol=1e-12, err_msg="scaled Schoenfeld residuals")
    assert_allclose(z.x, np.array(ref["km_x"]), rtol=1e-10, err_msg="x")
    assert_allclose(z.time, np.array(ref["km_time"]), rtol=1e-12,
                    err_msg="time")
    assert_allclose(z.var, np.array(ref["km_var"]), rtol=1e-10,
                    err_msg="var")


@pytest.mark.parametrize("transform", ["km", "rank"])
def test_zph_stratified_matches_r(transform):
    time, event, X, strata, ref = _load("z2", ["x1", "x2"])
    fit = coxph(time, event, X, strata=strata, names=["x1", "x2"])
    z = cox_zph(fit, transform=transform)
    _assert_table(z, ref[transform])


def test_zph_stratified_residuals_match_r():
    time, event, X, strata, ref = _load("z2", ["x1", "x2"])
    fit = coxph(time, event, X, strata=strata)
    z = cox_zph(fit, transform="km")
    assert_allclose(z.residuals, np.array(ref["km_y"]), rtol=1e-10,
                    atol=1e-12)
    assert_allclose(z.x, np.array(ref["km_x"]), rtol=1e-10)


@pytest.mark.parametrize("ties", ["efron", "breslow"])
@pytest.mark.parametrize("transform", ["km", "identity"])
def test_zph_heavy_ties_matches_r(ties, transform):
    """Heavy integer-time ties: the tie method must flow through the zph
    kernel (Efron and Breslow give materially different tables here)."""
    time, event, X, _, ref = _load("z3", ["x1", "x2"])
    fit = coxph(time, event, X, ties=ties, names=["x1", "x2"])
    z = cox_zph(fit, transform=transform)
    _assert_table(z, ref[ties][transform])


def test_zph_raw_schoenfeld_convention_matches_r():
    """Reconstruct raw Schoenfeld residuals from the scaled matrix and match
    R's residuals.coxph(type='schoenfeld') under both tie methods."""
    time, event, X, _, ref = _load("z3", ["x1", "x2"])
    for ties in ("efron", "breslow"):
        fit = coxph(time, event, X, ties=ties)
        z = cox_zph(fit, transform="km")
        raw = np.linalg.solve(z.var, (z.residuals - fit.coefficients).T).T
        assert_allclose(raw, np.array(ref[ties]["schoenfeld"]), rtol=1e-9,
                        atol=1e-12, err_msg=f"raw schoenfeld ({ties})")


def test_zph_input_validation():
    time, event, X, _, ref = _load("z1", ["x1", "x2"])
    fit = coxph(time, event, X)
    with pytest.raises(ValidationError, match="transform"):
        cox_zph(fit, transform="sqrt")
    with pytest.raises(ValidationError, match="CoxSolution"):
        cox_zph("not a fit")


def test_zph_summary_renders():
    time, event, X, _, _ = _load("z1", ["x1", "x2"])
    fit = coxph(time, event, X, names=["age", "sex"])
    z = cox_zph(fit)
    s = z.summary()
    assert "GLOBAL" in s and "age" in s and "km" in s
