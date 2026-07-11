"""Robust / cluster-robust Cox SE validated against R coxph(robust=, cluster=).

References in ``fixtures_robust/`` from ``generate_robust_fixtures.R``
(R 4.5.2 / survival 3.8.3). The sandwich SE, the retained model-based SE, and
the robust z / p all match R to machine precision, including the Efron tie
correction to the dfbeta residuals, stratified fits, and counting-process data.

R1 robust no cluster (+ Breslow variant); R1c heavy ties (Efron vs Breslow
dfbeta); R2 cluster(id) with repeated subjects; R3 robust + strata; R4
counting-process + cluster.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.survival import coxph

_FX = Path(__file__).parent / "fixtures_robust"


def _load(name: str):
    rows = list(csv.DictReader((_FX / f"{name}_data.csv").open()))
    return rows, json.load((_FX / f"{name}_ref.json").open())


def _col(rows, name):
    return np.array([float(r[name]) for r in rows])


def test_robust_se_no_cluster_matches_r():
    rows, r = _load("r1_robust")
    sol = coxph(_col(rows, "time"), _col(rows, "event"),
                np.column_stack([_col(rows, "x1"), _col(rows, "x2")]),
                robust=True, names=["x1", "x2"])
    assert sol.robust
    assert_allclose(sol.standard_errors, r["robust_se"], rtol=1e-8)
    assert_allclose(sol.naive_standard_errors, r["naive_se"], rtol=1e-8)
    assert_allclose(sol.z_values, r["robust_z"], rtol=1e-7)
    assert_allclose(sol.p_values, r["robust_p"], rtol=1e-6, atol=1e-14)


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_robust_se_heavy_ties_matches_r(ties):
    """The dfbeta residuals must carry the tie method (Efron gives a materially
    different sandwich than Breslow under heavy ties)."""
    rows, r = _load("r1c_ties")
    sol = coxph(_col(rows, "time"), _col(rows, "event"),
                np.column_stack([_col(rows, "x1"), _col(rows, "x2")]),
                robust=True, ties=ties)
    assert_allclose(sol.standard_errors, r[ties]["robust_se"], rtol=1e-8)


def test_cluster_robust_se_matches_r():
    rows, r = _load("r2_cluster")
    sol = coxph(_col(rows, "time"), _col(rows, "event"),
                np.column_stack([_col(rows, "x1"), _col(rows, "x2")]),
                cluster=[rw["id"] for rw in rows], names=["x1", "x2"])
    assert sol.robust  # cluster implies robust
    assert_allclose(sol.standard_errors, r["robust_se"], rtol=1e-8)
    assert_allclose(sol.naive_standard_errors, r["naive_se"], rtol=1e-8)


def test_robust_se_with_strata_matches_r():
    rows, r = _load("r3_robust_strata")
    sol = coxph(_col(rows, "time"), _col(rows, "event"),
                _col(rows, "x1").reshape(-1, 1),
                strata=[rw["g"] for rw in rows], robust=True)
    assert_allclose(sol.standard_errors, np.atleast_1d(r["robust_se"]),
                    rtol=1e-7)


def test_cluster_robust_counting_process_matches_r():
    rows, r = _load("r4_cp_cluster")
    sol = coxph(_col(rows, "stop"), _col(rows, "event"),
                np.column_stack([_col(rows, "x1"), _col(rows, "x2")]),
                start=_col(rows, "start"),
                cluster=[rw["id"] for rw in rows])
    assert_allclose(sol.standard_errors, r["robust_se"], rtol=1e-7)


def test_naive_se_unchanged_by_robust():
    """robust=True must not move the point estimate, and the naive SE must
    equal the ordinary (non-robust) fit's SE."""
    rows, _ = _load("r1_robust")
    X = np.column_stack([_col(rows, "x1"), _col(rows, "x2")])
    plain = coxph(_col(rows, "time"), _col(rows, "event"), X)
    rob = coxph(_col(rows, "time"), _col(rows, "event"), X, robust=True)
    assert_allclose(rob.coefficients, plain.coefficients, rtol=1e-12)
    assert not plain.robust
    assert_allclose(rob.naive_standard_errors, plain.standard_errors,
                    rtol=1e-10)


def test_cluster_length_validation():
    rows, _ = _load("r1_robust")
    from pystatistics.core.exceptions import ValidationError
    with pytest.raises(ValidationError, match="cluster"):
        coxph(_col(rows, "time"), _col(rows, "event"),
              _col(rows, "x1").reshape(-1, 1), cluster=[1, 2, 3])


def test_robust_counting_process_requires_cluster():
    """R refuses robust=TRUE on (start,stop] data without cluster/id ("one of
    cluster or id is needed") — a rowwise sandwich on correlated spells would
    be silently wrong. We refuse identically."""
    rows, _ = _load("r4_cp_cluster")
    from pystatistics.core.exceptions import ValidationError
    with pytest.raises(ValidationError, match="cluster"):
        coxph(_col(rows, "stop"), _col(rows, "event"),
              _col(rows, "x1").reshape(-1, 1),
              start=_col(rows, "start"), robust=True)
