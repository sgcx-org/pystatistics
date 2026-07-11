"""Fail-loud input validation and edge-case fidelity for the survival module.

These pin the fixes for issues an adversarial review surfaced: non-finite
time/event, missing stratum labels, the covariate-centering that keeps the Cox
information well-conditioned on large-magnitude covariates (matching R rather
than silently returning se=0), the median-survival minmin convention, and the
degenerate no-events fit reporting converged=False (not a fabricated fit).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.core.exceptions import ValidationError
from pystatistics.survival import coxph, kaplan_meier

_T = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
_E = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
_X = np.array([[0.0], [1.0], [0.5], [1.5], [2.0]])


# ---- fail-loud input validation ----------------------------------------------

@pytest.mark.parametrize("bad_time", [
    [1.0, 2.0, np.nan, 4.0, 5.0],
    [1.0, 2.0, np.inf, 4.0, 5.0],
])
def test_non_finite_time_rejected(bad_time):
    with pytest.raises(ValidationError, match="finite"):
        kaplan_meier(bad_time, _E)
    with pytest.raises(ValidationError, match="finite"):
        coxph(bad_time, _E, _X)


def test_nan_event_rejected():
    bad = np.array([1.0, np.nan, 1.0, 1.0, 0.0])
    with pytest.raises(ValidationError, match="event"):
        kaplan_meier(_T, bad)


def test_nan_strata_rejected():
    with pytest.raises(ValidationError, match="strata"):
        coxph(_T, _E, _X, strata=[1.0, 1.0, np.nan, 2.0, 2.0])
    with pytest.raises(ValidationError, match="strata"):
        kaplan_meier(_T, _E, strata=np.array([1.0, np.nan, 2.0, 2.0, 1.0]))


def test_none_object_strata_rejected():
    with pytest.raises(ValidationError, match="strata"):
        coxph(_T, _E, _X, strata=np.array(["a", "a", None, "b", "b"],
                                          dtype=object))


# ---- covariate centering keeps SEs correct on large-magnitude covariates -----

def test_large_covariate_se_matches_small_covariate():
    """A covariate shifted by a huge constant (timestamp / genomic-coordinate
    magnitude) must give the SAME fit as the unshifted one — the Cox likelihood
    is shift-invariant, and centering keeps the information matrix
    well-conditioned instead of collapsing the SE to 0."""
    rng = np.random.default_rng(0)
    n = 300
    signal = rng.normal(size=n)
    t = -np.log(rng.uniform(size=n)) / np.exp(1.3 * signal)
    e = np.ones(n)
    base = coxph(t, e, signal.reshape(-1, 1))
    shifted = coxph(t, e, (signal + 3e9).reshape(-1, 1))
    assert_allclose(shifted.coefficients, base.coefficients, rtol=1e-6)
    assert_allclose(shifted.standard_errors, base.standard_errors, rtol=1e-5)
    assert np.isfinite(shifted.standard_errors).all()
    assert shifted.standard_errors[0] > 0  # not the silent-wrong se=0


# ---- degenerate no-events fit is flagged, not fabricated ---------------------

def test_no_events_reports_not_converged():
    sol = coxph(_T, np.zeros(5), _X)
    assert sol.n_events == 0
    assert not sol.converged            # was a fabricated converged=True
    assert np.isnan(sol.concordance)    # was a fabricated 0.5
    assert_allclose(sol.coefficients, [0.0], atol=1e-12)


def test_no_events_stratified_reports_not_converged():
    sol = coxph(_T, np.zeros(5), _X, strata=[1, 1, 1, 2, 2])
    assert not sol.converged
    assert np.isnan(sol.concordance)
    assert sol.n_strata == 2


# ---- median-survival minmin convention ---------------------------------------

def test_median_survival_averages_when_curve_touches_half():
    # 2 subjects, both events: S = 0.5 at t=1, 0 at t=2. R averages -> 1.5.
    km = kaplan_meier([1.0, 2.0], [1.0, 1.0])
    assert km.median_survival == pytest.approx(1.5)


def test_median_survival_no_average_when_curve_crosses_half():
    # S jumps 0.75 -> 0.375 (never equals 0.5): median is the crossing time.
    km = kaplan_meier([1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 1.0, 1.0])
    assert km.median_survival == pytest.approx(3.0)


def test_median_survival_none_when_never_reaches_half():
    km = kaplan_meier([1.0, 2.0, 3.0], [1.0, 0.0, 0.0])
    assert km.median_survival is None
