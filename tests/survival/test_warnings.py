"""
Tests for the .warnings property on survival Solution classes.

Every survival Solution (KMSolution, LogRankSolution, CoxSolution,
DiscreteTimeSolution) exposes a .warnings tuple delegating to the underlying
Result, matching every other domain. These tests verify both that warnings
the solvers populate are reachable through the Solution, and that a clean fit
exposes an empty tuple.
"""

import numpy as np

from pystatistics.survival import (
    coxph,
    discrete_time,
    kaplan_meier,
    survdiff,
)


# ── Fixtures ─────────────────────────────────────────────────────────

# Balanced two-group example (12 subjects per group, all events): expected
# counts are comfortably above 5 and every group has events, so survdiff
# raises no warnings.
BALANCED_TIME = np.array(
    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23,
     2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
    dtype=np.float64,
)
BALANCED_EVENT = np.ones(24, dtype=np.float64)
BALANCED_GROUP = np.array([1] * 12 + [2] * 12)

# Tiny group A (1 subject, 1 event) against a large group B: group A's
# expected event count is well below 5, triggering the small-expected
# warning, while every group still has an event.
SMALL_EXP_TIME = np.array([5, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11], dtype=np.float64)
SMALL_EXP_EVENT = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
SMALL_EXP_GROUP = np.array(["A", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B"])

# Group A is entirely censored (zero observed events) against an active
# group B, triggering the zero-events warning.
ZERO_EVT_TIME = np.array([3, 5, 7, 1, 2, 4, 6, 8, 9, 10], dtype=np.float64)
ZERO_EVT_EVENT = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
ZERO_EVT_GROUP = np.array(["A", "A", "A", "B", "B", "B", "B", "B", "B", "B"])

# Simple Cox fixture (single binary covariate). The covariate is deliberately
# NON-separating (alternating 0/1) so the fit is genuinely clean — R's coxph
# converges in a few iterations with no warning. (An earlier version used a
# perfectly separating covariate, which R warns on: "coefficient may be
# infinite"; that is now covered by SEP_X below.)
COX_TIME = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
COX_EVENT = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1], dtype=np.float64)
COX_X = np.array([[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]],
                 dtype=np.float64)

# Perfectly separating covariate (low x dies early) — a monotone likelihood.
# R's coxph warns "coefficient may be infinite" and runs to the iteration cap.
SEP_X = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]],
                 dtype=np.float64)

# Discrete-time fixture (person-period logistic).
DT_TIME = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=np.float64)
DT_EVENT = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1], dtype=np.float64)
DT_X = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]],
                dtype=np.float64)


# ── survdiff / LogRankSolution ───────────────────────────────────────

class TestLogRankWarnings:
    """survdiff warnings reach the LogRankSolution."""

    def test_clean_fit_has_no_warnings(self):
        result = survdiff(BALANCED_TIME, BALANCED_EVENT, BALANCED_GROUP)
        assert result.warnings == ()

    def test_small_expected_count_warns(self):
        result = survdiff(SMALL_EXP_TIME, SMALL_EXP_EVENT, SMALL_EXP_GROUP)
        assert any("small expected event count" in w for w in result.warnings)

    def test_zero_observed_events_warns(self):
        result = survdiff(ZERO_EVT_TIME, ZERO_EVT_EVENT, ZERO_EVT_GROUP)
        assert any("zero observed events" in w for w in result.warnings)

    def test_warnings_is_tuple(self):
        result = survdiff(BALANCED_TIME, BALANCED_EVENT, BALANCED_GROUP)
        assert isinstance(result.warnings, tuple)


# ── coxph / CoxSolution ──────────────────────────────────────────────

class TestCoxWarnings:
    """coxph convergence warnings reach the CoxSolution."""

    def test_clean_fit_has_no_warnings(self):
        result = coxph(COX_TIME, COX_EVENT, COX_X)
        assert result.warnings == ()

    def test_non_convergence_warns(self):
        # One Newton-Raphson iteration is not enough to converge here, so
        # the solver records a non-convergence warning that must surface.
        result = coxph(COX_TIME, COX_EVENT, COX_X, max_iter=1)
        assert not result.converged
        assert any("converge" in w for w in result.warnings)

    def test_separation_warns_like_r(self):
        # Perfectly separating covariate: R's coxph warns "coefficient may be
        # infinite" and drives the coefficient to a large magnitude. pystatistics
        # must match that warning (RIGOR R10), not silently return the value.
        result = coxph(COX_TIME, COX_EVENT, SEP_X, names=["x"])
        assert any("infinite" in w for w in result.warnings)
        assert abs(result.coefficients[0]) > 10.0


# ── kaplan_meier / KMSolution ────────────────────────────────────────

class TestKMWarnings:
    """KMSolution exposes an empty warnings tuple on a clean fit."""

    def test_clean_fit_has_no_warnings(self):
        result = kaplan_meier(BALANCED_TIME, BALANCED_EVENT)
        assert result.warnings == ()
        assert isinstance(result.warnings, tuple)


# ── discrete_time / DiscreteTimeSolution ─────────────────────────────

class TestDiscreteTimeWarnings:
    """DiscreteTimeSolution exposes an empty warnings tuple on a clean fit."""

    def test_clean_fit_has_no_warnings(self):
        result = discrete_time(DT_TIME, DT_EVENT, DT_X)
        assert result.warnings == ()
        assert isinstance(result.warnings, tuple)
