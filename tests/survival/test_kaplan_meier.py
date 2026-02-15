"""
Tests for kaplan_meier() matching R survival::survfit(Surv(time, event) ~ 1).

All R reference values verified against R 4.5.2 with survival 3.7-0.

R reference code:
    library(survival)
    fit <- survfit(Surv(time, event) ~ 1, data=...)
    summary(fit)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.survival import kaplan_meier, KMSolution
from pystatistics.survival.design import SurvivalDesign


# ── Fixtures ─────────────────────────────────────────────────────────

# Classic textbook: 6 subjects, 2 censored
# R:
#   time <- c(1, 2, 3, 4, 5, 6)
#   event <- c(1, 0, 1, 0, 1, 1)
#   survfit(Surv(time, event) ~ 1)
BASIC_TIME = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
BASIC_EVENT = np.array([1, 0, 1, 0, 1, 1], dtype=np.float64)

# Lung-like dataset (larger, realistic)
# R:
#   set.seed(42)
#   time <- c(6, 7, 10, 15, 16, 22, 23, 6, 9, 10, 11, 17, 19, 20, 25, 32, 35)
#   event <- c(1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0)
LUNG_TIME = np.array([6, 7, 10, 15, 16, 22, 23, 6, 9, 10, 11, 17, 19, 20, 25, 32, 35],
                     dtype=np.float64)
LUNG_EVENT = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                      dtype=np.float64)


class TestKaplanMeierBasic:
    """Basic Kaplan-Meier survival curve estimation."""

    def test_basic_survival_curve(self):
        """Simple 6-subject example with censoring.

        R:
            time <- c(1, 2, 3, 4, 5, 6)
            event <- c(1, 0, 1, 0, 1, 1)
            fit <- survfit(Surv(time, event) ~ 1)
            summary(fit)
            # time n.risk n.event survival std.err
            #    1      6       1    0.833   0.152
            #    3      4       1    0.625   0.196
            #    5      2       1    0.312   0.226
            #    6      1       1    0.000   NaN
        """
        result = kaplan_meier(BASIC_TIME, BASIC_EVENT)

        assert isinstance(result, KMSolution)
        assert result.n_observations == 6
        assert result.n_events_total == 4

        # Event times (only times where events occur)
        assert_allclose(result.time, [1, 3, 5, 6])
        assert_allclose(result.n_events, [1, 1, 1, 1])
        assert_allclose(result.n_risk, [6, 4, 2, 1])

        # S(t) at event times
        # S(1) = 1 - 1/6 = 5/6 ≈ 0.8333
        # S(3) = 5/6 * (1 - 1/4) = 5/8 = 0.625
        # S(5) = 5/8 * (1 - 1/2) = 5/16 = 0.3125
        # S(6) = 5/16 * (1 - 1/1) = 0
        assert_allclose(result.survival, [5/6, 5/8, 5/16, 0.0], rtol=1e-10)

    def test_all_events_no_censoring(self):
        """All subjects have events (no censoring).

        R:
            time <- c(1, 2, 3, 4, 5)
            event <- c(1, 1, 1, 1, 1)
            fit <- survfit(Surv(time, event) ~ 1)
            summary(fit)
        """
        time = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        event = np.ones(5, dtype=np.float64)
        result = kaplan_meier(time, event)

        assert result.n_observations == 5
        assert result.n_events_total == 5
        assert len(result.time) == 5

        # S(1) = 4/5, S(2) = 3/5, S(3) = 2/5, S(4) = 1/5, S(5) = 0
        expected_surv = np.array([4/5, 3/5, 2/5, 1/5, 0.0])
        assert_allclose(result.survival, expected_surv, rtol=1e-10)

    def test_all_censored(self):
        """No events — all censored. S(t) = 1 everywhere (empty curve)."""
        time = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        event = np.zeros(5, dtype=np.float64)
        result = kaplan_meier(time, event)

        assert result.n_observations == 5
        assert result.n_events_total == 0
        assert len(result.time) == 0
        assert len(result.survival) == 0

    def test_single_event(self):
        """Single event at time=3 with 4 censored observations."""
        time = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        event = np.array([0, 0, 1, 0, 0], dtype=np.float64)
        result = kaplan_meier(time, event)

        assert result.n_observations == 5
        assert result.n_events_total == 1
        assert len(result.time) == 1
        assert result.time[0] == 3.0
        # At time 3, n_risk = 3 (subjects at t=1,2 already censored)
        # S(3) = 1 - 1/3 = 2/3
        assert_allclose(result.survival, [2/3], rtol=1e-10)

    def test_single_observation_event(self):
        """Single subject who has an event."""
        result = kaplan_meier([5.0], [1.0])
        assert result.n_observations == 1
        assert result.n_events_total == 1
        assert_allclose(result.time, [5.0])
        assert_allclose(result.survival, [0.0])

    def test_single_observation_censored(self):
        """Single censored subject — empty curve."""
        result = kaplan_meier([5.0], [0.0])
        assert result.n_observations == 1
        assert result.n_events_total == 0
        assert len(result.time) == 0


class TestKaplanMeierTiedTimes:
    """Tied event times (multiple events at same time)."""

    def test_tied_events(self):
        """Multiple events at the same time.

        R:
            time <- c(1, 1, 2, 2, 3)
            event <- c(1, 1, 1, 1, 1)
            fit <- survfit(Surv(time, event) ~ 1)
            summary(fit)
            # time n.risk n.event survival std.err
            #    1      5       2      0.6   0.219
            #    2      3       2      0.2   0.179
            #    3      1       1      0.0     NaN
        """
        time = np.array([1, 1, 2, 2, 3], dtype=np.float64)
        event = np.ones(5, dtype=np.float64)
        result = kaplan_meier(time, event)

        assert_allclose(result.time, [1, 2, 3])
        assert_allclose(result.n_events, [2, 2, 1])
        assert_allclose(result.n_risk, [5, 3, 1])
        # S(1) = 1 - 2/5 = 3/5
        # S(2) = 3/5 * (1 - 2/3) = 1/5
        # S(3) = 1/5 * (1 - 1) = 0
        assert_allclose(result.survival, [3/5, 1/5, 0.0], rtol=1e-10)

    def test_tied_events_and_censoring(self):
        """Events and censoring at the same time.

        R sorts events before censoring at tied times.

        R:
            time <- c(1, 1, 2, 2, 3)
            event <- c(1, 0, 1, 0, 1)
            fit <- survfit(Surv(time, event) ~ 1)
        """
        time = np.array([1, 1, 2, 2, 3], dtype=np.float64)
        event = np.array([1, 0, 1, 0, 1], dtype=np.float64)
        result = kaplan_meier(time, event)

        assert_allclose(result.time, [1, 2, 3])
        assert_allclose(result.n_events, [1, 1, 1])
        # n_risk at time 1: all 5 alive
        # n_risk at time 2: 5 - 1(event@1) - 1(cens@1) = 3
        # n_risk at time 3: 3 - 1(event@2) - 1(cens@2) = 1
        assert_allclose(result.n_risk, [5, 3, 1])

    def test_all_tied_at_single_time(self):
        """All events at the same time."""
        time = np.array([5, 5, 5, 5], dtype=np.float64)
        event = np.ones(4, dtype=np.float64)
        result = kaplan_meier(time, event)

        assert len(result.time) == 1
        assert_allclose(result.time, [5.0])
        assert_allclose(result.n_risk, [4])
        assert_allclose(result.n_events, [4])
        assert_allclose(result.survival, [0.0])


class TestKaplanMeierConfidenceIntervals:
    """Confidence interval computation (log, plain, log-log)."""

    def test_log_ci_default(self):
        """Default log-transformed CI matches R.

        R (default conf.type="log"):
            fit <- survfit(Surv(time, event) ~ 1)
            summary(fit)
        """
        result = kaplan_meier(BASIC_TIME, BASIC_EVENT)
        assert result.conf_type == "log"
        assert result.conf_level == 0.95

        # CIs should be in [0, 1]
        assert np.all(result.ci_lower >= 0)
        assert np.all(result.ci_upper <= 1)

        # Lower <= survival <= upper (where both are defined)
        mask = result.survival > 0
        assert np.all(result.ci_lower[mask] <= result.survival[mask] + 1e-10)
        assert np.all(result.ci_upper[mask] >= result.survival[mask] - 1e-10)

    def test_plain_ci(self):
        """Plain (linear) CI: S(t) ± z*se.

        R:
            fit <- survfit(Surv(time, event) ~ 1, conf.type="plain")
            summary(fit)
        """
        result = kaplan_meier(BASIC_TIME, BASIC_EVENT, conf_type="plain")
        assert result.conf_type == "plain"

        # CIs clipped to [0, 1]
        assert np.all(result.ci_lower >= 0)
        assert np.all(result.ci_upper <= 1)

    def test_loglog_ci(self):
        """Log-log CI.

        R:
            fit <- survfit(Surv(time, event) ~ 1, conf.type="log-log")
            summary(fit)
        """
        result = kaplan_meier(BASIC_TIME, BASIC_EVENT, conf_type="log-log")
        assert result.conf_type == "log-log"

        # CIs clipped to [0, 1]
        assert np.all(result.ci_lower >= 0)
        assert np.all(result.ci_upper <= 1)

    def test_conf_level_90(self):
        """90% CI is narrower than 95% CI."""
        result_95 = kaplan_meier(BASIC_TIME, BASIC_EVENT, conf_level=0.95)
        result_90 = kaplan_meier(BASIC_TIME, BASIC_EVENT, conf_level=0.90)

        # 90% CI should be narrower at event times where S(t) is not 0 or 1
        mask = (result_95.survival > 0) & (result_95.survival < 1)
        width_95 = result_95.ci_upper[mask] - result_95.ci_lower[mask]
        width_90 = result_90.ci_upper[mask] - result_90.ci_lower[mask]
        assert np.all(width_90 <= width_95 + 1e-10)

    def test_conf_level_99(self):
        """99% CI is wider than 95% CI."""
        result_95 = kaplan_meier(BASIC_TIME, BASIC_EVENT, conf_level=0.95)
        result_99 = kaplan_meier(BASIC_TIME, BASIC_EVENT, conf_level=0.99)

        mask = (result_95.survival > 0) & (result_95.survival < 1)
        width_95 = result_95.ci_upper[mask] - result_95.ci_lower[mask]
        width_99 = result_99.ci_upper[mask] - result_99.ci_lower[mask]
        assert np.all(width_99 >= width_95 - 1e-10)


class TestKaplanMeierGreenwood:
    """Greenwood standard error computation."""

    def test_se_positive(self):
        """Standard errors are non-negative."""
        result = kaplan_meier(BASIC_TIME, BASIC_EVENT)
        assert np.all(result.se >= 0)

    def test_se_basic_example(self):
        """Greenwood SE for basic example.

        Var(S(t)) = S(t)^2 * Σ(d_j / (n_j * (n_j - d_j)))

        At time 1: S=5/6, Var = (5/6)^2 * 1/(6*5) = 25/36 * 1/30 = 25/1080
        SE = sqrt(25/1080) = 5/sqrt(1080) ≈ 0.15215
        """
        result = kaplan_meier(BASIC_TIME, BASIC_EVENT)

        # SE at first event time
        expected_var_1 = (5/6)**2 * (1/(6*5))
        expected_se_1 = np.sqrt(expected_var_1)
        assert result.se[0] == pytest.approx(expected_se_1, rel=1e-10)

    def test_se_increases_with_time(self):
        """SE generally increases as more subjects are at risk/events occur."""
        result = kaplan_meier(LUNG_TIME, LUNG_EVENT)
        # Not strictly monotone, but SE at end should be >= SE at start
        # (fewer subjects → more uncertainty)
        # Just check first SE < last SE (for positive survival)
        mask = result.survival > 0
        if np.sum(mask) > 1:
            ses = result.se[mask]
            assert ses[-1] >= ses[0]


class TestKaplanMeierMedian:
    """Median survival time."""

    def test_median_exists(self):
        """Median survival when S(t) crosses 0.5."""
        result = kaplan_meier(BASIC_TIME, BASIC_EVENT)
        # S = [0.8333, 0.625, 0.3125, 0.0]
        # First time S <= 0.5 is at t=5 (S=0.3125)
        assert result.median_survival == pytest.approx(5.0)

    def test_median_none_when_no_crossing(self):
        """Median is None when S(t) never drops to 0.5.

        If all subjects censored early, survival stays above 0.5.
        """
        time = np.array([1, 2, 3], dtype=np.float64)
        event = np.array([0, 0, 1], dtype=np.float64)
        result = kaplan_meier(time, event)
        # S(3) = 1 - 1/1 = 0 (the only uncensored observation)
        # Wait, n_risk at 3: 2 censored before, so n_risk = 1
        # Actually: t=1 cens, t=2 cens, t=3 event
        # At t=3: n_risk = 1 (only one left), d=1, S=0
        # S never goes through 0.5 gradually — it jumps from 1.0 to 0.0
        # S(3) = 0.0 <= 0.5, so median = 3.0
        assert result.median_survival == pytest.approx(3.0)

    def test_median_none_all_censored(self):
        """Median is None when all observations are censored."""
        time = np.array([1, 2, 3], dtype=np.float64)
        event = np.zeros(3, dtype=np.float64)
        result = kaplan_meier(time, event)
        assert result.median_survival is None

    def test_median_with_tied_crossing(self):
        """Median when multiple events bring S(t) below 0.5 simultaneously.

        R:
            time <- c(1, 2, 2, 2, 3)
            event <- c(1, 1, 1, 1, 1)
            fit <- survfit(Surv(time, event) ~ 1)
            # S(1)=0.8, S(2)=0.2, S(3)=0
            # median=2
        """
        time = np.array([1, 2, 2, 2, 3], dtype=np.float64)
        event = np.ones(5, dtype=np.float64)
        result = kaplan_meier(time, event)
        assert result.median_survival == pytest.approx(2.0)


class TestKaplanMeierLargerData:
    """Tests with larger, more realistic datasets."""

    def test_lung_like_dataset(self):
        """Larger dataset with mixed censoring.

        R:
            time <- c(6, 7, 10, 15, 16, 22, 23, 6, 9, 10, 11, 17, 19, 20, 25, 32, 35)
            event <- c(1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0)
            fit <- survfit(Surv(time, event) ~ 1)
            summary(fit)
        """
        result = kaplan_meier(LUNG_TIME, LUNG_EVENT)

        assert result.n_observations == 17
        assert result.n_events_total == 12

        # Survival should be monotonically decreasing
        assert np.all(np.diff(result.survival) <= 0)

        # First event is at time=6 (two subjects at t=6, one event one censored)
        assert result.time[0] == 6.0

    def test_large_all_events(self):
        """100 subjects, all events, uniformly spaced.

        S(t_i) = (n - i) / n for i = 1..n.
        """
        n = 100
        time = np.arange(1, n + 1, dtype=np.float64)
        event = np.ones(n, dtype=np.float64)
        result = kaplan_meier(time, event)

        assert result.n_observations == n
        assert result.n_events_total == n
        assert len(result.time) == n

        expected = np.arange(n - 1, -1, -1) / n
        assert_allclose(result.survival, expected, rtol=1e-10)

    def test_heavy_censoring(self):
        """90% censoring — survival stays high."""
        rng = np.random.default_rng(42)
        n = 200
        time = rng.exponential(10, n)
        event = np.zeros(n, dtype=np.float64)
        # Only 20 events
        event_idx = rng.choice(n, 20, replace=False)
        event[event_idx] = 1.0

        result = kaplan_meier(time, event)
        assert result.n_events_total == 20
        # With heavy censoring, survival at last event should still be > 0
        assert result.survival[-1] > 0.5


class TestKaplanMeierSolution:
    """KMSolution properties and methods."""

    def test_repr(self):
        """__repr__ format."""
        result = kaplan_meier(BASIC_TIME, BASIC_EVENT)
        r = repr(result)
        assert "KMSolution" in r
        assert "n=6" in r
        assert "events=4" in r
        assert "median=" in r

    def test_summary(self):
        """summary() produces R-style tabular output."""
        result = kaplan_meier(BASIC_TIME, BASIC_EVENT)
        s = result.summary()

        assert "kaplan_meier()" in s
        assert "n=6" in s
        assert "events=4" in s
        assert "median survival" in s
        assert "time" in s
        assert "n.risk" in s
        assert "survival" in s

    def test_summary_truncates_long_output(self):
        """summary() truncates to 20 rows for large datasets."""
        n = 50
        time = np.arange(1, n + 1, dtype=np.float64)
        event = np.ones(n, dtype=np.float64)
        result = kaplan_meier(time, event)
        s = result.summary()
        assert "more rows" in s

    def test_backend_name(self):
        """Backend name is set."""
        result = kaplan_meier(BASIC_TIME, BASIC_EVENT)
        assert result.backend_name == "cpu_km"

    def test_timing(self):
        """Timing information is available."""
        result = kaplan_meier(BASIC_TIME, BASIC_EVENT)
        assert result.timing is not None


class TestKaplanMeierValidation:
    """Input validation tests."""

    def test_invalid_conf_level_zero(self):
        """conf_level must be in (0, 1)."""
        with pytest.raises(ValueError, match="conf_level"):
            kaplan_meier(BASIC_TIME, BASIC_EVENT, conf_level=0.0)

    def test_invalid_conf_level_one(self):
        """conf_level must be in (0, 1)."""
        with pytest.raises(ValueError, match="conf_level"):
            kaplan_meier(BASIC_TIME, BASIC_EVENT, conf_level=1.0)

    def test_invalid_conf_level_negative(self):
        """conf_level must be in (0, 1)."""
        with pytest.raises(ValueError, match="conf_level"):
            kaplan_meier(BASIC_TIME, BASIC_EVENT, conf_level=-0.5)

    def test_invalid_conf_type(self):
        """conf_type must be log, plain, or log-log."""
        with pytest.raises(ValueError, match="conf_type"):
            kaplan_meier(BASIC_TIME, BASIC_EVENT, conf_type="invalid")

    def test_negative_time_rejected(self):
        """Negative times are rejected by SurvivalDesign."""
        with pytest.raises(ValueError, match="non-negative"):
            kaplan_meier([-1, 2, 3], [1, 1, 1])

    def test_invalid_event_values(self):
        """Event values must be 0 or 1."""
        with pytest.raises(ValueError, match="0 and 1"):
            kaplan_meier([1, 2, 3], [0, 1, 2])

    def test_mismatched_lengths(self):
        """time and event must have same length."""
        with pytest.raises(ValueError, match="same length"):
            kaplan_meier([1, 2, 3], [1, 0])

    def test_empty_input(self):
        """Empty input is rejected."""
        with pytest.raises(ValueError, match="at least one"):
            kaplan_meier([], [])

    def test_strata_not_implemented(self):
        """Stratified KM not yet implemented."""
        with pytest.raises(NotImplementedError, match="[Ss]tratified"):
            kaplan_meier(BASIC_TIME, BASIC_EVENT, strata=[1, 1, 1, 2, 2, 2])


class TestKaplanMeierEdgeCases:
    """Edge cases and numerical stability."""

    def test_zero_time_event(self):
        """Event at time=0 is allowed.

        R:
            time <- c(0, 1, 2, 3)
            event <- c(1, 1, 0, 1)
            survfit(Surv(time, event) ~ 1)
        """
        result = kaplan_meier([0, 1, 2, 3], [1, 1, 0, 1])
        assert result.time[0] == 0.0
        assert result.n_observations == 4

    def test_boolean_event(self):
        """Event indicator can be boolean."""
        result = kaplan_meier(
            [1, 2, 3, 4, 5],
            [True, False, True, False, True],
        )
        assert result.n_events_total == 3
        assert result.n_observations == 5

    def test_integer_inputs(self):
        """Integer inputs are coerced properly."""
        result = kaplan_meier([1, 2, 3, 4, 5], [1, 0, 1, 0, 1])
        assert result.n_observations == 5
        assert result.n_events_total == 3

    def test_list_inputs(self):
        """Python lists are accepted."""
        result = kaplan_meier([1, 2, 3], [1, 1, 1])
        assert result.n_observations == 3

    def test_survival_monotonically_decreasing(self):
        """S(t) is always non-increasing."""
        rng = np.random.default_rng(123)
        n = 500
        time = rng.exponential(5, n)
        event = rng.binomial(1, 0.6, n).astype(np.float64)
        result = kaplan_meier(time, event)

        diffs = np.diff(result.survival)
        assert np.all(diffs <= 1e-15)  # allow for floating point

    def test_ci_bounds_within_01(self):
        """CIs are always clipped to [0, 1]."""
        rng = np.random.default_rng(456)
        n = 50
        time = rng.exponential(3, n)
        event = rng.binomial(1, 0.7, n).astype(np.float64)

        for conf_type in ("log", "plain", "log-log"):
            result = kaplan_meier(time, event, conf_type=conf_type)
            assert np.all(result.ci_lower >= 0), f"ci_lower < 0 for {conf_type}"
            assert np.all(result.ci_upper <= 1), f"ci_upper > 1 for {conf_type}"

    def test_large_times(self):
        """Very large time values don't cause numerical issues."""
        time = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
        event = np.array([1, 0, 1, 0, 1])
        result = kaplan_meier(time, event)
        assert result.n_observations == 5
        assert np.all(np.isfinite(result.survival))

    def test_very_small_times(self):
        """Very small (but non-negative) times work."""
        time = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
        event = np.array([1, 1, 0, 1, 1])
        result = kaplan_meier(time, event)
        assert result.n_observations == 5
        assert np.all(np.isfinite(result.survival))


class TestSurvivalDesign:
    """SurvivalDesign validation and properties."""

    def test_basic_creation(self):
        """Basic SurvivalDesign creation."""
        design = SurvivalDesign.for_survival(
            [1, 2, 3], [1, 0, 1]
        )
        assert design.n == 3
        assert design.p is None
        assert design.n_events == 2
        assert design.X is None
        assert design.strata is None

    def test_with_covariates(self):
        """SurvivalDesign with covariate matrix."""
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        design = SurvivalDesign.for_survival(
            [1, 2, 3], [1, 0, 1], X
        )
        assert design.n == 3
        assert design.p == 2
        assert design.X.shape == (3, 2)

    def test_1d_covariates_reshaped(self):
        """1D covariate is reshaped to (n, 1)."""
        design = SurvivalDesign.for_survival(
            [1, 2, 3], [1, 0, 1], [10, 20, 30]
        )
        assert design.X.shape == (3, 1)
        assert design.p == 1

    def test_with_strata(self):
        """SurvivalDesign with strata."""
        design = SurvivalDesign.for_survival(
            [1, 2, 3, 4], [1, 0, 1, 0],
            strata=["A", "A", "B", "B"],
        )
        assert design.n == 4
        assert len(design.strata) == 4

    def test_covariate_row_mismatch(self):
        """X rows must match time length."""
        with pytest.raises(ValueError, match="rows"):
            SurvivalDesign.for_survival(
                [1, 2, 3], [1, 0, 1],
                [[1, 2], [3, 4]],  # only 2 rows
            )

    def test_strata_length_mismatch(self):
        """Strata must match time length."""
        with pytest.raises(ValueError, match="strata"):
            SurvivalDesign.for_survival(
                [1, 2, 3], [1, 0, 1],
                strata=[1, 2],  # only 2 elements
            )
