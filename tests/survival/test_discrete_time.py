"""
Tests for discrete_time() — person-period logistic regression survival model.

The discrete-time model converts time-to-event data into a person-period
long-format dataset and fits logistic regression. This is the GPU-accelerated
survival method.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.survival import discrete_time, DiscreteTimeSolution


# ── Fixtures ─────────────────────────────────────────────────────────

# Simple example: 10 subjects, 2 covariates
DT_TIME = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=np.float64)
DT_EVENT = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1], dtype=np.float64)
DT_X = np.column_stack([
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],    # binary covariate
    [1.2, 0.5, -0.3, 0.8, -1.0, 0.3, -0.5, 1.1, -0.8, 0.2],  # continuous
]).astype(np.float64)


class TestDiscreteTimeBasic:
    """Basic discrete-time survival model."""

    def test_basic_fit(self):
        """Basic model fitting produces valid results."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X)

        assert isinstance(result, DiscreteTimeSolution)
        assert result.n_observations == 10
        assert result.n_events == 7  # 7 events
        assert len(result.coefficients) == 2  # 2 covariates
        assert result.n_intervals > 0
        assert result.person_period_n > result.n_observations  # expanded dataset

    def test_coefficients_finite(self):
        """All coefficients are finite."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X)
        assert np.all(np.isfinite(result.coefficients))
        assert np.all(np.isfinite(result.standard_errors))

    def test_hazard_ratios_consistent(self):
        """hazard_ratios = exp(coefficients)."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X)
        assert_allclose(result.hazard_ratios, np.exp(result.coefficients),
                       rtol=1e-10)

    def test_baseline_hazard_range(self):
        """Baseline hazard values are in (0, 1)."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X)
        assert np.all(result.baseline_hazard > 0)
        assert np.all(result.baseline_hazard < 1)

    def test_z_statistics_consistent(self):
        """z = coef / se."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X)
        mask = result.standard_errors > 0
        expected_z = result.coefficients[mask] / result.standard_errors[mask]
        assert_allclose(result.z_statistics[mask], expected_z, rtol=1e-10)

    def test_p_values_valid(self):
        """p-values in [0, 1]."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X)
        assert np.all(result.p_values >= 0)
        assert np.all(result.p_values <= 1)

    def test_interval_labels(self):
        """Interval labels match unique event times."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X)
        # Default intervals = unique event times
        expected_times = np.unique(DT_TIME[DT_EVENT == 1])
        assert_allclose(result.interval_labels, expected_times)


class TestDiscreteTimeIntervals:
    """Custom interval specification."""

    def test_custom_intervals(self):
        """User-specified intervals."""
        intervals = np.array([1, 3, 5])
        result = discrete_time(DT_TIME, DT_EVENT, DT_X, intervals=intervals)

        assert result.n_intervals == 3
        assert_allclose(result.interval_labels, [1, 3, 5])

    def test_fewer_intervals(self):
        """Fewer intervals than default gives fewer baseline params."""
        result_default = discrete_time(DT_TIME, DT_EVENT, DT_X)
        result_fewer = discrete_time(DT_TIME, DT_EVENT, DT_X,
                                     intervals=[1, 3, 5])

        assert result_fewer.n_intervals <= result_default.n_intervals


class TestDiscreteTimePersonPeriod:
    """Person-period expansion correctness."""

    def test_expansion_size(self):
        """Person-period dataset is larger than original.

        Each subject contributes rows for each interval they're at risk.
        """
        result = discrete_time(DT_TIME, DT_EVENT, DT_X)
        assert result.person_period_n > 10
        # With 5 event times and 10 subjects, max is 50 rows
        assert result.person_period_n <= 50

    def test_all_events_at_single_time(self):
        """All events at same time — single interval."""
        time = np.array([5, 5, 5, 5, 5], dtype=np.float64)
        event = np.ones(5, dtype=np.float64)
        X = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)

        result = discrete_time(time, event, X)
        assert result.n_intervals == 1
        assert result.person_period_n == 5  # one row per subject


class TestDiscreteTimeBackend:
    """Backend selection."""

    def test_cpu_backend(self):
        """Explicit CPU backend."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X, backend="cpu")
        assert result.n_observations == 10
        assert np.all(np.isfinite(result.coefficients))

    def test_auto_backend(self):
        """Auto backend (default)."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X, backend="auto")
        assert result.n_observations == 10


class TestDiscreteTimeSolution:
    """DiscreteTimeSolution properties and methods."""

    def test_repr(self):
        """__repr__ format."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X)
        r = repr(result)
        assert "DiscreteTimeSolution" in r
        assert "n=" in r
        assert "events=" in r
        assert "intervals=" in r

    def test_summary(self):
        """summary() produces formatted output."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X)
        s = result.summary()

        assert "discrete_time()" in s
        assert "events=" in s
        assert "intervals=" in s
        assert "person-period" in s
        assert "coef" in s
        assert "exp(coef)" in s
        assert "Deviance" in s
        assert "AIC" in s

    def test_glm_diagnostics(self):
        """GLM deviance and AIC are available."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X)
        assert result.glm_deviance > 0
        assert result.glm_aic > 0

    def test_timing(self):
        """Timing information available."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X)
        assert result.timing is not None


class TestDiscreteTimeValidation:
    """Input validation."""

    def test_no_covariates(self):
        """X is required."""
        with pytest.raises(ValueError, match="[Cc]ovariate|X"):
            discrete_time([1, 2, 3], [1, 1, 1], None)

    def test_negative_time_rejected(self):
        """Negative times rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            discrete_time([-1, 2, 3], [1, 1, 1], [[1], [2], [3]])

    def test_invalid_event_values(self):
        """Event must be 0/1."""
        with pytest.raises(ValueError, match="0 and 1"):
            discrete_time([1, 2, 3], [0, 1, 2], [[1], [2], [3]])


class TestDiscreteTimeEdgeCases:
    """Edge cases."""

    def test_all_censored(self):
        """All censored — degenerate result."""
        time = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        event = np.zeros(5, dtype=np.float64)
        X = np.ones((5, 1), dtype=np.float64)

        result = discrete_time(time, event, X)
        assert result.n_events == 0
        assert result.n_intervals == 0

    def test_single_covariate(self):
        """Single covariate works."""
        result = discrete_time(DT_TIME, DT_EVENT, DT_X[:, :1])
        assert len(result.coefficients) == 1

    def test_many_time_points(self):
        """Many unique event times."""
        rng = np.random.default_rng(42)
        n = 100
        time = rng.exponential(5, n)
        event = rng.binomial(1, 0.5, n).astype(np.float64)
        X = rng.standard_normal((n, 2))

        result = discrete_time(time, event, X)
        assert result.n_observations == 100
        assert len(result.coefficients) == 2

    def test_list_inputs(self):
        """Python lists accepted."""
        result = discrete_time(
            [1, 2, 3, 4, 5],
            [1, 1, 0, 1, 1],
            [[1], [2], [3], [4], [5]],
        )
        assert result.n_observations == 5
