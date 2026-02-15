"""
Tests for bootstrap confidence intervals.

Tests all 5 CI methods: normal, basic, percentile, BCa, studentized.
Validates formulas against known analytical results.
"""

import numpy as np
import pytest

from pystatistics.montecarlo import boot, boot_ci


def mean_stat(data, indices):
    """Bootstrap statistic: sample mean."""
    return np.array([np.mean(data[indices])])


def mean_var_stat(data, indices):
    """Bootstrap statistic returning mean and its variance estimate."""
    d = data[indices]
    m = np.mean(d)
    v = np.var(d, ddof=1) / len(d)  # variance of the mean
    return np.array([m, v])


# ---------------------------------------------------------------------------
# Tests: Individual CI types
# ---------------------------------------------------------------------------

class TestPercentileCI:
    """Tests for percentile bootstrap CI."""

    def test_basic_percentile(self):
        """Percentile CI gives reasonable bounds."""
        data = np.arange(1.0, 101.0)
        result = boot(data, mean_stat, R=2000, seed=42)
        ci_result = boot_ci(result, type="perc")

        assert "perc" in ci_result.ci
        ci = ci_result.ci["perc"]
        assert ci.shape == (1, 2)

        # CI should contain the true mean (50.5)
        assert ci[0, 0] < 50.5 < ci[0, 1]

        # Bounds should be ordered
        assert ci[0, 0] < ci[0, 1]

    def test_percentile_formula(self):
        """Percentile CI uses correct quantiles."""
        # Construct a bootstrap with known distribution
        data = np.arange(1.0, 11.0)
        result = boot(data, mean_stat, R=10000, seed=42)

        ci_result = boot_ci(result, type="perc", conf=0.90)
        ci = ci_result.ci["perc"]

        # The 5th and 95th percentiles of bootstrap replicates
        expected_lo = np.quantile(result.t[:, 0], 0.05)
        expected_hi = np.quantile(result.t[:, 0], 0.95)

        assert ci[0, 0] == pytest.approx(expected_lo, rel=1e-10)
        assert ci[0, 1] == pytest.approx(expected_hi, rel=1e-10)


class TestBasicCI:
    """Tests for basic (pivotal) bootstrap CI."""

    def test_basic_ci(self):
        """Basic CI gives reasonable bounds."""
        data = np.arange(1.0, 101.0)
        result = boot(data, mean_stat, R=2000, seed=42)
        ci_result = boot_ci(result, type="basic")

        assert "basic" in ci_result.ci
        ci = ci_result.ci["basic"]
        assert ci.shape == (1, 2)
        assert ci[0, 0] < 50.5 < ci[0, 1]

    def test_basic_formula(self):
        """Basic CI uses correct pivot formula."""
        data = np.arange(1.0, 11.0)
        result = boot(data, mean_stat, R=10000, seed=42)

        ci_result = boot_ci(result, type="basic", conf=0.90)
        ci = ci_result.ci["basic"]

        t0 = result.t0[0]
        q_lo = np.quantile(result.t[:, 0], 0.05)
        q_hi = np.quantile(result.t[:, 0], 0.95)

        # Basic: [2*t0 - Q(1-alpha/2), 2*t0 - Q(alpha/2)]
        expected_lo = 2.0 * t0 - q_hi
        expected_hi = 2.0 * t0 - q_lo

        assert ci[0, 0] == pytest.approx(expected_lo, rel=1e-10)
        assert ci[0, 1] == pytest.approx(expected_hi, rel=1e-10)


class TestNormalCI:
    """Tests for normal approximation CI."""

    def test_normal_ci(self):
        """Normal CI gives reasonable bounds."""
        data = np.arange(1.0, 101.0)
        result = boot(data, mean_stat, R=2000, seed=42)
        ci_result = boot_ci(result, type="normal")

        assert "normal" in ci_result.ci
        ci = ci_result.ci["normal"]
        assert ci.shape == (1, 2)
        assert ci[0, 0] < 50.5 < ci[0, 1]

    def test_normal_formula(self):
        """Normal CI uses bias-corrected center."""
        from scipy import stats as sp_stats

        data = np.arange(1.0, 11.0)
        result = boot(data, mean_stat, R=10000, seed=42)

        ci_result = boot_ci(result, type="normal", conf=0.90)
        ci = ci_result.ci["normal"]

        t0 = result.t0[0]
        center = 2.0 * t0 - np.mean(result.t[:, 0])
        se = np.std(result.t[:, 0], ddof=1)
        z = sp_stats.norm.ppf(0.95)

        expected_lo = center - z * se
        expected_hi = center + z * se

        assert ci[0, 0] == pytest.approx(expected_lo, rel=1e-10)
        assert ci[0, 1] == pytest.approx(expected_hi, rel=1e-10)


class TestBCaCI:
    """Tests for bias-corrected and accelerated CI."""

    def test_bca_ci(self):
        """BCa CI gives reasonable bounds."""
        data = np.arange(1.0, 51.0)
        result = boot(data, mean_stat, R=2000, seed=42)
        ci_result = boot_ci(result, type="bca")

        assert "bca" in ci_result.ci
        ci = ci_result.ci["bca"]
        assert ci.shape == (1, 2)

        # Should contain the true mean (25.5)
        assert ci[0, 0] < 25.5 < ci[0, 1]

    def test_bca_reduces_to_percentile_when_symmetric(self):
        """BCa ≈ percentile for symmetric distributions."""
        # For a symmetric distribution, z0 ≈ 0 and a ≈ 0
        # so BCa should be close to percentile
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)

        result = boot(data, mean_stat, R=5000, seed=42)
        ci_perc = boot_ci(result, type="perc").ci["perc"]
        ci_bca = boot_ci(result, type="bca").ci["bca"]

        # They should be close for symmetric data
        assert ci_bca[0, 0] == pytest.approx(ci_perc[0, 0], abs=0.3)
        assert ci_bca[0, 1] == pytest.approx(ci_perc[0, 1], abs=0.3)


class TestStudentizedCI:
    """Tests for studentized (bootstrap-t) CI."""

    def test_studentized_ci(self):
        """Studentized CI with variance estimates."""
        data = np.arange(1.0, 51.0)

        # Use mean_var_stat to get per-replicate variance
        result = boot(data, mean_var_stat, R=2000, seed=42)

        # Var estimates are in t[:, 1]
        var_t = result.t[:, 1]
        var_t0 = float(result.t0[1])

        ci_result = boot_ci(
            result, type="stud", index=0,
            var_t0=var_t0, var_t=var_t,
        )

        assert "stud" in ci_result.ci
        ci = ci_result.ci["stud"]
        assert ci.shape == (2, 2)  # k=2 statistics

        # CI for index=0 (mean) should contain true mean (25.5)
        assert ci[0, 0] < 25.5 < ci[0, 1]

    def test_studentized_requires_var_t(self):
        """Studentized CI raises without var_t."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = boot(data, mean_stat, R=100, seed=42)

        with pytest.raises(ValueError, match="var_t"):
            boot_ci(result, type="stud")


# ---------------------------------------------------------------------------
# Tests: "all" type
# ---------------------------------------------------------------------------

class TestAllCI:
    """Tests for type='all'."""

    def test_all_without_var_t(self):
        """type='all' computes normal, basic, perc, bca (not stud)."""
        data = np.arange(1.0, 21.0)
        result = boot(data, mean_stat, R=500, seed=42)
        ci_result = boot_ci(result, type="all")

        assert "normal" in ci_result.ci
        assert "basic" in ci_result.ci
        assert "perc" in ci_result.ci
        assert "bca" in ci_result.ci
        assert "stud" not in ci_result.ci

    def test_all_with_var_t(self):
        """type='all' includes stud when var_t provided."""
        data = np.arange(1.0, 21.0)
        result = boot(data, mean_var_stat, R=500, seed=42)

        var_t = result.t[:, 1]
        ci_result = boot_ci(result, type="all", var_t=var_t)

        assert "stud" in ci_result.ci


# ---------------------------------------------------------------------------
# Tests: Confidence levels
# ---------------------------------------------------------------------------

class TestConfLevels:
    """Tests for different confidence levels."""

    def test_wider_at_higher_conf(self):
        """Higher confidence level gives wider CI."""
        data = np.arange(1.0, 51.0)
        result = boot(data, mean_stat, R=2000, seed=42)

        ci_90 = boot_ci(result, type="perc", conf=0.90).ci["perc"]
        ci_95 = boot_ci(result, type="perc", conf=0.95).ci["perc"]
        ci_99 = boot_ci(result, type="perc", conf=0.99).ci["perc"]

        width_90 = ci_90[0, 1] - ci_90[0, 0]
        width_95 = ci_95[0, 1] - ci_95[0, 0]
        width_99 = ci_99[0, 1] - ci_99[0, 0]

        assert width_90 < width_95 < width_99

    def test_conf_level_stored(self):
        """Confidence level is stored in result."""
        data = np.arange(1.0, 11.0)
        result = boot(data, mean_stat, R=100, seed=42)
        ci_result = boot_ci(result, type="perc", conf=0.90)
        assert ci_result.ci_conf_level == 0.90


# ---------------------------------------------------------------------------
# Tests: CI summary display
# ---------------------------------------------------------------------------

class TestCISummary:
    """Tests that CI is shown in summary."""

    def test_summary_shows_ci(self):
        """summary() includes CI information when available."""
        data = np.arange(1.0, 11.0)
        result = boot(data, mean_stat, R=100, seed=42)
        ci_result = boot_ci(result, type="perc")
        s = ci_result.summary()

        assert "perc" in s
        assert "CI" in s
