"""
Tests for permutation tests.

Tests two-sample permutation tests with various statistics
and alternative hypotheses. Validates p-value computation.
"""

import numpy as np
import pytest

from pystatistics.montecarlo import permutation_test


def mean_diff(x, y):
    """Test statistic: difference in means."""
    return np.mean(x) - np.mean(y)


def median_diff(x, y):
    """Test statistic: difference in medians."""
    return np.median(x) - np.median(y)


# ---------------------------------------------------------------------------
# Tests: Basic permutation test
# ---------------------------------------------------------------------------

class TestPermutationTest:
    """Tests for permutation_test()."""

    def test_significant_difference(self):
        """Detects significant difference between groups."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        result = permutation_test(x, y, mean_diff, R=9999, seed=42)

        # Should be highly significant (p < 0.05)
        assert result.p_value < 0.05
        assert result.observed_stat == pytest.approx(-5.0, rel=1e-10)
        assert result.R == 9999

    def test_no_difference(self):
        """No significant difference for identical distributions."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 20)
        y = rng.normal(0, 1, 20)

        result = permutation_test(x, y, mean_diff, R=999, seed=42)

        # p-value should be large (not significant)
        assert result.p_value > 0.05

    def test_two_sided(self):
        """Two-sided test uses absolute values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        result = permutation_test(
            x, y, mean_diff, R=999,
            alternative="two.sided", seed=42,
        )
        assert result.alternative == "two.sided"
        assert result.p_value < 0.05

    def test_less(self):
        """One-sided test: less."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        result = permutation_test(
            x, y, mean_diff, R=999,
            alternative="less", seed=42,
        )
        assert result.alternative == "less"
        # mean(x) < mean(y), so mean_diff < 0, and "less" should be significant
        assert result.p_value < 0.05

    def test_greater(self):
        """One-sided test: greater."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        result = permutation_test(
            x, y, mean_diff, R=999,
            alternative="greater", seed=42,
        )
        assert result.alternative == "greater"
        # mean(x) < mean(y), so mean_diff < 0, "greater" should NOT be significant
        assert result.p_value > 0.5

    def test_seed_reproducibility(self):
        """Same seed gives same results."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])

        r1 = permutation_test(x, y, mean_diff, R=999, seed=42)
        r2 = permutation_test(x, y, mean_diff, R=999, seed=42)

        assert r1.p_value == r2.p_value
        np.testing.assert_array_equal(r1.perm_stats, r2.perm_stats)

    def test_different_seeds_differ(self):
        """Different seeds give different results."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])

        r1 = permutation_test(x, y, mean_diff, R=999, seed=42)
        r2 = permutation_test(x, y, mean_diff, R=999, seed=99)

        assert not np.allclose(r1.perm_stats, r2.perm_stats)

    def test_custom_statistic(self):
        """Works with custom statistic function (median diff)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        result = permutation_test(x, y, median_diff, R=9999, seed=42)
        assert result.observed_stat == pytest.approx(-5.0, rel=1e-10)
        assert result.p_value < 0.10  # stochastic, use wider threshold


# ---------------------------------------------------------------------------
# Tests: P-value properties
# ---------------------------------------------------------------------------

class TestPValue:
    """Tests for p-value computation properties."""

    def test_p_value_range(self):
        """P-value is in (0, 1]."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])

        result = permutation_test(x, y, mean_diff, R=999, seed=42)
        assert 0 < result.p_value <= 1.0

    def test_p_value_phipson_smyth(self):
        """P-value uses (count + 1) / (R + 1) correction."""
        # With very different groups, all permutations should have
        # |stat| < |observed|, giving p = 1/(R+1) minimum
        x = np.array([1.0])
        y = np.array([1000.0])

        result = permutation_test(x, y, mean_diff, R=99, seed=42)

        # With n1=1, n2=1, there are only 2 possible permutations
        # p should be 1/(99+1) = 0.01 minimum, or we might get 2/100 etc.
        assert result.p_value >= 1.0 / 100.0

    def test_perm_stats_shape(self):
        """Permutation statistics have shape (R,)."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])

        result = permutation_test(x, y, mean_diff, R=500, seed=42)
        assert result.perm_stats.shape == (500,)


# ---------------------------------------------------------------------------
# Tests: Solution display
# ---------------------------------------------------------------------------

class TestPermutationSolution:
    """Tests for PermutationSolution display."""

    def test_summary(self):
        """summary() produces readable output."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])

        result = permutation_test(x, y, mean_diff, R=99, seed=42)
        s = result.summary()

        assert "PERMUTATION TEST" in s
        assert "p-value" in s

    def test_repr(self):
        """__repr__ is informative."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])

        result = permutation_test(x, y, mean_diff, R=99, seed=42)
        r = repr(result)

        assert "PermutationSolution" in r
        assert "R=99" in r

    def test_backend_name(self):
        """Backend name is correct."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])

        result = permutation_test(x, y, mean_diff, R=10, seed=42)
        assert "cpu" in result.backend_name


# ---------------------------------------------------------------------------
# Tests: Validation
# ---------------------------------------------------------------------------

class TestPermutationValidation:
    """Tests for input validation."""

    def test_invalid_alternative(self):
        """Invalid alternative raises ValueError."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        with pytest.raises(ValueError, match="alternative"):
            permutation_test(x, y, mean_diff, R=10, alternative="invalid")

    def test_R_must_be_positive(self):
        """R must be >= 1."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        with pytest.raises(ValueError, match="R must be >= 1"):
            permutation_test(x, y, mean_diff, R=0)

    def test_empty_x(self):
        """Empty x raises ValueError."""
        with pytest.raises(ValueError):
            permutation_test(np.array([]), np.array([1.0]), mean_diff, R=10)

    def test_empty_y(self):
        """Empty y raises ValueError."""
        with pytest.raises(ValueError):
            permutation_test(np.array([1.0]), np.array([]), mean_diff, R=10)
