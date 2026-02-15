"""
Tests for bootstrap resampling.

Tests ordinary, balanced, and parametric bootstrap with various statistics.
Verifies seed reproducibility, bias/SE properties, and basic sanity.
"""

import numpy as np
import pytest

from pystatistics.montecarlo import boot


# ---------------------------------------------------------------------------
# Statistic functions
# ---------------------------------------------------------------------------

def mean_stat(data, indices):
    """Bootstrap statistic: sample mean."""
    return np.array([np.mean(data[indices])])


def mean_var_stat(data, indices):
    """Bootstrap statistic: mean and variance (2 statistics)."""
    d = data[indices]
    return np.array([np.mean(d), np.var(d, ddof=1)])


def regression_coef_stat(data, indices):
    """Bootstrap statistic: simple regression slope and intercept."""
    d = data[indices]
    x, y = d[:, 0], d[:, 1]
    n = len(x)
    x_bar, y_bar = np.mean(x), np.mean(y)
    slope = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar) ** 2)
    intercept = y_bar - slope * x_bar
    return np.array([intercept, slope])


# ---------------------------------------------------------------------------
# Tests: Ordinary bootstrap
# ---------------------------------------------------------------------------

class TestOrdinaryBootstrap:
    """Tests for sim='ordinary' bootstrap."""

    def test_basic_mean(self):
        """Bootstrap of the mean works correctly."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = boot(data, mean_stat, R=999, seed=42)

        # t0 should be the actual mean
        assert result.t0[0] == pytest.approx(5.5, rel=1e-10)

        # Shape checks
        assert result.t.shape == (999, 1)
        assert result.R == 999
        assert len(result.bias) == 1
        assert len(result.se) == 1

        # Bias should be small (bootstrap is approximately unbiased for mean)
        assert abs(result.bias[0]) < 0.5

        # SE should be reasonable (true SE of mean = sd/sqrt(n) ≈ 0.96)
        assert 0.5 < result.se[0] < 1.5

    def test_multi_statistic(self):
        """Bootstrap with multiple statistics (mean + var)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = boot(data, mean_var_stat, R=500, seed=42)

        assert result.t0.shape == (2,)
        assert result.t.shape == (500, 2)
        assert result.t0[0] == pytest.approx(3.0, rel=1e-10)
        assert result.t0[1] == pytest.approx(2.5, rel=1e-10)

    def test_regression_bootstrap(self):
        """Bootstrap of regression coefficients."""
        rng = np.random.default_rng(123)
        n = 50
        x = rng.uniform(0, 10, n)
        y = 2.0 + 3.0 * x + rng.normal(0, 1, n)
        data = np.column_stack([x, y])

        result = boot(data, regression_coef_stat, R=500, seed=42)

        # t0 should be close to true values (intercept≈2, slope≈3)
        assert result.t0[0] == pytest.approx(2.0, abs=1.0)
        assert result.t0[1] == pytest.approx(3.0, abs=0.3)

        # Shape
        assert result.t.shape == (500, 2)

    def test_seed_reproducibility(self):
        """Same seed gives same results."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r1 = boot(data, mean_stat, R=100, seed=42)
        r2 = boot(data, mean_stat, R=100, seed=42)

        np.testing.assert_array_equal(r1.t, r2.t)
        np.testing.assert_array_equal(r1.t0, r2.t0)
        np.testing.assert_array_equal(r1.bias, r2.bias)
        np.testing.assert_array_equal(r1.se, r2.se)

    def test_different_seeds_differ(self):
        """Different seeds give different results."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r1 = boot(data, mean_stat, R=100, seed=42)
        r2 = boot(data, mean_stat, R=100, seed=99)

        assert not np.allclose(r1.t, r2.t)

    def test_large_R(self):
        """Large R produces stable estimates."""
        data = np.arange(1.0, 101.0)
        result = boot(data, mean_stat, R=5000, seed=42)

        # True mean = 50.5, true SE ≈ 28.87/sqrt(100) ≈ 2.887
        assert result.t0[0] == pytest.approx(50.5, rel=1e-10)
        assert result.se[0] == pytest.approx(2.887, rel=0.1)
        assert abs(result.bias[0]) < 0.5

    def test_single_observation(self):
        """Bootstrap with n=1 (degenerate case)."""
        data = np.array([5.0])
        result = boot(data, mean_stat, R=100, seed=42)

        # All replicates should be 5.0
        assert result.t0[0] == pytest.approx(5.0)
        np.testing.assert_allclose(result.t[:, 0], 5.0)


# ---------------------------------------------------------------------------
# Tests: Balanced bootstrap
# ---------------------------------------------------------------------------

class TestBalancedBootstrap:
    """Tests for sim='balanced' bootstrap."""

    def test_balanced_basic(self):
        """Balanced bootstrap produces valid results."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = boot(data, mean_stat, R=100, sim="balanced", seed=42)

        assert result.t0[0] == pytest.approx(3.0, rel=1e-10)
        assert result.t.shape == (100, 1)
        assert result.sim == "balanced"

    def test_balanced_coverage(self):
        """In balanced bootstrap, each obs appears exactly R times total."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        R = 100
        n = len(data)

        # We can verify this indirectly by checking the frequency-based
        # statistic. Use a stat that reveals the frequencies.
        counts = np.zeros(n, dtype=int)
        call_count = 0

        def counting_stat(d, indices):
            nonlocal counts, call_count
            call_count += 1
            # Only count during bootstrap replicates, not t0 computation
            if call_count > 1:
                for idx in indices:
                    counts[idx] += 1
            return np.array([np.mean(d[indices])])

        result = boot(data, counting_stat, R=R, sim="balanced", seed=42)

        # Each observation should appear exactly R times total
        np.testing.assert_array_equal(counts, np.full(n, R))

    def test_balanced_vs_ordinary_similar_se(self):
        """Balanced and ordinary should give similar SE estimates."""
        data = np.arange(1.0, 21.0)
        r_ord = boot(data, mean_stat, R=1000, sim="ordinary", seed=42)
        r_bal = boot(data, mean_stat, R=1000, sim="balanced", seed=42)

        # SEs should be in the same ballpark
        assert r_ord.se[0] == pytest.approx(r_bal.se[0], rel=0.3)


# ---------------------------------------------------------------------------
# Tests: Stratified bootstrap
# ---------------------------------------------------------------------------

class TestStratifiedBootstrap:
    """Tests for stratified bootstrap."""

    def test_stratified_basic(self):
        """Stratified bootstrap resamples within strata."""
        # Group A: [1,2,3], Group B: [10,20,30]
        data = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        strata = np.array([0, 0, 0, 1, 1, 1])

        result = boot(data, mean_stat, R=100, strata=strata, seed=42)

        # Each replicate should mix values from both strata
        # Mean should be between group means
        for b in range(result.R):
            # The mean should be roughly between 2 and 20
            assert 0.5 < result.t[b, 0] < 35.0

    def test_stratified_preserves_strata_sizes(self):
        """Stratified bootstrap preserves the size of each stratum."""
        data = np.array([1.0, 2.0, 3.0, 100.0, 200.0])
        strata = np.array([0, 0, 0, 1, 1])

        # Use a stat that checks strata composition
        def strata_check_stat(d, indices):
            d_sample = d[indices]
            # Count how many are < 50 (stratum 0) and >= 50 (stratum 1)
            n_s0 = np.sum(d_sample < 50)
            n_s1 = np.sum(d_sample >= 50)
            return np.array([float(n_s0), float(n_s1)])

        result = boot(data, strata_check_stat, R=50, strata=strata, seed=42)

        # Each replicate should have exactly 3 from s0 and 2 from s1
        np.testing.assert_array_equal(result.t[:, 0], 3.0)
        np.testing.assert_array_equal(result.t[:, 1], 2.0)


# ---------------------------------------------------------------------------
# Tests: stype variations
# ---------------------------------------------------------------------------

class TestStype:
    """Tests for different stype options."""

    def test_stype_f(self):
        """stype='f' passes frequency counts."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def freq_mean_stat(d, freqs):
            """Compute weighted mean from frequencies."""
            total = np.sum(freqs)
            if total == 0:
                return np.array([0.0])
            return np.array([np.sum(d * freqs) / total])

        result = boot(data, freq_mean_stat, R=100, stype="f", seed=42)

        # t0 with uniform frequencies should give the regular mean
        assert result.t0[0] == pytest.approx(3.0, rel=1e-10)

    def test_stype_w(self):
        """stype='w' passes normalized weights."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def weight_mean_stat(d, weights):
            """Compute weighted mean."""
            return np.array([np.sum(d * weights)])

        result = boot(data, weight_mean_stat, R=100, stype="w", seed=42)

        # t0 with uniform weights should give the regular mean
        assert result.t0[0] == pytest.approx(3.0, rel=1e-10)


# ---------------------------------------------------------------------------
# Tests: Solution object
# ---------------------------------------------------------------------------

class TestBootstrapSolution:
    """Tests for BootstrapSolution properties and display."""

    def test_summary(self):
        """summary() produces readable output."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = boot(data, mean_stat, R=100, seed=42)
        s = result.summary()

        assert "ORDINARY NONPARAMETRIC BOOTSTRAP" in s
        assert "t1*" in s
        assert "original" in s
        assert "bias" in s
        assert "std. error" in s

    def test_repr(self):
        """__repr__ produces useful string."""
        data = np.array([1.0, 2.0, 3.0])
        result = boot(data, mean_stat, R=100, seed=42)
        r = repr(result)

        assert "BootstrapSolution" in r
        assert "R=100" in r
        assert "k=1" in r

    def test_backend_name(self):
        """Backend name is set correctly."""
        data = np.array([1.0, 2.0, 3.0])
        result = boot(data, mean_stat, R=10, seed=42)
        assert "cpu" in result.backend_name

    def test_timing(self):
        """Timing info is available."""
        data = np.array([1.0, 2.0, 3.0])
        result = boot(data, mean_stat, R=10, seed=42)
        assert result.timing is not None
        assert 'total_seconds' in result.timing


# ---------------------------------------------------------------------------
# Tests: Edge cases and validation
# ---------------------------------------------------------------------------

class TestBootstrapValidation:
    """Tests for input validation."""

    def test_R_must_be_positive(self):
        """R must be >= 1."""
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="R must be >= 1"):
            boot(data, mean_stat, R=0)

    def test_invalid_sim(self):
        """Invalid sim raises ValueError."""
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="sim must be"):
            boot(data, mean_stat, R=10, sim="invalid")

    def test_invalid_stype(self):
        """Invalid stype raises ValueError."""
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="stype must be"):
            boot(data, mean_stat, R=10, stype="x")

    def test_empty_data(self):
        """Empty data raises ValueError."""
        with pytest.raises(ValueError):
            boot(np.array([]), mean_stat, R=10)

    def test_2d_data(self):
        """2D data works correctly."""
        data = np.column_stack([
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
        ])
        result = boot(data, regression_coef_stat, R=100, seed=42)
        assert result.t0.shape == (2,)
        assert result.t.shape == (100, 2)

    def test_strata_length_mismatch(self):
        """Strata length must match data rows."""
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="strata length"):
            boot(data, mean_stat, R=10, strata=np.array([0, 1]))

    def test_parametric_requires_ran_gen(self):
        """Parametric bootstrap requires ran_gen."""
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="ran_gen is required"):
            boot(data, mean_stat, R=10, sim="parametric")


# ---------------------------------------------------------------------------
# Tests: Parametric bootstrap
# ---------------------------------------------------------------------------

class TestParametricBootstrap:
    """Tests for sim='parametric' bootstrap."""

    def test_parametric_normal_mean(self):
        """Parametric bootstrap for normal mean."""
        rng_data = np.random.default_rng(123)
        data = rng_data.normal(5.0, 2.0, 30)

        # MLE estimates
        mle_params = {"mean": np.mean(data), "std": np.std(data, ddof=0)}

        def ran_gen(d, mle, rng):
            return rng.normal(mle["mean"], mle["std"], len(d))

        def mean_param_stat(d):
            return np.array([np.mean(d)])

        result = boot(
            data, mean_param_stat, R=500,
            sim="parametric", ran_gen=ran_gen, mle=mle_params,
            seed=42,
        )

        # t0 should be the sample mean
        assert result.t0[0] == pytest.approx(np.mean(data), rel=1e-10)

        # SE should be close to sigma/sqrt(n)
        expected_se = mle_params["std"] / np.sqrt(len(data))
        assert result.se[0] == pytest.approx(expected_se, rel=0.2)

        assert result.sim == "parametric"

    def test_parametric_seed_reproducibility(self):
        """Parametric bootstrap is reproducible with seed."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mle_params = {"mean": 3.0, "std": 1.58}

        def ran_gen(d, mle, rng):
            return rng.normal(mle["mean"], mle["std"], len(d))

        def mean_param_stat(d):
            return np.array([np.mean(d)])

        r1 = boot(
            data, mean_param_stat, R=100,
            sim="parametric", ran_gen=ran_gen, mle=mle_params,
            seed=42,
        )
        r2 = boot(
            data, mean_param_stat, R=100,
            sim="parametric", ran_gen=ran_gen, mle=mle_params,
            seed=42,
        )

        np.testing.assert_array_equal(r1.t, r2.t)

    def test_parametric_regression_residual_bootstrap(self):
        """Residual bootstrap for regression."""
        rng_data = np.random.default_rng(456)
        n = 50
        x = rng_data.uniform(0, 10, n)
        y = 1.5 + 2.5 * x + rng_data.normal(0, 1, n)
        data = np.column_stack([x, y])

        # Fit original model
        x_bar = np.mean(x)
        y_bar = np.mean(y)
        slope = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar) ** 2)
        intercept = y_bar - slope * x_bar
        fitted = intercept + slope * x
        residuals = y - fitted

        mle_params = {
            "intercept": intercept,
            "slope": slope,
            "residuals": residuals,
            "x": x,
        }

        def ran_gen(d, mle, rng):
            """Generate new y from fitted + resampled residuals."""
            resid_boot = rng.choice(mle["residuals"], size=len(mle["x"]),
                                     replace=True)
            y_new = mle["intercept"] + mle["slope"] * mle["x"] + resid_boot
            return np.column_stack([mle["x"], y_new])

        def reg_stat(d):
            """Compute regression coefficients."""
            x, y = d[:, 0], d[:, 1]
            x_bar, y_bar = np.mean(x), np.mean(y)
            sl = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar) ** 2)
            ic = y_bar - sl * x_bar
            return np.array([ic, sl])

        result = boot(
            data, reg_stat, R=500,
            sim="parametric", ran_gen=ran_gen, mle=mle_params,
            seed=42,
        )

        # Coefficients should be close to true values
        assert result.t0[0] == pytest.approx(intercept, rel=1e-10)
        assert result.t0[1] == pytest.approx(slope, rel=1e-10)

        # Bootstrap estimates should be centered near the originals
        assert np.mean(result.t[:, 0]) == pytest.approx(intercept, abs=0.5)
        assert np.mean(result.t[:, 1]) == pytest.approx(slope, abs=0.2)

    def test_parametric_summary(self):
        """Parametric bootstrap shows correct summary."""
        data = np.array([1.0, 2.0, 3.0])
        mle_params = {"mean": 2.0, "std": 1.0}

        def ran_gen(d, mle, rng):
            return rng.normal(mle["mean"], mle["std"], len(d))

        def mean_param_stat(d):
            return np.array([np.mean(d)])

        result = boot(
            data, mean_param_stat, R=50,
            sim="parametric", ran_gen=ran_gen, mle=mle_params,
            seed=42,
        )

        s = result.summary()
        assert "PARAMETRIC BOOTSTRAP" in s
