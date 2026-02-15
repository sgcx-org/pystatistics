"""
Tests for ks_test() matching R ks.test().

All R reference values verified against R 4.5.2.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.hypothesis import ks_test


class TestKSTwoSample:
    """Two-sample Kolmogorov-Smirnov test."""

    def test_basic(self):
        """
        R 4.5.2: ks.test(c(1,2,3,4,5), c(3,4,5,6,7))
        D = 0.4, p-value = 0.8730158730158730
        (Exact two-sample Kolmogorov-Smirnov test)
        """
        x = [1, 2, 3, 4, 5]
        y = [3, 4, 5, 6, 7]
        result = ks_test(x, y)
        assert result.statistic == pytest.approx(0.4, rel=1e-10)
        assert result.p_value == pytest.approx(0.87301587301587302, rel=1e-10)
        assert result.statistic_name == "D"
        assert "two-sample" in result.method

    def test_alternative_less(self):
        """
        R 4.5.2: ks.test(c(1,2,3,4,5), c(3,4,5,6,7), alternative="less")
        D^- ~ 0, p-value = 1
        """
        x = [1, 2, 3, 4, 5]
        y = [3, 4, 5, 6, 7]
        result = ks_test(x, y, alternative="less")
        assert result.p_value == pytest.approx(1.0, rel=1e-10)
        assert result.statistic_name == "D^-"

    def test_alternative_greater(self):
        """
        R 4.5.2: ks.test(c(1,2,3,4,5), c(3,4,5,6,7), alternative="greater")
        D^+ = 0.4, p-value = 0.4761904761904762
        """
        x = [1, 2, 3, 4, 5]
        y = [3, 4, 5, 6, 7]
        result = ks_test(x, y, alternative="greater")
        assert result.statistic == pytest.approx(0.4, rel=1e-10)
        assert result.p_value == pytest.approx(0.47619047619047616, rel=1e-10)
        assert result.statistic_name == "D^+"

    def test_identical_samples(self):
        """
        R 4.5.2: ks.test(c(1,2,3), c(1,2,3))
        D = 0, p-value = 1
        """
        x = [1, 2, 3]
        y = [1, 2, 3]
        result = ks_test(x, y)
        assert result.statistic == pytest.approx(0.0, abs=1e-15)
        assert result.p_value == pytest.approx(1.0, abs=1e-15)

    def test_with_ties(self):
        """
        R 4.5.2: ks.test(c(1,2,3,4,5), c(1,2,3,6,7))
        D = 0.4, p-value = 0.8730158730158730
        Same D and sample sizes as basic test, so same exact p-value.
        """
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 6, 7]
        result = ks_test(x, y)
        assert result.statistic == pytest.approx(0.4, rel=1e-10)
        assert result.p_value == pytest.approx(0.87301587301587302, rel=1e-10)

    def test_completely_separated(self):
        """
        R 4.5.2: ks.test(c(1,2,3), c(4,5,6))
        D = 1, p-value = 0.1
        Completely separated samples: exact test.
        """
        x = [1, 2, 3]
        y = [4, 5, 6]
        result = ks_test(x, y)
        assert result.statistic == pytest.approx(1.0, rel=1e-10)
        assert result.p_value == pytest.approx(0.1, rel=1e-4)

    def test_larger_samples(self):
        """Shifted normal distributions should be detected."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0.5, 1, 100)
        result = ks_test(x, y)
        assert result.p_value < 0.05  # Should detect the shift


class TestKSOneSample:
    """One-sample Kolmogorov-Smirnov test."""

    def test_normal(self):
        """
        R 4.5.2: ks.test(c(1,2,3,4,5), "pnorm", mean=3, sd=1.5)
        D = 0.14750746245307711, p-value = 0.99907073326002416
        """
        x = [1, 2, 3, 4, 5]
        result = ks_test(x, distribution="norm", mean=3, sd=1.5)
        assert result.statistic == pytest.approx(
            0.14750746245307711, rel=1e-10
        )
        assert result.p_value == pytest.approx(
            0.99907073326002416, rel=1e-10
        )
        assert "one-sample" in result.method

    def test_uniform(self):
        """
        R 4.5.2: ks.test(c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9), "punif")
        D = 0.1, p-value = 0.99987428406468037
        """
        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        result = ks_test(x, distribution="unif")
        assert result.statistic == pytest.approx(0.1, rel=1e-10)
        assert result.p_value == pytest.approx(0.99987428406468037, rel=1e-4)

    def test_default_standard_normal(self):
        """Default distribution is standard normal when neither y nor dist given."""
        x = [0.1, -0.2, 0.3, -0.1, 0.05]
        result = ks_test(x)
        assert result.statistic_name == "D"
        assert "one-sample" in result.method

    def test_exponential(self):
        """
        R 4.5.2: ks.test(c(0.5, 1.0, 1.5, 2.0, 3.0), "pexp", rate=1)
        """
        x = [0.5, 1.0, 1.5, 2.0, 3.0]
        result = ks_test(x, distribution="exp", rate=1)
        assert result.statistic > 0
        assert 0 <= result.p_value <= 1

    def test_pnorm_alias(self):
        """R's pnorm alias should work."""
        x = [1, 2, 3, 4, 5]
        result = ks_test(x, distribution="pnorm", mean=3, sd=1.5)
        assert result.statistic == pytest.approx(
            0.14750746245307711, rel=1e-10
        )

    def test_unknown_distribution_raises(self):
        """Unknown distribution name should raise."""
        with pytest.raises(ValueError, match="Unknown distribution"):
            ks_test([1, 2, 3], distribution="poisson")


class TestKSEdgeCases:
    """Edge cases and validation."""

    def test_summary(self):
        """summary() produces readable output."""
        result = ks_test([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
        s = result.summary()
        assert "Kolmogorov-Smirnov" in s
        assert "p-value" in s

    def test_repr(self):
        """repr is informative."""
        result = ks_test([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
        r = repr(result)
        assert "HTestSolution" in r
        assert "p_value" in r

    def test_backend_name(self):
        """Backend is cpu_hypothesis."""
        result = ks_test([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
        assert result.backend_name == "cpu_hypothesis"

    def test_design_passthrough(self):
        """Can pass a pre-built HypothesisDesign."""
        from pystatistics.hypothesis.design import HypothesisDesign
        design = HypothesisDesign.for_ks_test(
            [1, 2, 3, 4, 5], [3, 4, 5, 6, 7],
        )
        result = ks_test(design)
        assert result.statistic == pytest.approx(0.4, rel=1e-10)

    def test_data_name_two_sample(self):
        """data_name set correctly for two-sample."""
        result = ks_test([1, 2, 3], [4, 5, 6])
        assert result.data_name == "x and y"

    def test_data_name_one_sample(self):
        """data_name set correctly for one-sample."""
        result = ks_test([1, 2, 3], distribution="norm")
        assert result.data_name == "x"
