"""
Tests for var_test() matching R var.test().

All R reference values verified against R 4.5.2.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.hypothesis import var_test


class TestVarTestBasic:
    """Basic F-test for comparing two variances."""

    def test_equal_variances(self):
        """
        R: var.test(c(1,2,3,4,5), c(6,7,8,9,10))
        F = 1, num df = 4, denom df = 4, p-value = 1
        CI: [0.1041228, 9.604534]
        ratio of variances: 1
        """
        x = [1, 2, 3, 4, 5]
        y = [6, 7, 8, 9, 10]
        result = var_test(x, y)
        assert result.statistic == pytest.approx(1.0, rel=1e-10)
        assert result.parameter == {"num df": 4.0, "denom df": 4.0}
        assert result.p_value == pytest.approx(1.0, rel=1e-10)
        assert result.estimate == {
            "ratio of variances": pytest.approx(1.0),
        }
        assert result.null_value == {"ratio of variances": 1.0}
        assert_allclose(
            result.conf_int,
            [0.10412280695590746, 9.6041564705451866],
            rtol=1e-4,
        )
        assert result.method == "F test to compare two variances"

    def test_alternative_less(self):
        """
        R: var.test(c(1,2,3,4,5), c(6,7,8,9,10), alternative="less")
        F = 1, p-value = 0.5
        CI: [0, 6.388233]
        """
        x = [1, 2, 3, 4, 5]
        y = [6, 7, 8, 9, 10]
        result = var_test(x, y, alternative="less")
        assert result.statistic == pytest.approx(1.0, rel=1e-10)
        assert result.p_value == pytest.approx(0.5, rel=1e-10)
        assert result.conf_int[0] == pytest.approx(0.0, abs=1e-15)
        assert result.conf_int[1] == pytest.approx(6.3882296388279746, rel=1e-4)

    def test_alternative_greater(self):
        """
        R: var.test(c(1,2,3,4,5), c(6,7,8,9,10), alternative="greater")
        F = 1, p-value = 0.5
        CI: [0.1565375, Inf]
        """
        x = [1, 2, 3, 4, 5]
        y = [6, 7, 8, 9, 10]
        result = var_test(x, y, alternative="greater")
        assert result.statistic == pytest.approx(1.0, rel=1e-10)
        assert result.p_value == pytest.approx(0.5, rel=1e-10)
        assert result.conf_int[0] == pytest.approx(0.15653752299604893, rel=1e-4)
        assert result.conf_int[1] == float('inf')

    def test_ratio_hypothesis(self):
        """
        R: var.test(c(1,2,3,4,5), c(6,7,8,9,10), ratio=2)
        F = 0.5, p-value = 0.5185185
        """
        x = [1, 2, 3, 4, 5]
        y = [6, 7, 8, 9, 10]
        result = var_test(x, y, ratio=2)
        assert result.statistic == pytest.approx(0.5, rel=1e-10)
        assert result.p_value == pytest.approx(0.51851851851851849, rel=1e-10)
        assert result.null_value == {"ratio of variances": 2.0}


class TestVarTestUnequal:
    """Tests with clearly different variances."""

    def test_very_different(self):
        """
        R: var.test(c(1,2,3,4,5), c(1,10,100,1000,10000))
        F = 0.01, p-value ≈ 0.0005843
        """
        x = [1, 2, 3, 4, 5]
        y = [1, 10, 100, 1000, 10000]
        result = var_test(x, y)

        # Compute expected ratio
        var_x = np.var(x, ddof=1)
        var_y = np.var(y, ddof=1)
        expected_f = var_x / var_y
        assert result.statistic == pytest.approx(expected_f, rel=1e-10)
        assert result.p_value < 0.001

    def test_unequal_ci(self):
        """
        R: var.test(c(1,2,3,4,5), c(1,10,100,1000,10000))
        CI: [0.0010412, 0.096045]
        """
        x = [1, 2, 3, 4, 5]
        y = [1, 10, 100, 1000, 10000]
        result = var_test(x, y)
        # CI should not contain 1 (significant at 95%)
        assert result.conf_int[1] < 1.0

    def test_larger_samples(self):
        """Larger samples with different variances."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0, 3, 50)
        result = var_test(x, y)
        # Should detect variance difference
        assert result.p_value < 0.05

    def test_custom_conf_level(self):
        """
        R: var.test(c(1,2,3,4,5), c(6,7,8,9,10), conf.level=0.99)
        """
        x = [1, 2, 3, 4, 5]
        y = [6, 7, 8, 9, 10]
        result = var_test(x, y, conf_level=0.99)
        # 99% CI should be wider than 95%
        result_95 = var_test(x, y, conf_level=0.95)
        assert result.conf_int[0] < result_95.conf_int[0]
        assert result.conf_int[1] > result_95.conf_int[1]


class TestVarTestEdgeCases:
    """Edge cases and validation."""

    def test_invalid_ratio(self):
        """Negative ratio should raise."""
        with pytest.raises(Exception):
            var_test([1, 2, 3], [4, 5, 6], ratio=-1)

    def test_zero_ratio(self):
        """Zero ratio should raise."""
        with pytest.raises(Exception):
            var_test([1, 2, 3], [4, 5, 6], ratio=0)

    def test_too_few_observations(self):
        """Need at least 2 observations in each sample."""
        with pytest.raises(Exception):
            var_test([1], [2, 3, 4])

    def test_y_required(self):
        """y is required for var_test."""
        with pytest.raises(Exception):
            var_test([1, 2, 3])

    def test_summary(self):
        """summary() produces readable output."""
        result = var_test([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        s = result.summary()
        assert "F test" in s
        assert "p-value" in s

    def test_repr(self):
        """repr is informative."""
        result = var_test([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        r = repr(result)
        assert "HTestSolution" in r
        assert "p_value" in r

    def test_backend_name(self):
        """Backend is cpu_hypothesis."""
        result = var_test([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        assert result.backend_name == "cpu_hypothesis"

    def test_design_passthrough(self):
        """Can pass a pre-built HypothesisDesign."""
        from pystatistics.hypothesis.design import HypothesisDesign
        design = HypothesisDesign.for_var_test(
            [1, 2, 3, 4, 5], [6, 7, 8, 9, 10],
        )
        result = var_test(design)
        assert result.statistic == pytest.approx(1.0, rel=1e-10)

    def test_statistic_name(self):
        """Statistic should be named F."""
        result = var_test([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        assert result.statistic_name == "F"
