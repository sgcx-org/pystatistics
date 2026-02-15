"""
Tests for prop_test() matching R prop.test().

All R reference values verified against R 4.5.2.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.hypothesis import prop_test


class TestPropTestOneSample:
    """One-sample proportion test."""

    def test_basic(self):
        """
        R: prop.test(30, 100, p=0.5)
        X-squared = 15.21, df = 1, p-value = 9.619e-05
        """
        result = prop_test(30, 100, p=0.5)
        assert result.statistic == pytest.approx(15.21, rel=1e-10)
        assert result.parameter == {"df": 1.0}
        assert result.p_value == pytest.approx(9.6192688035205337e-05, rel=1e-10)
        assert result.estimate == {"p": pytest.approx(0.3)}
        assert result.null_value == {"p": 0.5}
        assert "continuity correction" in result.method

    def test_ci(self):
        """CI matches R Wilson score interval."""
        result = prop_test(30, 100, p=0.5)
        assert_allclose(
            result.conf_int,
            [0.21454256992906129, 0.40106042434519262],
            rtol=1e-5,
        )

    def test_no_correction(self):
        """
        R: prop.test(30, 100, p=0.5, correct=FALSE)
        X-squared = 16, p-value = 6.334e-05
        """
        result = prop_test(30, 100, p=0.5, correct=False)
        assert result.statistic == pytest.approx(16.0, rel=1e-10)
        assert result.p_value == pytest.approx(6.3342483666239835e-05, rel=1e-10)

    def test_alternative_less(self):
        """
        R: prop.test(30, 100, p=0.5, alternative="less")
        """
        result = prop_test(30, 100, p=0.5, alternative="less")
        assert result.statistic == pytest.approx(15.21, rel=1e-10)
        assert result.p_value == pytest.approx(4.8096344017602736e-05, rel=1e-10)
        assert result.conf_int[0] == pytest.approx(0.0, abs=1e-15)
        assert result.conf_int[1] == pytest.approx(0.38503933562525128, rel=1e-5)

    def test_alternative_greater(self):
        """
        R: prop.test(30, 100, p=0.5, alternative="greater")
        """
        result = prop_test(30, 100, p=0.5, alternative="greater")
        assert result.statistic == pytest.approx(15.21, rel=1e-10)
        assert result.p_value == pytest.approx(0.99995190365598241, rel=1e-10)
        assert result.conf_int[0] == pytest.approx(0.22618577036919929, rel=1e-5)
        assert result.conf_int[1] == pytest.approx(1.0, abs=1e-15)

    def test_zero_successes(self):
        """
        R: prop.test(0, 100, p=0.5)
        """
        result = prop_test(0, 100, p=0.5)
        assert result.statistic == pytest.approx(98.01, rel=1e-10)
        assert result.p_value == pytest.approx(4.1627504389864181e-23, rel=1e-5)
        assert result.conf_int[0] == pytest.approx(0.0, abs=1e-15)
        assert result.conf_int[1] == pytest.approx(0.046101340612186362, rel=1e-4)

    def test_custom_conf_level(self):
        """
        R: prop.test(30, 100, p=0.5, conf.level=0.99)
        """
        result = prop_test(30, 100, p=0.5, conf_level=0.99)
        assert result.statistic == pytest.approx(15.21, rel=1e-10)
        assert result.p_value == pytest.approx(9.6192688035205337e-05, rel=1e-10)
        assert_allclose(
            result.conf_int,
            [0.19328671386533222, 0.43261654286295581],
            rtol=1e-4,
        )

    def test_data_name(self):
        """data_name set correctly for one-sample."""
        result = prop_test(30, 100, p=0.5)
        assert "out of" in result.data_name


class TestPropTestTwoSample:
    """Two-sample proportion test (equality of proportions)."""

    def test_basic(self):
        """
        R: prop.test(c(30, 50), c(100, 100))
        X-squared = 7.5208, df = 1, p-value = 0.0061
        """
        result = prop_test([30, 50], [100, 100])
        assert result.statistic == pytest.approx(7.520833333333333, rel=1e-10)
        assert result.parameter == {"df": 1.0}
        assert result.p_value == pytest.approx(0.0060989459312143666, rel=1e-10)
        assert result.estimate == {
            "prop 1": pytest.approx(0.3),
            "prop 2": pytest.approx(0.5),
        }
        assert "equality" in result.method

    def test_ci(self):
        """CI for two-sample difference."""
        result = prop_test([30, 50], [100, 100])
        assert_allclose(
            result.conf_int,
            [-0.34293122498191675, -0.057068775018083273],
            rtol=1e-4,
        )

    def test_no_correction(self):
        """
        R: prop.test(c(30, 50), c(100, 100), correct=FALSE)
        """
        result = prop_test([30, 50], [100, 100], correct=False)
        assert result.statistic == pytest.approx(8.3333333333333339, rel=1e-10)
        assert result.p_value == pytest.approx(0.0038924171227786306, rel=1e-10)
        assert_allclose(
            result.conf_int,
            [-0.33293122498191674, -0.067068775018083282],
            rtol=1e-4,
        )


class TestPropTestKSamples:
    """K-sample proportion test (k > 2)."""

    def test_three_groups(self):
        """
        R: prop.test(c(30, 50, 40), c(100, 100, 100))
        X-squared = 8.3333, df = 2, p-value = 0.01550
        """
        result = prop_test([30, 50, 40], [100, 100, 100])
        assert result.statistic == pytest.approx(8.3333333333333339, rel=1e-10)
        assert result.parameter == {"df": 2.0}
        assert result.p_value == pytest.approx(0.015503853599009314, rel=1e-10)
        assert result.estimate == {
            "prop 1": pytest.approx(0.3),
            "prop 2": pytest.approx(0.5),
            "prop 3": pytest.approx(0.4),
        }

    def test_no_ci_for_k_groups(self):
        """No CI returned for k > 2 groups."""
        result = prop_test([30, 50, 40], [100, 100, 100])
        assert result.conf_int is None


class TestPropTestEdgeCases:
    """Edge cases and validation."""

    def test_invalid_alternative_k(self):
        """One-sided alternative not allowed for k > 1."""
        with pytest.raises(Exception):
            prop_test([30, 50], [100, 100], alternative="less")

    def test_negative_successes(self):
        """Negative successes should raise."""
        with pytest.raises(Exception):
            prop_test(-1, 100, p=0.5)

    def test_successes_exceed_trials(self):
        """x > n should raise."""
        with pytest.raises(Exception):
            prop_test(101, 100, p=0.5)

    def test_invalid_null_p(self):
        """Null p outside (0,1) should raise."""
        with pytest.raises(Exception):
            prop_test(30, 100, p=0.0)

    def test_n_required(self):
        """n must be provided."""
        with pytest.raises(Exception):
            prop_test(30)

    def test_summary(self):
        """summary() produces readable output."""
        result = prop_test(30, 100, p=0.5)
        s = result.summary()
        assert "proportions" in s
        assert "p-value" in s

    def test_repr(self):
        """repr is informative."""
        result = prop_test(30, 100, p=0.5)
        r = repr(result)
        assert "HTestSolution" in r
        assert "p_value" in r

    def test_backend_name(self):
        """Backend is cpu_hypothesis."""
        result = prop_test(30, 100, p=0.5)
        assert result.backend_name == "cpu_hypothesis"

    def test_design_passthrough(self):
        """Can pass a pre-built HypothesisDesign."""
        from pystatistics.hypothesis.design import HypothesisDesign
        design = HypothesisDesign.for_prop_test(
            [30], [100], p=[0.5],
        )
        result = prop_test(design)
        assert result.statistic == pytest.approx(15.21, rel=1e-10)
