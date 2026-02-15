"""
Tests for t_test() matching R t.test().

All R reference values verified against R 4.5.2.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.hypothesis import t_test


class TestOneSampleTTest:
    """One-sample t-test: H0: mean(x) = mu."""

    def test_basic(self):
        """t.test(1:5, mu=3) -> t=0, df=4, p=1."""
        result = t_test([1, 2, 3, 4, 5], mu=3)
        assert result.statistic == pytest.approx(0.0, abs=1e-15)
        assert result.parameter == {"df": 4.0}
        assert result.p_value == pytest.approx(1.0, abs=1e-15)
        assert result.method == "One Sample t-test"
        assert result.alternative == "two.sided"
        assert result.estimate == {"mean of x": pytest.approx(3.0)}
        assert result.null_value == {"mean": 3.0}

    def test_basic_mu0(self):
        """t.test(1:5) -> t=4.2426, df=4, p=0.01324."""
        result = t_test([1, 2, 3, 4, 5])
        assert result.statistic == pytest.approx(4.2426406871192848, rel=1e-10)
        assert result.parameter == {"df": 4.0}
        assert result.p_value == pytest.approx(0.013235599563682695, rel=1e-10)

    def test_ci(self):
        """Confidence interval matches R."""
        result = t_test([1, 2, 3, 4, 5], mu=3)
        assert_allclose(result.conf_int, [1.036757, 4.963243], rtol=1e-5)

    def test_alternative_greater(self):
        """One-sided greater."""
        result = t_test([1, 2, 3, 4, 5], mu=0, alternative="greater")
        assert result.statistic == pytest.approx(4.242640687119285, rel=1e-10)
        assert result.p_value == pytest.approx(0.0066177997818413, rel=1e-10)
        assert result.conf_int[0] == pytest.approx(1.492556680937677, rel=1e-5)
        assert np.isinf(result.conf_int[1]) and result.conf_int[1] > 0

    def test_alternative_less(self):
        """One-sided less."""
        result = t_test([1, 2, 3, 4, 5], mu=0, alternative="less")
        assert result.statistic == pytest.approx(4.242640687119285, rel=1e-10)
        assert result.p_value == pytest.approx(0.99338220021815871, rel=1e-10)
        assert np.isinf(result.conf_int[0]) and result.conf_int[0] < 0
        assert result.conf_int[1] == pytest.approx(4.507443319062323, rel=1e-5)

    def test_nan_removal(self):
        """NaN values are silently removed."""
        result = t_test([1, 2, np.nan, 4, 5], mu=3)
        # Should behave like t_test([1, 2, 4, 5], mu=3)
        result2 = t_test([1, 2, 4, 5], mu=3)
        assert result.statistic == pytest.approx(result2.statistic, rel=1e-10)
        assert result.p_value == pytest.approx(result2.p_value, rel=1e-10)

    def test_summary_format(self):
        """summary() produces R-style output."""
        result = t_test([1, 2, 3, 4, 5], mu=3)
        s = result.summary()
        assert "One Sample t-test" in s
        assert "data:  x" in s
        assert "p-value" in s
        assert "confidence interval" in s
        assert "sample estimates" in s


class TestTwoSampleTTest:
    """Two-sample t-test (Welch and pooled)."""

    def test_welch_default(self):
        """t.test(1:5, 4:8) -> Welch (default)."""
        result = t_test([1, 2, 3, 4, 5], [4, 5, 6, 7, 8])
        assert result.statistic == pytest.approx(-3.0, rel=1e-10)
        assert result.parameter == {"df": pytest.approx(8.0, rel=1e-10)}
        assert result.p_value == pytest.approx(0.017071681233782634, rel=1e-10)
        assert result.method == "Welch Two Sample t-test"
        assert result.estimate == {
            "mean of x": pytest.approx(3.0),
            "mean of y": pytest.approx(6.0),
        }

    def test_welch_ci(self):
        """Welch CI matches R."""
        result = t_test([1, 2, 3, 4, 5], [4, 5, 6, 7, 8])
        assert_allclose(
            result.conf_int,
            [-5.306004135204166, -0.693995864795834],
            rtol=1e-5,
        )

    def test_pooled(self):
        """t.test(1:5, 4:8, var.equal=TRUE) -> pooled."""
        result = t_test(
            [1, 2, 3, 4, 5], [4, 5, 6, 7, 8], var_equal=True
        )
        assert result.statistic == pytest.approx(-3.0, rel=1e-10)
        assert result.parameter == {"df": pytest.approx(8.0, rel=1e-10)}
        assert result.p_value == pytest.approx(0.017071681233782634, rel=1e-10)
        assert result.method == " Two Sample t-test"

    def test_welch_fractional_df(self):
        """Welch df is fractional when variances differ."""
        # x has var=2.5, y has var=18.5 -> unequal
        result = t_test(
            [1, 2, 3, 4, 5], [2, 4, 5, 4, 5, 10, 1, 3]
        )
        # df should be fractional
        df = result.parameter["df"]
        assert df != int(df), "Welch df should be fractional for unequal variances"

    def test_mu_nonzero(self):
        """Two-sample with non-zero mu (testing difference != mu)."""
        result = t_test([1, 2, 3, 4, 5], [4, 5, 6, 7, 8], mu=-3)
        # Under H0: mean(x) - mean(y) = -3, actual diff = -3
        assert result.statistic == pytest.approx(0.0, abs=1e-15)
        assert result.p_value == pytest.approx(1.0, abs=1e-15)


class TestPairedTTest:
    """Paired t-test: H0: mean(x - y) = mu."""

    def test_basic(self):
        """t.test(c(1,2,3,4,5), c(2,4,5,4,5), paired=TRUE)."""
        result = t_test(
            [1, 2, 3, 4, 5], [2, 4, 5, 4, 5], paired=True
        )
        assert result.statistic == pytest.approx(-2.2360679774997898, rel=1e-10)
        assert result.parameter == {"df": pytest.approx(4.0)}
        assert result.p_value == pytest.approx(0.089009342500085645, rel=1e-10)
        assert result.method == "Paired t-test"
        assert result.estimate == {"mean difference": pytest.approx(-1.0)}

    def test_paired_ci(self):
        """Paired t-test CI matches R."""
        result = t_test(
            [1, 2, 3, 4, 5], [2, 4, 5, 4, 5], paired=True
        )
        assert_allclose(
            result.conf_int,
            [-2.241663998203764, 0.241663998203764],
            rtol=1e-5,
        )

    def test_paired_length_mismatch(self):
        """Paired test requires equal lengths."""
        with pytest.raises(Exception):
            t_test([1, 2, 3], [4, 5], paired=True)


class TestTTestEdgeCases:
    """Edge cases and validation."""

    def test_too_few_observations(self):
        """Need at least 2 observations."""
        with pytest.raises(Exception):
            t_test([5])

    def test_invalid_alternative(self):
        """Invalid alternative raises error."""
        with pytest.raises(Exception):
            t_test([1, 2, 3], alternative="invalid")

    def test_invalid_conf_level(self):
        """conf_level outside (0,1) raises error."""
        with pytest.raises(Exception):
            t_test([1, 2, 3], conf_level=1.5)

    def test_repr(self):
        """repr is informative."""
        result = t_test([1, 2, 3, 4, 5])
        r = repr(result)
        assert "HTestSolution" in r
        assert "p_value" in r

    def test_design_passthrough(self):
        """Can pass a pre-built HypothesisDesign."""
        from pystatistics.hypothesis.design import HypothesisDesign
        design = HypothesisDesign.for_t_test([1, 2, 3, 4, 5], mu=3)
        result = t_test(design)
        assert result.statistic == pytest.approx(0.0, abs=1e-15)

    def test_backend_name(self):
        """Backend name is 'cpu_hypothesis'."""
        result = t_test([1, 2, 3, 4, 5])
        assert result.backend_name == "cpu_hypothesis"

    def test_timing(self):
        """Timing is recorded."""
        result = t_test([1, 2, 3, 4, 5])
        assert result.timing is not None
