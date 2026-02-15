"""
Tests for chisq_test() matching R chisq.test().

All R reference values verified against R 4.5.2.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.hypothesis import chisq_test


class TestChisqIndependence:
    """Chi-squared test of independence (contingency tables)."""

    def test_2x3_table(self):
        """
        R: chisq.test(matrix(c(10,20,30,10,20,10), nrow=2, byrow=TRUE))
        X-squared = 6.25, df = 2, p-value = 0.04394
        """
        table = np.array([[10, 20, 30], [10, 20, 10]])
        result = chisq_test(table)
        assert result.statistic == pytest.approx(6.25, rel=1e-10)
        assert result.parameter == {"df": 2.0}
        assert result.p_value == pytest.approx(0.04393693362340742, rel=1e-10)
        assert result.method == "Pearson's Chi-squared test"

    def test_2x3_expected(self):
        """Expected counts match R."""
        table = np.array([[10, 20, 30], [10, 20, 10]])
        result = chisq_test(table)
        expected = np.array([[12, 24, 24], [8, 16, 16]])
        assert_allclose(result.expected, expected, rtol=1e-10)

    def test_2x3_residuals(self):
        """Pearson residuals match R."""
        table = np.array([[10, 20, 30], [10, 20, 10]])
        result = chisq_test(table)
        # residuals = (O - E) / sqrt(E)
        # R values
        expected_res = np.array([
            [(10 - 12) / np.sqrt(12), (20 - 24) / np.sqrt(24), (30 - 24) / np.sqrt(24)],
            [(10 - 8) / np.sqrt(8), (20 - 16) / np.sqrt(16), (10 - 16) / np.sqrt(16)],
        ])
        assert_allclose(result.residuals, expected_res, rtol=1e-10)

    def test_2x3_stdres(self):
        """Standardized residuals match R."""
        table = np.array([[10, 20, 30], [10, 20, 10]])
        result = chisq_test(table)
        # From R: stdres
        assert_allclose(
            result.stdres,
            [[-1.0206207261596576, -1.6666666666666667, 2.5],
             [1.0206207261596576, 1.6666666666666667, -2.5]],
            rtol=1e-10,
        )

    def test_2x2_yates(self):
        """
        R: chisq.test(matrix(c(10,5,8,12), nrow=2))
        Yates' correction applied by default for 2x2.
        """
        table = np.array([[10, 5], [8, 12]])
        result = chisq_test(table)
        assert result.statistic == pytest.approx(1.4893110021786498, rel=1e-10)
        assert result.parameter == {"df": 1.0}
        assert result.p_value == pytest.approx(0.2223233925159192, rel=1e-10)
        assert "Yates" in result.method

    def test_2x2_no_correction(self):
        """
        R: chisq.test(matrix(c(10,5,8,12), nrow=2), correct=FALSE)
        """
        table = np.array([[10, 5], [8, 12]])
        result = chisq_test(table, correct=False)
        assert result.statistic == pytest.approx(2.4400871459694997, rel=1e-10)
        assert result.p_value == pytest.approx(0.11826965501636745, rel=1e-10)
        assert "Yates" not in result.method

    def test_yates_only_2x2(self):
        """Yates correction is only applied to 2x2, even if correct=True."""
        table = np.array([[10, 20, 30], [10, 20, 10]])
        result = chisq_test(table, correct=True)
        # 2x3 table should NOT have Yates correction
        assert "Yates" not in result.method

    def test_small_expected_warning(self):
        """Warning when expected counts < 5."""
        table = np.array([[1, 2], [3, 4]])
        result = chisq_test(table)
        assert any("approximation may be incorrect" in w for w in result.warnings)

    def test_observed_preserved(self):
        """observed in extras matches input."""
        table = np.array([[10, 20], [30, 40]])
        result = chisq_test(table)
        assert_allclose(result.observed, table)


class TestChisqGOF:
    """Chi-squared goodness-of-fit test."""

    def test_uniform(self):
        """
        R: chisq.test(c(25, 25, 25, 25))
        Equal counts => X^2 = 0, p = 1.
        """
        result = chisq_test([25, 25, 25, 25])
        assert result.statistic == pytest.approx(0.0, abs=1e-15)
        assert result.parameter == {"df": 3.0}
        assert result.p_value == pytest.approx(1.0, abs=1e-15)
        assert "given probabilities" in result.method

    def test_specified_p(self):
        """
        R: chisq.test(c(30, 40, 30), p=c(0.25, 0.5, 0.25))
        X-squared = 4, df = 2, p-value = 0.1353
        """
        result = chisq_test([30, 40, 30], p=[0.25, 0.5, 0.25])
        assert result.statistic == pytest.approx(4.0, rel=1e-10)
        assert result.parameter == {"df": 2.0}
        assert result.p_value == pytest.approx(0.1353352832366127, rel=1e-10)

    def test_gof_expected(self):
        """Expected counts match R."""
        result = chisq_test([30, 40, 30], p=[0.25, 0.5, 0.25])
        assert_allclose(result.expected, [25.0, 50.0, 25.0], rtol=1e-10)

    def test_rescale_p(self):
        """
        R: chisq.test(c(30, 40, 30), p=c(1, 2, 1), rescale.p=TRUE)
        Same as p=c(0.25, 0.5, 0.25).
        """
        result = chisq_test([30, 40, 30], p=[1, 2, 1], rescale_p=True)
        assert result.statistic == pytest.approx(4.0, rel=1e-10)
        assert result.p_value == pytest.approx(0.1353352832366127, rel=1e-10)

    def test_gof_unequal(self):
        """GOF test with clearly unequal distribution."""
        result = chisq_test([100, 0, 0, 0])
        # All in one category vs uniform
        # E = 25 each, X^2 = 3*(25^2/25) + (75^2/25) = 75 + 225 = 300
        assert result.statistic == pytest.approx(300.0, rel=1e-10)
        assert result.parameter == {"df": 3.0}

    def test_p_not_summing_to_one(self):
        """Probabilities that don't sum to 1 should raise error."""
        with pytest.raises(Exception):
            chisq_test([10, 20, 30], p=[0.3, 0.3, 0.3])


class TestChisqSummary:
    """Test summary() output for chi-squared tests."""

    def test_independence_summary(self):
        """Summary format for independence test."""
        table = np.array([[10, 20, 30], [10, 20, 10]])
        result = chisq_test(table)
        s = result.summary()
        assert "Pearson's Chi-squared test" in s
        assert "X-squared" in s
        assert "p-value" in s

    def test_gof_summary(self):
        """Summary format for GOF test."""
        result = chisq_test([25, 25, 25, 25])
        s = result.summary()
        assert "given probabilities" in s


class TestChisqDesign:
    """Test design factory."""

    def test_passthrough(self):
        """Can pass a pre-built HypothesisDesign."""
        from pystatistics.hypothesis.design import HypothesisDesign
        design = HypothesisDesign.for_chisq_test(
            np.array([[10, 20], [30, 40]])
        )
        result = chisq_test(design)
        assert result.statistic is not None

    def test_crosstab(self):
        """Two 1D vectors build a contingency table."""
        # x=[1,1,2,2], y=[a,b,a,b] should give 2x2 table
        x = [1, 1, 2, 2, 1, 2]
        y = [0, 1, 0, 1, 0, 0]
        result = chisq_test(x, y)
        assert result.statistic is not None
        assert result.observed is not None

    def test_backend_name(self):
        """Backend name is cpu_hypothesis."""
        result = chisq_test([25, 25, 25, 25])
        assert result.backend_name == "cpu_hypothesis"

    def test_timing(self):
        """Timing recorded."""
        result = chisq_test([25, 25, 25, 25])
        assert result.timing is not None
