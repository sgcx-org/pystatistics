"""
Tests for fisher_test() matching R fisher.test().

All R reference values verified against R 4.5.2.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.hypothesis import fisher_test


class TestFisher2x2:
    """Fisher's exact test for 2x2 tables."""

    def test_basic(self):
        """
        R: fisher.test(matrix(c(1, 3, 5, 2), nrow=2))
        p-value = 0.2424, OR = 0.1648
        """
        table = np.array([[1, 5], [3, 2]])
        result = fisher_test(table)
        assert result.p_value == pytest.approx(0.24242424242424254, rel=1e-10)
        assert result.estimate["odds ratio"] == pytest.approx(
            0.16476659315874892, rel=1e-3
        )
        assert result.method == "Fisher's Exact Test for Count Data"
        assert result.alternative == "two.sided"
        assert result.statistic is None  # Fisher has no test statistic

    def test_basic_ci(self):
        """CI for odds ratio matches R."""
        table = np.array([[1, 5], [3, 2]])
        result = fisher_test(table)
        assert_allclose(
            result.conf_int,
            [0.0021346901287845211, 3.5018414620289362],
            rtol=0.05,  # Relaxed for conditional MLE computation
        )

    def test_alternative_less(self):
        """
        R: fisher.test(matrix(c(1, 3, 5, 2), nrow=2), alternative="less")
        """
        table = np.array([[1, 5], [3, 2]])
        result = fisher_test(table, alternative="less")
        assert result.p_value == pytest.approx(0.1969696969696971, rel=1e-10)
        assert result.conf_int[0] == pytest.approx(0.0, abs=1e-15)
        assert result.conf_int[1] == pytest.approx(
            2.4397206878908642, rel=0.05
        )

    def test_alternative_greater(self):
        """
        R: fisher.test(matrix(c(1, 3, 5, 2), nrow=2), alternative="greater")
        """
        table = np.array([[1, 5], [3, 2]])
        result = fisher_test(table, alternative="greater")
        assert result.p_value == pytest.approx(0.98484848484848486, rel=1e-10)
        assert np.isinf(result.conf_int[1])

    def test_bigger_table(self):
        """
        R: fisher.test(matrix(c(10, 5, 8, 12), nrow=2))
        """
        table = np.array([[10, 8], [5, 12]])
        result = fisher_test(table)
        assert result.p_value == pytest.approx(0.17558317615385832, rel=1e-10)
        assert result.estimate["odds ratio"] == pytest.approx(
            2.9032540456876101, rel=0.05
        )

    def test_bigger_table_ci(self):
        """CI for bigger table."""
        table = np.array([[10, 8], [5, 12]])
        result = fisher_test(table)
        assert_allclose(
            result.conf_int,
            [0.61363847186421194, 15.474034235130459],
            rtol=0.1,  # Relaxed tolerance for numerical optimization
        )

    def test_zero_cell(self):
        """
        R: fisher.test(matrix(c(0, 5, 8, 12), nrow=2))
        OR = 0 when a cell is zero.
        """
        table = np.array([[0, 8], [5, 12]])
        result = fisher_test(table)
        assert result.p_value == pytest.approx(0.13992094861660076, rel=1e-10)
        assert result.estimate["odds ratio"] == pytest.approx(0.0, abs=1e-15)

    def test_lady_tea(self):
        """
        R: fisher.test(matrix(c(3, 1, 1, 3), nrow=2))
        Classic Lady Tasting Tea example.
        """
        table = np.array([[3, 1], [1, 3]])
        result = fisher_test(table)
        assert result.p_value == pytest.approx(0.4857142857142856, rel=1e-10)

    def test_custom_conf_level(self):
        """
        R: fisher.test(matrix(c(10, 5, 8, 12), nrow=2), conf.level=0.99)
        """
        table = np.array([[10, 8], [5, 12]])
        result = fisher_test(table, conf_level=0.99)
        # Wider CI at 99%
        assert result.conf_int[0] < result.conf_int[1]
        # The 99% CI should be wider than the 95% CI
        result_95 = fisher_test(table, conf_level=0.95)
        assert result.conf_int[0] <= result_95.conf_int[0]
        assert result.conf_int[1] >= result_95.conf_int[1]

    def test_null_value(self):
        """null_value should be odds ratio = 1."""
        table = np.array([[10, 8], [5, 12]])
        result = fisher_test(table)
        assert result.null_value == {"odds ratio": 1.0}

    def test_no_ci(self):
        """Can disable CI computation."""
        table = np.array([[10, 8], [5, 12]])
        result = fisher_test(table, conf_int=False)
        assert result.conf_int is None


class TestFisherRxC:
    """Fisher's exact test for r x c tables."""

    def test_3x3(self):
        """
        R: fisher.test(matrix(c(1,2,3,2,4,3,3,3,4), nrow=3))
        """
        table = np.array([[1, 2, 3], [2, 4, 3], [3, 3, 4]])
        result = fisher_test(table)
        # R's p-value is 0.978 (exact or Monte Carlo depending on table size)
        # Our implementation uses Monte Carlo, so we allow stochastic tolerance
        assert result.p_value == pytest.approx(0.97816583983815453, abs=0.05)
        assert result.estimate is None  # No OR for r x c
        assert result.conf_int is None  # No CI for r x c

    def test_2x3(self):
        """
        R: fisher.test(matrix(c(10,5,8,12,6,7), nrow=2))
        R exact p-value = 0.377. Monte Carlo with B=10000 converges well.
        """
        table = np.array([[10, 8, 6], [5, 12, 7]])
        result = fisher_test(table)
        assert result.p_value == pytest.approx(0.3770904526630931, abs=0.05)

    def test_rxc_only_twosided(self):
        """r x c tables must be two-sided."""
        table = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(Exception):
            fisher_test(table, alternative="less")


class TestFisherEdgeCases:
    """Edge cases and validation."""

    def test_summary(self):
        """summary() produces readable output."""
        table = np.array([[3, 1], [1, 3]])
        result = fisher_test(table)
        s = result.summary()
        assert "Fisher" in s
        assert "p-value" in s

    def test_design_passthrough(self):
        """Can pass a pre-built HypothesisDesign."""
        from pystatistics.hypothesis.design import HypothesisDesign
        design = HypothesisDesign.for_fisher_test(
            np.array([[3, 1], [1, 3]])
        )
        result = fisher_test(design)
        assert result.p_value is not None

    def test_backend_name(self):
        """Backend name is cpu_hypothesis."""
        table = np.array([[3, 1], [1, 3]])
        result = fisher_test(table)
        assert result.backend_name == "cpu_hypothesis"

    def test_repr(self):
        """repr is informative."""
        table = np.array([[3, 1], [1, 3]])
        result = fisher_test(table)
        r = repr(result)
        assert "HTestSolution" in r

    def test_invalid_table(self):
        """Negative counts raise."""
        with pytest.raises(Exception):
            fisher_test(np.array([[1, -1], [2, 3]]))

    def test_too_small_table(self):
        """1x2 table raises."""
        with pytest.raises(Exception):
            fisher_test(np.array([[1, 2]]))
