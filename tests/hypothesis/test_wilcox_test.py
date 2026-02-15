"""
Tests for wilcox_test() matching R wilcox.test().

All R reference values verified against R 4.5.2.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.hypothesis import wilcox_test


class TestSignedRank:
    """Wilcoxon signed-rank test (one-sample)."""

    def test_basic(self):
        """
        R: wilcox.test(c(1.5, 2.2, 3.1, 4.0, 5.3), mu=3)
        V = 9, p-value = 0.8125 (exact)
        """
        result = wilcox_test([1.5, 2.2, 3.1, 4.0, 5.3], mu=3)
        assert result.statistic == pytest.approx(9.0, abs=1e-10)
        assert result.statistic_name == "V"
        assert result.p_value == pytest.approx(0.8125, rel=1e-10)
        assert "signed rank" in result.method.lower()

    def test_exact(self):
        """Exact test for small n, no ties."""
        result = wilcox_test([1.5, 2.2, 3.1, 4.0, 5.3], mu=3)
        assert "exact" in result.method.lower()

    def test_ci(self):
        """Hodges-Lehmann pseudomedian and CI."""
        result = wilcox_test(
            [1.5, 2.2, 3.1, 4.0, 5.3], mu=3, conf_int=True
        )
        # R: pseudomedian = 3.1, CI = [1.5, 5.3]
        assert result.estimate is not None
        assert "(pseudo)median" in result.estimate
        # Pseudomedian should be close to the median of Walsh averages + mu
        assert result.conf_int is not None

    def test_alternative_less(self):
        """
        R: wilcox.test(c(1.5, 2.2, 3.1, 4.0, 5.3), mu=3, alternative="less")
        V = 9, p = 0.6875
        """
        result = wilcox_test(
            [1.5, 2.2, 3.1, 4.0, 5.3], mu=3, alternative="less"
        )
        assert result.statistic == pytest.approx(9.0, abs=1e-10)
        assert result.p_value == pytest.approx(0.6875, rel=1e-10)


class TestRankSum:
    """Wilcoxon rank-sum (Mann-Whitney U) test."""

    def test_basic(self):
        """
        R: wilcox.test(1:5, 4:8)
        W = 2, p-value = 0.03558 (with continuity correction)
        """
        result = wilcox_test([1, 2, 3, 4, 5], [4, 5, 6, 7, 8])
        assert result.statistic == pytest.approx(2.0, abs=1e-10)
        assert result.statistic_name == "W"
        assert result.p_value == pytest.approx(
            0.03557883323959413, rel=1e-3
        )
        assert "rank sum" in result.method.lower()

    def test_no_correction(self):
        """
        R: wilcox.test(1:5, 4:8, correct=FALSE)
        W = 2, p = 0.02733
        """
        result = wilcox_test(
            [1, 2, 3, 4, 5], [4, 5, 6, 7, 8], correct=False
        )
        assert result.statistic == pytest.approx(2.0, abs=1e-10)
        assert result.p_value == pytest.approx(
            0.02732847198760490, rel=1e-3
        )

    def test_ci(self):
        """Hodges-Lehmann CI for rank-sum."""
        result = wilcox_test(
            [1, 2, 3, 4, 5], [4, 5, 6, 7, 8], conf_int=True
        )
        # R: estimate = -3, CI ≈ [-6, ~0]
        assert result.estimate is not None
        assert "difference in location" in result.estimate
        assert result.estimate["difference in location"] == pytest.approx(
            -3.0, abs=0.5
        )

    def test_with_ties(self):
        """
        R: wilcox.test(c(1, 1, 2, 3, 4), c(2, 3, 3, 4, 5))
        Normal approximation used with ties.
        """
        result = wilcox_test([1, 1, 2, 3, 4], [2, 3, 3, 4, 5])
        assert result.statistic == pytest.approx(6.0, abs=1e-10)
        assert result.p_value == pytest.approx(
            0.20025601192138390, rel=1e-2
        )
        # Should warn about ties
        assert any("ties" in w for w in result.warnings)

    def test_exact_small(self):
        """
        R: wilcox.test(c(1, 3, 5), c(2, 4, 6), exact=TRUE)
        W = 3, p = 0.7
        """
        result = wilcox_test([1, 3, 5], [2, 4, 6], exact=True)
        assert result.statistic == pytest.approx(3.0, abs=1e-10)
        assert result.p_value == pytest.approx(0.7, rel=1e-2)


class TestPairedWilcoxon:
    """Paired Wilcoxon test (reduces to signed-rank on differences)."""

    def test_basic(self):
        """
        R: wilcox.test(c(1,2,3,4,5), c(2,4,5,4,5), paired=TRUE)
        V = 0, p = 0.1736 (normal approx, ties in differences)
        """
        result = wilcox_test(
            [1, 2, 3, 4, 5], [2, 4, 5, 4, 5], paired=True
        )
        assert result.statistic == pytest.approx(0.0, abs=1e-10)
        assert result.p_value == pytest.approx(
            0.17356816655592155, rel=1e-2
        )


class TestWilcoxEdgeCases:
    """Edge cases."""

    def test_summary(self):
        """summary() produces readable output."""
        result = wilcox_test([1, 2, 3, 4, 5], mu=0)
        s = result.summary()
        assert "Wilcoxon" in s
        assert "p-value" in s

    def test_backend_name(self):
        """Backend is cpu_hypothesis."""
        result = wilcox_test([1, 2, 3, 4, 5], mu=0)
        assert result.backend_name == "cpu_hypothesis"

    def test_design_passthrough(self):
        """Can pass pre-built design."""
        from pystatistics.hypothesis.design import HypothesisDesign
        design = HypothesisDesign.for_wilcox_test(
            [1, 2, 3, 4, 5], mu=3
        )
        result = wilcox_test(design)
        assert result.statistic is not None

    def test_repr(self):
        """repr works."""
        result = wilcox_test([1, 2, 3], [4, 5, 6])
        r = repr(result)
        assert "HTestSolution" in r
