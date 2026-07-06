"""Tests for the exact Wilcoxon null-distribution quantiles and the
Hodges-Lehmann confidence intervals (``_wilcox_ci``), matching R's
``wilcox.test`` / ``qsignrank`` / ``qwilcox``.

All R reference values verified against R 4.5.2.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.hypothesis import wilcox_test
from pystatistics.hypothesis.backends import _wilcox_ci as wci


class TestExactQuantiles:
    """qsignrank / qwilcox against R."""

    def test_signrank_counts_sum_to_2n(self):
        for n in (1, 5, 6, 12):
            assert wci._signrank_counts(n).sum() == pytest.approx(2.0 ** n)

    def test_wilcox_counts_sum_to_choose(self):
        from math import comb
        for m, n in ((5, 5), (3, 7), (4, 6)):
            assert np.array(wci._wilcox_counts(m, n)).sum() == pytest.approx(
                comb(m + n, m))

    def test_qwilcox_matches_r(self):
        # R: qwilcox(0.025, 5, 5) = 3 ; qwilcox(0.05, 7, 7) = 12
        assert wci.qwilcox(0.025, 5, 5) == 3
        assert wci.qwilcox(0.05, 7, 7) == 12

    def test_qsignrank_matches_r(self):
        # R: qsignrank(0.025, 9) = 6 ; qsignrank(0.025, 15) = 26
        assert wci.qsignrank(0.025, 9) == 6
        assert wci.qsignrank(0.025, 15) == 26


class TestRankSumCI:
    """Two-sample Hodges-Lehmann CI vs R wilcox.test."""

    def test_exact_two_sided(self):
        x = [1.83, 0.50, 1.62, 2.48, 1.68]
        y = [0.878, 0.647, 0.598, 2.05, 1.06]
        # R: wilcox.test(x, y, conf.int=TRUE)$conf.int -> (-0.43, 1.602)
        r = wilcox_test(x, y, conf_int=True)
        assert_allclose(r.conf_int, [-0.43, 1.602], atol=1e-9)

    def test_exact_one_sided(self):
        x = [1.83, 0.50, 1.62, 2.48, 1.68]
        y = [0.878, 0.647, 0.598, 2.05, 1.06]
        # R greater -> (-0.37, Inf) ; less -> (-Inf, 1.232)
        rg = wilcox_test(x, y, conf_int=True, alternative="greater")
        assert_allclose(rg.conf_int[0], -0.37, atol=1e-9)
        assert rg.conf_int[1] == np.inf
        rl = wilcox_test(x, y, conf_int=True, alternative="less")
        assert rl.conf_int[0] == -np.inf
        assert_allclose(rl.conf_int[1], 1.232, atol=1e-9)

    def test_approx_ties(self):
        x = [1.0, 2, 2, 3, 4, 4, 5]
        y = [2.0, 3, 3, 4, 5, 5, 6]
        # R (ties -> normal approx): conf.int ~ (-3.0, 1.0) to uniroot tol
        r = wilcox_test(x, y, conf_int=True)
        assert_allclose(r.conf_int, [-3.0, 1.0], atol=1e-3)


class TestSignedRankCI:
    """Signed-rank Hodges-Lehmann CI + pseudomedian vs R."""

    def test_exact_two_sided(self):
        d = [1.83, 0.50, 1.62, 2.48, 1.68, -0.3, 0.9, -1.2, 2.1, 0.05]
        # R: wilcox.test(d, conf.int=TRUE)$conf.int -> (0.1, 1.86)
        r = wilcox_test(d, conf_int=True)
        assert_allclose(r.conf_int, [0.1, 1.86], atol=1e-9)

    def test_zero_differences_dropped(self):
        # A zero difference is dropped before CI + pseudomedian (matching R).
        p1 = [125.0, 115, 130, 140, 140, 115, 140, 125, 140, 135]
        p2 = [110.0, 122, 125, 120, 140, 124, 123, 137, 135, 145]
        r = wilcox_test(p1, p2, paired=True, conf_int=True)
        # R: pseudomedian = 3.5, conf.int = (-9.5, 15.0)
        assert_allclose(r.estimate["(pseudo)median"], 3.5, atol=1e-3)
        assert_allclose(r.conf_int, [-9.5, 15.0], atol=1e-3)
