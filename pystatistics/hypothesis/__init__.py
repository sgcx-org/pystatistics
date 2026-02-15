"""
Hypothesis testing module.

Provides hypothesis tests matching R's implementations,
validated against R to rtol=1e-10.

Public API:
    t_test(x, y)         - Student's t-test (one-sample, two-sample, paired)
    chisq_test(x)        - Pearson's chi-squared test (independence, GOF)
    fisher_test(x)       - Fisher's exact test (2x2 and r x c)
    wilcox_test(x, y)    - Wilcoxon rank-sum / signed-rank test
    ks_test(x, y)        - Kolmogorov-Smirnov test
    prop_test(x, n)      - Test of proportions
    var_test(x, y)       - F-test to compare two variances
    p_adjust(p)          - Multiple testing correction (Holm, BH, etc.)
"""

from pystatistics.hypothesis.solvers import (
    t_test, chisq_test, fisher_test, wilcox_test, ks_test, prop_test,
    var_test,
)
from pystatistics.hypothesis._p_adjust import p_adjust
from pystatistics.hypothesis.design import HypothesisDesign
from pystatistics.hypothesis._common import HTestParams
from pystatistics.hypothesis.solution import HTestSolution

__all__ = [
    "t_test",
    "chisq_test",
    "fisher_test",
    "wilcox_test",
    "ks_test",
    "prop_test",
    "var_test",
    "p_adjust",
    "HypothesisDesign",
    "HTestParams",
    "HTestSolution",
]
