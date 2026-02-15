"""
Analysis of Variance (ANOVA).

Public API:
    anova_oneway(y, group, ...) -> AnovaSolution
    anova(y, factors, ...) -> AnovaSolution         # factorial / ANCOVA
    anova_rm(y, subject, within, ...) -> AnovaRMSolution    # repeated measures
    anova_posthoc(result, ...) -> PostHocSolution    # Tukey / Bonferroni / Dunnett
    levene_test(y, group, ...) -> LeveneSolution     # homogeneity of variances
"""

from pystatistics.anova.solvers import (
    anova,
    anova_oneway,
    anova_posthoc,
    anova_rm,
    levene_test,
)
from pystatistics.anova.solution import (
    AnovaSolution,
    AnovaRMSolution,
    LeveneSolution,
    PostHocSolution,
)

__all__ = [
    "anova",
    "anova_oneway",
    "anova_posthoc",
    "anova_rm",
    "levene_test",
    "AnovaSolution",
    "AnovaRMSolution",
    "LeveneSolution",
    "PostHocSolution",
]
