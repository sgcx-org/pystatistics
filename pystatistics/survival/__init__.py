"""
Survival analysis.

Public API:
    kaplan_meier(time, event, ...) -> KMSolution
    survdiff(time, event, group, ...) -> LogRankSolution
    coxph(time, event, X, ...) -> CoxSolution           # CPU only
    discrete_time(time, event, X, ...) -> DiscreteTimeSolution  # GPU accelerated
"""

from pystatistics.survival.solvers import (
    coxph,
    discrete_time,
    kaplan_meier,
    survdiff,
)
from pystatistics.survival.solution import (
    CoxSolution,
    DiscreteTimeSolution,
    KMSolution,
    LogRankSolution,
)

__all__ = [
    "coxph",
    "discrete_time",
    "kaplan_meier",
    "survdiff",
    "CoxSolution",
    "DiscreteTimeSolution",
    "KMSolution",
    "LogRankSolution",
]
