"""
Survival analysis.

Public API:
    kaplan_meier(time, event, ...) -> KMSolution
    survdiff(time, event, group, ...) -> LogRankSolution
    coxph(time, event, X, ...) -> CoxSolution           # CPU only
    discrete_time(time, event, X, ...) -> DiscreteTimeSolution  # GPU accelerated
"""

from pystatistics.survival.solvers import (
    cox_zph,
    coxph,
    discrete_time,
    kaplan_meier,
    survdiff,
)
from pystatistics.survival._cox_zph import CoxZphSolution
from pystatistics.survival.solution import (
    CoxSolution,
    DiscreteTimeSolution,
    KMSolution,
    LogRankSolution,
)
from pystatistics.survival._km_strata import StratifiedKMSolution

__all__ = [
    "cox_zph",
    "coxph",
    "discrete_time",
    "kaplan_meier",
    "survdiff",
    "CoxSolution",
    "CoxZphSolution",
    "DiscreteTimeSolution",
    "KMSolution",
    "LogRankSolution",
    "StratifiedKMSolution",
]
