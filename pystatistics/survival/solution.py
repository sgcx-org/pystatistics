"""
Solution wrappers for survival analysis results — compatibility facade.

Each Solution class lives in its own module (one module, one job):
    _solution_km.py        KMSolution
    _solution_logrank.py   LogRankSolution
    _solution_cox.py       CoxSolution
    _solution_discrete.py  DiscreteTimeSolution

This module re-exports them so existing imports of
``pystatistics.survival.solution`` keep working unchanged.
"""

from __future__ import annotations

from pystatistics.survival._solution_cox import CoxSolution
from pystatistics.survival._solution_discrete import DiscreteTimeSolution
from pystatistics.survival._solution_km import KMSolution
from pystatistics.survival._solution_logrank import LogRankSolution

__all__ = [
    "CoxSolution",
    "DiscreteTimeSolution",
    "KMSolution",
    "LogRankSolution",
]
