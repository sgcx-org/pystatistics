"""
Ordinal regression (proportional odds / cumulative link models).

Provides the proportional odds model matching R's MASS::polr(),
with logistic, probit, and complementary log-log link functions.

Public API:
    polr(y, X, ...)        - Fit a proportional odds model
    OrdinalSolution        - Result wrapper with R-style summary()
"""

from pystatistics.ordinal._solver import polr
from pystatistics.ordinal.solution import OrdinalSolution

__all__ = [
    "polr",
    "OrdinalSolution",
]
