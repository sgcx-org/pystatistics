"""
Generalized Additive Models (GAM).

Provides smooth term specification and basis construction for
penalized regression spline GAMs following Wood (2017).

Usage::

    from pystatistics.gam import s

    # Declare smooth terms
    smooths = [s('x1', k=15), s('x2', bs='tp')]

The fitting function ``gam()`` and ``GAMSolution`` will be added
by the fitting module and appended to this file's exports.
"""

from pystatistics.gam._common import GAMParams, SmoothInfo
from pystatistics.gam._smooth import s, SmoothTerm
from pystatistics.gam._gam import gam
from pystatistics.gam.solution import GAMSolution

__all__ = [
    "GAMParams",
    "SmoothInfo",
    "s",
    "SmoothTerm",
    "gam",
    "GAMSolution",
]
