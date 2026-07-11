"""
Generalized Additive Models (GAM).

Provides smooth term specification and basis construction for
penalized regression spline GAMs following Wood (2017).

Usage::

    from pystatistics.gam import gam, s, te, ti

    sol = gam(
        y,
        smooths=[s('x1', k=15), te('x2', 'x3')],
        smooth_data={'x1': x1, 'x2': x2, 'x3': x3},
    )
    print(sol.summary())
"""

from pystatistics.gam._common import GAMParams, SmoothInfo
from pystatistics.gam._smooth import s, SmoothTerm
from pystatistics.gam._tensor_smooth import te, ti, TensorSmooth
from pystatistics.gam._gam import gam
from pystatistics.gam.solution import GAMSolution

__all__ = [
    "GAMParams",
    "SmoothInfo",
    "s",
    "SmoothTerm",
    "te",
    "ti",
    "TensorSmooth",
    "gam",
    "GAMSolution",
]
