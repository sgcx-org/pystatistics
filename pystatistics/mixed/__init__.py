"""
Mixed models: Linear Mixed Models (LMM) and Generalized Linear Mixed Models (GLMM).

Public API:
    lmm()           — fit a linear mixed model (REML or ML)
    glmm()          — fit a generalized linear mixed model (Laplace approximation)
    LMMSolution     — result wrapper for LMM
    GLMMSolution    — result wrapper for GLMM
"""

from pystatistics.mixed.solvers import lmm, glmm
from pystatistics.mixed.solution import LMMSolution, GLMMSolution

__all__ = [
    "lmm",
    "glmm",
    "LMMSolution",
    "GLMMSolution",
]
