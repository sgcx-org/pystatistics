"""
Mixed models: Linear Mixed Models (LMM) and Generalized Linear Mixed Models (GLMM).

Public API:
    lmm()           — fit a linear mixed model (REML or ML)
    glmm()          — fit a generalized linear mixed model (Laplace approximation)
    grm_lmm()       — fit a low-rank / GRM mixed model (CPU/GPU; genomics regime)
    LMMSolution     — result wrapper for LMM
    GLMMSolution    — result wrapper for GLMM
    GRMSolution     — result wrapper for the low-rank / GRM mixed model
"""

from pystatistics.mixed.solvers import lmm, glmm
from pystatistics.mixed.solution import LMMSolution, GLMMSolution
from pystatistics.mixed.grm import grm_lmm
from pystatistics.mixed.grm_solution import GRMSolution

__all__ = [
    "lmm",
    "glmm",
    "grm_lmm",
    "LMMSolution",
    "GLMMSolution",
    "GRMSolution",
]
