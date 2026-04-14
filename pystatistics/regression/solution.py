"""
Regression solution types — re-export shim.

Linear and GLM solutions are implemented in separate modules per Rule 3
(one module, one job). This file re-exports them for backward compatibility.
"""

from pystatistics.regression._linear import LinearParams, LinearSolution
from pystatistics.regression._glm import GLMParams, GLMSolution
from pystatistics.regression._formatting import significance_stars as _significance_stars

__all__ = [
    "LinearParams",
    "LinearSolution",
    "GLMParams",
    "GLMSolution",
]
