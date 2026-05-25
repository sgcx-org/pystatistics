"""
ANOVA contrast coding — re-export shim.

The categorical-encoding engine is shared library-wide and lives in
``pystatistics.core.encoding`` (used by both ANOVA and regression). ANOVA
historically refers to these operations with R's "contrast" vocabulary, so
this module re-exports the engine under the legacy ANOVA-facing names:

    encode_treatment   -> encode_dummy
    build_model_matrix -> build_design_matrix
    ModelMatrix        -> DesignMatrix
"""

from pystatistics.core.encoding import (
    DesignMatrix as ModelMatrix,
    build_design_matrix as build_model_matrix,
    encode_deviation,
    encode_dummy as encode_treatment,
    interaction_columns,
)

__all__ = [
    "ModelMatrix",
    "build_model_matrix",
    "encode_deviation",
    "encode_treatment",
    "interaction_columns",
]
