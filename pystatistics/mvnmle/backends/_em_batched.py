"""Compatibility shim for the batched EM/MCAR building blocks.

The implementation was split into three modules on 2026-04-20 to stay
under the 500-SLOC file limit (Coding Bible rule 4):

  * ``_em_batched_patterns`` — pattern-index dataclass + ``build_pattern_index``
  * ``_em_batched_np``       — NumPy backend (CPU)
  * ``_em_batched_torch``    — Torch backend (GPU + CPU torch)

This module continues to expose every name that callers imported
pre-split, so no call-site changes were required. New code may import
from the specific backend module directly.
"""
from pystatistics.mvnmle.backends._em_batched_patterns import (
    _BatchedPatternIndex,
    _pattern_n,
    build_pattern_index,
)
from pystatistics.mvnmle.backends._em_batched_np import (
    chi_square_mcar_batched_np,
    compute_conditional_parameters_np,
    compute_loglik_batched_np,
    e_step_full_batched_np,
)
from pystatistics.mvnmle.backends._em_batched_torch import (
    _e_step_full_torch,
    _loglik_full_torch,
    chi_square_mcar_batched_torch,
    compute_conditional_parameters_torch,
)

__all__ = [
    "_BatchedPatternIndex",
    "_pattern_n",
    "build_pattern_index",
    "chi_square_mcar_batched_np",
    "compute_conditional_parameters_np",
    "compute_loglik_batched_np",
    "e_step_full_batched_np",
    "_e_step_full_torch",
    "_loglik_full_torch",
    "chi_square_mcar_batched_torch",
    "compute_conditional_parameters_torch",
]
