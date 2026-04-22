"""Compatibility shim for the batched EM building blocks.

The implementation was split into three modules on 2026-04-20 to stay
under the 500-SLOC file limit (Coding Bible rule 4):

  * ``_em_batched_patterns`` — pattern-index dataclass + ``build_pattern_index``
  * ``_em_batched_np``       — NumPy backend (CPU)
  * ``_em_batched_torch``    — Torch backend (GPU + CPU torch)

This module continues to expose every remaining name that callers
imported pre-split, so no call-site changes were required. New code
may import from the specific backend module directly.

The ``chi_square_mcar_batched_np`` / ``chi_square_mcar_batched_torch``
functions that used to be re-exported here (and lived in the np /
torch backend files) were removed in 3.0.0 together with
``mom_mcar_test``, which was their only caller.
"""
from pystatistics.mvnmle.backends._em_batched_patterns import (
    _BatchedPatternIndex,
    _pattern_n,
    build_pattern_index,
)
from pystatistics.mvnmle.backends._em_batched_np import (
    compute_conditional_parameters_np,
    compute_loglik_batched_np,
    e_step_full_batched_np,
)
from pystatistics.mvnmle.backends._em_batched_torch import (
    _e_step_full_torch,
    _loglik_full_torch,
    compute_conditional_parameters_torch,
)

__all__ = [
    "_BatchedPatternIndex",
    "_pattern_n",
    "build_pattern_index",
    "compute_conditional_parameters_np",
    "compute_loglik_batched_np",
    "e_step_full_batched_np",
    "_e_step_full_torch",
    "_loglik_full_torch",
    "compute_conditional_parameters_torch",
]
