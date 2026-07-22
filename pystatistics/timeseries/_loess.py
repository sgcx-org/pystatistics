"""
R-faithful univariate loess smoother for STL — public import surface.

Clean-room implementation of the local-regression smoother used inside R's
``stats::stl`` (Cleveland, Cleveland, McRae & Terpenning, 1990). The kernels
are compiled Cython (:mod:`._stl_kernels`), built with ``-ffp-contract=off`` so
they reproduce the pure-Python reference (:mod:`._stl_ref`) bit-for-bit and
track R's Fortran to floating-point noise (``test_stl_r_parity.py``). This
module keeps the historical import path (``loess_smooth_nb`` /
``loess_subseries_nb`` / ``_eval_window``) and the validated Python wrappers.

Semantics (all positions 1-based, neighbourhood on the integer design points
``1..n``): tricube neighbourhood weights clamped at ``0.001*h`` / ``0.999*h``,
optional degree-1 linear adjustment (skipped when the weighted design spread is
degenerate), jump-grid evaluation with linear interpolation and the trailing-
endpoint rule, and a zero-weight fallback to the observed value.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._stl_kernels import (  # noqa: F401
    _eval_window,
    loess_smooth_nb,
    loess_subseries_nb,
)


def loess_smooth(
    y: NDArray,
    span: int,
    degree: int,
    jump: int,
    weights: NDArray | None = None,
) -> NDArray:
    """Smooth a whole series, evaluating every *jump*-th point.

    Thin wrapper over :func:`loess_smooth_nb`. With ``jump=1`` every point is
    evaluated directly and no interpolation occurs.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    if weights is None:
        w = np.empty(0, dtype=np.float64)
        use_w = False
    else:
        w = np.ascontiguousarray(weights, dtype=np.float64)
        use_w = True
    return loess_smooth_nb(y, span, degree, jump, w, use_w)


def loess_subseries_smooth(
    sub_y: NDArray,
    span: int,
    degree: int,
    jump: int,
    sub_weights: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Smooth a group of equal-length cycle-subseries and extend each by one
    position at both ends — STL's cycle-subseries kernel.

    Thin wrapper over :func:`loess_subseries_nb`. Returns
    ``(smoothed (g,k), head (g,), tail (g,))``.
    """
    sub_y = np.ascontiguousarray(sub_y, dtype=np.float64)
    if sub_weights is None:
        w = np.empty((1, 1), dtype=np.float64)
        use_w = False
    else:
        w = np.ascontiguousarray(sub_weights, dtype=np.float64)
        use_w = True
    return loess_subseries_nb(sub_y, span, degree, jump, w, use_w)
