"""
STL robustness weighting, matching R ``stats::stl`` — public import surface.

The outer STL loop downweights outlying observations with Tukey's bisquare
applied to the remainder, scaled by six times the median absolute residual.
The reference selects its two "middle" order statistics through a partial
quicksort whose behaviour for even-length inputs deviates from a true partial
sort (:func:`psort_pair_nb`); R inherits that behaviour, so exact parity
requires replicating the algorithm rather than substituting a correct median.

The kernels are compiled Cython (:mod:`._stl_kernels`); this module keeps the
historical import path and the validated Python wrappers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._stl_kernels import (  # noqa: F401
    psort_pair_nb,
    robustness_weights_nb,
)


def _psort_pair(a: NDArray, first: int, second: int) -> tuple[float, float]:
    """Wrapper over :func:`psort_pair_nb` (see it for the semantics)."""
    return psort_pair_nb(np.ascontiguousarray(a, dtype=np.float64), first, second)


def _robustness_weights(y: NDArray, fit: NDArray) -> NDArray:
    """Wrapper over :func:`robustness_weights_nb`."""
    return robustness_weights_nb(
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(fit, dtype=np.float64),
    )
