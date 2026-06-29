"""Boundary validation for regression prior weights and linear-predictor offset.

These two inputs cross the public ``fit()`` / ``ridge()`` boundary as raw
array-likes, so they are validated here once (Rule 2: validate input at the
boundary; Rule 1: fail loud). Both concepts are reserved library-wide — see
``pystatistics/CONVENTIONS.md``:

    weights : per-observation prior weights (n,); WLS weights for OLS, IRLS
              prior weights for a GLM. Non-negative, not all zero.
    offset  : additive term in the linear predictor, η = Xβ + offset (n,);
              a fixed quantity, never estimated.

Each resolver returns a contiguous float64 array of length ``n`` or ``None``
(the "unit weights" / "no offset" sentinel the backends treat as the fast
default), so downstream code never has to re-validate.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pystatistics.core.exceptions import ValidationError


def resolve_weights(weights: ArrayLike | None, n: int) -> NDArray[np.float64] | None:
    """Validate and normalize prior ``weights`` against ``n`` observations.

    Returns a float64 array of length ``n``, or ``None`` when ``weights`` is
    ``None`` (the unit-weight fast path). Fails loud on the wrong length, a
    non-finite entry, any negative weight, or an all-zero vector (R's
    ``glm`` rejects negative weights and needs positive total weight).
    """
    if weights is None:
        return None

    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1:
        raise ValidationError(
            f"weights must be 1-dimensional (n,), got {w.ndim}-d array"
        )
    if w.shape[0] != n:
        raise ValidationError(
            f"weights has {w.shape[0]} entries but the design has {n} "
            f"observations; lengths must match"
        )
    if not np.all(np.isfinite(w)):
        raise ValidationError("weights must all be finite (no NaN/Inf)")
    if np.any(w < 0):
        raise ValidationError("weights must be non-negative (negative prior weights are not allowed)")
    if not np.any(w > 0):
        raise ValidationError("weights must not all be zero (positive total weight required)")
    return np.ascontiguousarray(w)


def resolve_offset(offset: ArrayLike | None, n: int) -> NDArray[np.float64] | None:
    """Validate and normalize an ``offset`` against ``n`` observations.

    Returns a float64 array of length ``n``, or ``None`` when ``offset`` is
    ``None`` (the no-offset fast path). Fails loud on the wrong length or a
    non-finite entry. The offset enters the linear predictor as η = Xβ +
    offset and is not estimated.
    """
    if offset is None:
        return None

    off = np.asarray(offset, dtype=np.float64)
    if off.ndim != 1:
        raise ValidationError(
            f"offset must be 1-dimensional (n,), got {off.ndim}-d array"
        )
    if off.shape[0] != n:
        raise ValidationError(
            f"offset has {off.shape[0]} entries but the design has {n} "
            f"observations; lengths must match"
        )
    if not np.all(np.isfinite(off)):
        raise ValidationError("offset must all be finite (no NaN/Inf)")
    return np.ascontiguousarray(off)
