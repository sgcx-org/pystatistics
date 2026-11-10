"""Group the rows of an incomplete data matrix by missingness pattern.

One job: partition the rows of ``data`` into groups that share the same set
of observed variables, and return each group's observed sub-matrix.

A missingness pattern is the boolean vector saying which variables are
missing in a row. Rows are grouped by exact equality of that vector, compared
as packed bytes, so the grouping is exact for any number of variables.

Order (deterministic, and independent of the order of the rows in ``data``):

1. patterns with fewer missing variables come first, so complete cases, when
   present, are pattern 0;
2. ties are broken lexicographically on the missing-indicator vector
   (``False < True``, variable 0 most significant).

Within a pattern, rows keep their original relative order.

Input contract: ``data`` is a 2-D float array with ``np.nan`` marking missing
values. Rows that are entirely missing form a pattern with no observed
variables; rejecting them is the caller's validation concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ValidationError


@dataclass(frozen=True)
class PatternData:
    """The rows of the data that share one missingness pattern."""

    pattern_id: int                       # position in the pattern order
    n_obs: int                            # number of rows with this pattern
    data: NDArray[np.float64]             # (n_obs, n_observed) observed values
    observed_indices: NDArray[np.int64]   # column indices observed
    missing_indices: NDArray[np.int64]    # column indices missing
    row_indices: NDArray[np.int64]        # positions of the rows in the input


def group_rows_by_pattern(data: NDArray[np.float64]) -> List[PatternData]:
    """Partition the rows of ``data`` by missingness pattern.

    Parameters
    ----------
    data : np.ndarray, shape (n_obs, n_vars)
        Data matrix with ``np.nan`` for missing values.

    Returns
    -------
    list of PatternData
        One entry per distinct pattern, in the order documented in the module
        docstring. Every row of ``data`` appears in exactly one entry.

    Raises
    ------
    ValidationError
        If ``data`` is not two-dimensional or has no rows.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValidationError(f"Data must be 2D, got shape {data.shape}")
    n_obs, _ = data.shape
    if n_obs == 0:
        raise ValidationError("Data has no rows to group")

    missing = np.isnan(data)

    # Exact pattern identity: one byte string per row. np.unique sorts the
    # packed rows lexicographically, which (big-endian bit packing) is the
    # lexicographic order of the missing-indicator vectors themselves.
    packed = np.packbits(missing, axis=1)
    _, first_row, inverse = np.unique(
        packed, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)

    # Fewest-missing first; the stable sort keeps the lexicographic tie-break.
    n_missing = missing[first_row].sum(axis=1)
    order = np.argsort(n_missing, kind='stable')
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    pattern_of_row = rank[inverse]

    # Stable sort: rows of a pattern stay in their original relative order.
    rows_by_pattern = np.argsort(pattern_of_row, kind='stable')
    counts = np.bincount(pattern_of_row, minlength=order.size)
    stops = np.cumsum(counts)

    patterns: List[PatternData] = []
    for k in range(order.size):
        rows = rows_by_pattern[stops[k] - counts[k]:stops[k]]
        is_missing = missing[rows[0]]
        observed = np.flatnonzero(~is_missing)
        patterns.append(PatternData(
            pattern_id=k,
            n_obs=int(counts[k]),
            data=data[np.ix_(rows, observed)],
            observed_indices=observed,
            missing_indices=np.flatnonzero(is_missing),
            row_indices=rows,
        ))
    return patterns
