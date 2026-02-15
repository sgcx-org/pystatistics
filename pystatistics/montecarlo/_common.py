"""
Common data structures for Monte Carlo methods.

BootParams and PermutationParams are the parameter payloads
wrapped by Result[P] and exposed through Solution classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BootParams:
    """
    Parameter payload for bootstrap results.

    Matches R's boot object structure:
    - t0: observed statistic(s) on original data
    - t: matrix of bootstrap replicates (R rows, k columns)
    - bias: mean(t) - t0
    - se: sd(t)
    - ci: confidence intervals (populated by boot_ci)
    """
    t0: NDArray[np.floating[Any]]              # shape (k,)
    t: NDArray[np.floating[Any]]               # shape (R, k)
    R: int                                      # number of replicates
    bias: NDArray[np.floating[Any]]            # shape (k,)
    se: NDArray[np.floating[Any]]              # shape (k,)
    ci: dict[str, NDArray] | None = None       # keyed by CI type
    ci_conf_level: float | None = None


@dataclass(frozen=True)
class PermutationParams:
    """
    Parameter payload for permutation test results.

    - observed_stat: test statistic on original (unpermuted) data
    - perm_stats: test statistics from R permutations
    - p_value: (count + 1) / (R + 1) with Phipson-Smyth correction
    """
    observed_stat: float
    perm_stats: NDArray[np.floating[Any]]      # shape (R,)
    p_value: float
    R: int
    alternative: str                            # "two.sided" | "less" | "greater"
