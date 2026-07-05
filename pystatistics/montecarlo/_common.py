"""
Common data structures for Monte Carlo methods.

BootParams and PermutationParams are the parameter payloads
wrapped by Result[P] and exposed through Solution classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ValidationError


def perm_pvalue(perm_stats: NDArray, observed: float, alternative: str,
                R: int) -> float:
    """Permutation p-value with the Phipson-Smyth (count+1)/(R+1) correction.

    One-sided uses the natural tail count. The TWO-SIDED value doubles the
    smaller tail — ``min(1, 2*min(p_greater, p_less))`` — rather than counting
    ``|perm| >= |obs|``. The tail-doubling is correct for ANY statistic, not
    only one whose permutation null is centred at zero: for a null-centred
    statistic (a difference in means) it equals the ``|.|`` count (both tails are
    equal), and for a non-centred statistic (a ratio of means) it gives the
    proper two-sided value where ``|perm| >= |obs|`` would not. It is also
    atom-safe on a discrete permutation distribution — unlike centring on the
    (noisy) empirical mean, which can arbitrarily include/exclude the observed
    atom.
    """
    if alternative == "greater":
        count = int(np.sum(perm_stats >= observed))
        return float(count + 1) / float(R + 1)
    if alternative == "less":
        count = int(np.sum(perm_stats <= observed))
        return float(count + 1) / float(R + 1)
    if alternative == "two-sided":
        p_g = float(int(np.sum(perm_stats >= observed)) + 1) / float(R + 1)
        p_l = float(int(np.sum(perm_stats <= observed)) + 1) / float(R + 1)
        return min(1.0, 2.0 * min(p_g, p_l))
    raise ValidationError(f"Unknown alternative: {alternative!r}")


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
    alternative: str                            # "two-sided" | "less" | "greater"
