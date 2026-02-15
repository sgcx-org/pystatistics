"""
Jackknife influence values for BCa confidence intervals.

Computes leave-one-out jackknife influence values used by the BCa
method to estimate the acceleration parameter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pystatistics.montecarlo.solution import BootstrapSolution


def jackknife_influence(
    boot_out: 'BootstrapSolution',
    stat_index: int = 0,
) -> NDArray:
    """
    Compute jackknife influence values for a bootstrap statistic.

    Uses the standard delete-1 jackknife:
        L_i = (n-1) * (theta_bar_{-i} - theta_{-i})

    where theta_{-i} is the statistic computed on data with observation i removed,
    and theta_bar_{-i} is the mean of all leave-one-out estimates.

    Args:
        boot_out: Bootstrap solution containing data and statistic function.
        stat_index: Which element of the statistic vector to compute
            influence for (0-indexed).

    Returns:
        Influence values, shape (n,).
    """
    data = boot_out.data
    statistic = boot_out._design.statistic
    sim = boot_out._design.sim
    stype = boot_out._design.stype

    n = data.shape[0]
    jack_stats = np.empty(n, dtype=np.float64)

    for i in range(n):
        # Leave-one-out indices
        loo_indices = np.concatenate([np.arange(i), np.arange(i + 1, n)])

        if sim == "parametric":
            # For parametric bootstrap, jackknife on original data
            if data.ndim == 1:
                loo_data = data[loo_indices]
            else:
                loo_data = data[loo_indices]
            val = np.atleast_1d(np.asarray(statistic(loo_data)))
        else:
            # Nonparametric: call statistic with leave-one-out
            if stype == "i":
                # Pass the leave-one-out data and identity indices
                if data.ndim == 1:
                    loo_data = data[loo_indices]
                else:
                    loo_data = data[loo_indices]
                loo_idx = np.arange(n - 1)
                val = np.atleast_1d(np.asarray(
                    statistic(loo_data, loo_idx)
                ))
            elif stype == "f":
                freqs = np.ones(n, dtype=np.float64)
                freqs[i] = 0.0
                val = np.atleast_1d(np.asarray(
                    statistic(data, freqs)
                ))
            elif stype == "w":
                weights = np.ones(n, dtype=np.float64)
                weights[i] = 0.0
                weights = weights / weights.sum()
                val = np.atleast_1d(np.asarray(
                    statistic(data, weights)
                ))

        jack_stats[i] = val[stat_index]

    # Influence values: L_i = (n-1) * (mean_jack - jack_i)
    mean_jack = np.mean(jack_stats)
    L = (n - 1) * (mean_jack - jack_stats)

    return L
