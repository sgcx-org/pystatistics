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


def regression_influence(
    boot_out: 'BootstrapSolution',
    stat_index: int = 0,
) -> NDArray | None:
    """Regression estimate of the empirical influence values (R empinf type="reg").

    This is the estimate R's ``boot.ci`` uses BY DEFAULT for the BCa acceleration.
    It regresses the bootstrap replicates on the resample relative frequencies:

        L = pinv(P_c) @ (t - mean(t)),   P = freq/n,  P_c = P - colmean(P)

    (then centred to sum zero), where ``freq`` is the R×n matrix of how many times
    each observation appeared in each replicate. The frequencies are regenerated
    deterministically from the stored seed rather than stored (matching R's own
    regenerate-from-seed approach), so there is no memory cost unless BCa is
    requested.

    Returns ``None`` (caller falls back to the jackknife) when the estimate does
    not apply — a non-ordinary simulation, stratified resampling, no seed, or a
    regeneration that fails to reproduce the stored replicates (a self-check that
    prevents a silently-wrong influence if the resampling RNG path ever drifts).
    """
    design = boot_out._design
    if design.sim != "ordinary" or design.strata is not None:
        return None
    if design.seed is None:
        return None

    data = boot_out.data
    statistic = design.statistic
    stype = design.stype
    t = boot_out.t[:, stat_index]
    n = data.shape[0]
    R = t.shape[0]

    rng = np.random.default_rng(design.seed)
    freq = np.empty((R, n), dtype=np.float64)
    first_idx = None
    for b in range(R):
        idx = rng.choice(n, size=n, replace=True)
        if b == 0:
            first_idx = idx
        freq[b] = np.bincount(idx, minlength=n)

    # Self-check: the regenerated replicate 0 must reproduce the stored t[0].
    # If it does not, the stored replicates did not come from this seed/path
    # (e.g. an injected/GPU solution) — fail safe to the jackknife.
    if stype == "i":
        chk = np.atleast_1d(np.asarray(statistic(data, first_idx)))[stat_index]
    elif stype == "f":
        chk = np.atleast_1d(np.asarray(statistic(data, freq[0])))[stat_index]
    else:  # "w"
        chk = np.atleast_1d(np.asarray(
            statistic(data, freq[0] / n)))[stat_index]
    if not np.isfinite(chk) or abs(chk - t[0]) > 1e-9 * (abs(t[0]) + 1e-12):
        return None

    P = freq / n
    Pc = P - P.mean(axis=0)
    L = np.linalg.pinv(Pc) @ (t - t.mean())
    return L - L.mean()
