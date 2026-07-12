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
    method = boot_out._design.method
    statistic_type = boot_out._design.statistic_type

    n = data.shape[0]
    jack_stats = np.empty(n, dtype=np.float64)

    for i in range(n):
        # Leave-one-out indices
        loo_indices = np.concatenate([np.arange(i), np.arange(i + 1, n)])

        if method == "parametric":
            # For parametric bootstrap, jackknife on original data
            if data.ndim == 1:
                loo_data = data[loo_indices]
            else:
                loo_data = data[loo_indices]
            val = np.atleast_1d(np.asarray(statistic(loo_data)))
        else:
            # Nonparametric: call statistic with leave-one-out
            if statistic_type == "i":
                # Pass the leave-one-out data and identity indices
                if data.ndim == 1:
                    loo_data = data[loo_indices]
                else:
                    loo_data = data[loo_indices]
                loo_idx = np.arange(n - 1)
                val = np.atleast_1d(np.asarray(
                    statistic(loo_data, loo_idx)
                ))
            elif statistic_type == "f":
                freqs = np.ones(n, dtype=np.float64)
                freqs[i] = 0.0
                val = np.atleast_1d(np.asarray(
                    statistic(data, freqs)
                ))
            elif statistic_type == "w":
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


def _regen_index_matrix(design, rng, R: int, n: int) -> NDArray:
    """Regenerate the (R, n) resample-index matrix, mirroring the CPU backend's
    ``_ordinary_bootstrap`` / ``_balanced_bootstrap`` exactly (same RNG calls, in
    order), so the frequencies match the stored replicates. Supports ordinary and
    balanced simulations, with or without strata.
    """
    strata = design.strata
    idx = np.empty((R, n), dtype=int)

    if design.method == "ordinary":
        for b in range(R):
            if strata is None:
                idx[b] = rng.choice(n, size=n, replace=True)
            else:
                for s in np.unique(strata):
                    mask = strata == s
                    s_idx = np.where(mask)[0]
                    idx[b, mask] = rng.choice(s_idx, size=len(s_idx), replace=True)
        return idx

    # balanced
    if strata is None:
        pool = np.tile(np.arange(n), R)
        rng.shuffle(pool)
        return pool.reshape(R, n)
    for s in np.unique(strata):
        mask = strata == s
        ns = int(mask.sum())
        pool = np.tile(np.where(mask)[0], R)
        rng.shuffle(pool)
        for b in range(R):
            idx[b, mask] = pool[b * ns:(b + 1) * ns]
    return idx


def regression_influence(
    boot_out: 'BootstrapSolution',
    stat_index: int = 0,
) -> NDArray | None:
    """Regression estimate of the empirical influence values (R empinf type="reg").

    This is the estimate R's ``boot.ci`` uses BY DEFAULT for the BCa acceleration.
    It regresses the bootstrap replicates on the resample relative frequencies:

        L = pinv(P_c) @ (t - mean(t)),   P = freq/n,  P_c = P - colmean(P)

    (then centred to sum zero — within each stratum when the resampling is
    stratified), where ``freq`` is the R×n matrix of how many times each
    observation appeared in each replicate. The frequencies are regenerated
    deterministically from the stored seed rather than stored (matching R's own
    regenerate-from-seed approach), so there is no memory cost unless BCa is
    requested.

    Applies to the **ordinary and balanced** bootstrap, with or without strata —
    the resample frequencies exist and R's ``boot.ci`` uses this regression
    acceleration for all of them (jackknife-fallback BCa can shift a tail endpoint
    by several percent for a strongly non-linear statistic; see the A7
    measurement). Returns ``None`` (caller falls back to the jackknife) only when
    the estimate genuinely does not apply — a **parametric** simulation (no
    resample frequencies; R has the same limitation), no seed, or a regeneration
    that fails to reproduce the stored replicates (a self-check that prevents a
    silently-wrong influence if the resampling RNG path ever drifts).
    """
    design = boot_out._design
    if design.method not in ("ordinary", "balanced"):
        return None
    if design.seed is None:
        return None

    data = boot_out.data
    statistic = design.statistic
    statistic_type = design.statistic_type
    strata = design.strata
    t = boot_out.t[:, stat_index]
    n = data.shape[0]
    R = t.shape[0]

    rng = np.random.default_rng(design.seed)
    idx_matrix = _regen_index_matrix(design, rng, R, n)
    freq = np.stack([
        np.bincount(idx_matrix[b], minlength=n).astype(np.float64)
        for b in range(R)
    ])

    # Self-check: the regenerated replicate 0 must reproduce the stored t[0].
    # If it does not, the stored replicates did not come from this seed/path
    # (e.g. an injected/GPU solution) — fail safe to the jackknife.
    if statistic_type == "i":
        chk = np.atleast_1d(np.asarray(statistic(data, idx_matrix[0])))[stat_index]
    elif statistic_type == "f":
        chk = np.atleast_1d(np.asarray(statistic(data, freq[0])))[stat_index]
    else:  # "w"
        chk = np.atleast_1d(np.asarray(
            statistic(data, freq[0] / n)))[stat_index]
    if not np.isfinite(chk) or abs(chk - t[0]) > 1e-9 * (abs(t[0]) + 1e-12):
        return None

    P = freq / n
    Pc = P - P.mean(axis=0)
    # The frequency matrix is rank-deficient (each replicate's counts sum to n,
    # and within each stratum to n_s), so its trailing singular values are ~0.
    # A truncating rcond drops those degenerate directions; without it pinv's
    # min-norm solution loads them with huge (~1e12) offsetting values that only
    # cancel under centering with heavy round-off. The retained solution is the
    # same empirical-influence estimate, numerically clean.
    L = np.linalg.pinv(Pc, rcond=1e-8) @ (t - t.mean())
    if strata is None:
        return L - L.mean()
    # Stratified: the frequencies are collinear within each stratum, so L is
    # identified only up to a per-stratum constant — centre within strata
    # (matching R's empinf for a stratified boot object).
    L = L.copy()
    for s in np.unique(strata):
        mask = strata == s
        L[mask] -= L[mask].mean()
    return L
