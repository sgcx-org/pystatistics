"""
Bootstrap confidence interval computation.

Implements all 5 methods from R's boot.ci():
- normal: bias-corrected normal approximation
- basic: basic (pivotal) bootstrap interval
- perc: percentile method
- bca: bias-corrected and accelerated
- stud: studentized (bootstrap-t)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

if TYPE_CHECKING:
    from pystatistics.montecarlo.solution import BootstrapSolution


def compute_ci(
    boot_out: 'BootstrapSolution',
    types: list[str],
    conf_level: float,
    index: int = 0,
    var_t0: float | None = None,
    var_t: NDArray | None = None,
) -> dict[str, NDArray]:
    """
    Compute bootstrap confidence intervals.

    Args:
        boot_out: Bootstrap result.
        types: List of CI types to compute.
        conf_level: Confidence level (e.g., 0.95).
        index: Which statistic (column of t) to use. Currently computes
            CI for all k statistics but uses index for studentized/BCa.
        var_t0: Variance of observed statistic.
        var_t: Per-replicate variances, shape (R,).

    Returns:
        Dict mapping CI type name to NDArray of shape (k, 2).
    """
    t0 = boot_out.t0
    t = boot_out.t
    k = len(t0)
    alpha = 1.0 - conf_level

    ci_dict: dict[str, NDArray] = {}

    for ci_type in types:
        if ci_type == "normal":
            ci_dict["normal"] = _ci_normal(t0, t, alpha, var_t0)
        elif ci_type == "basic":
            ci_dict["basic"] = _ci_basic(t0, t, alpha)
        elif ci_type == "perc":
            ci_dict["perc"] = _ci_percentile(t, alpha)
        elif ci_type == "bca":
            ci_dict["bca"] = _ci_bca(boot_out, alpha)
        elif ci_type == "stud":
            if var_t is None:
                raise ValueError(
                    "Studentized CI requires var_t "
                    "(per-replicate variance estimates)"
                )
            ci_dict["stud"] = _ci_studentized(
                t0, t, alpha, var_t0, var_t, index,
            )
        else:
            raise ValueError(f"Unknown CI type: {ci_type!r}")

    return ci_dict


def _ci_normal(
    t0: NDArray,
    t: NDArray,
    alpha: float,
    var_t0: float | None,
) -> NDArray:
    """
    Normal approximation CI with bias correction.

    CI = [2*t0 - mean(t) + z_{alpha/2} * se,
          2*t0 - mean(t) + z_{1-alpha/2} * se]

    Centered at 2*t0 - mean(t) (bias-corrected), not at t0.
    """
    k = len(t0)
    ci = np.empty((k, 2), dtype=np.float64)

    z_lo = sp_stats.norm.ppf(alpha / 2.0)
    z_hi = sp_stats.norm.ppf(1.0 - alpha / 2.0)

    for j in range(k):
        center = 2.0 * t0[j] - np.mean(t[:, j])
        if var_t0 is not None and k == 1:
            se = np.sqrt(var_t0)
        else:
            se = np.std(t[:, j], ddof=1)
        ci[j, 0] = center + z_lo * se
        ci[j, 1] = center + z_hi * se

    return ci


def _ci_basic(t0: NDArray, t: NDArray, alpha: float) -> NDArray:
    """
    Basic (pivotal) bootstrap CI.

    CI = [2*t0 - Q(1-alpha/2), 2*t0 - Q(alpha/2)]

    Note: upper quantile of bootstrap gives lower bound.
    """
    k = len(t0)
    ci = np.empty((k, 2), dtype=np.float64)

    for j in range(k):
        q_lo = np.quantile(t[:, j], alpha / 2.0)
        q_hi = np.quantile(t[:, j], 1.0 - alpha / 2.0)
        ci[j, 0] = 2.0 * t0[j] - q_hi
        ci[j, 1] = 2.0 * t0[j] - q_lo

    return ci


def _ci_percentile(t: NDArray, alpha: float) -> NDArray:
    """
    Percentile bootstrap CI.

    CI = [Q(alpha/2), Q(1-alpha/2)]
    """
    k = t.shape[1]
    ci = np.empty((k, 2), dtype=np.float64)

    for j in range(k):
        ci[j, 0] = np.quantile(t[:, j], alpha / 2.0)
        ci[j, 1] = np.quantile(t[:, j], 1.0 - alpha / 2.0)

    return ci


def _ci_bca(boot_out: 'BootstrapSolution', alpha: float) -> NDArray:
    """
    BCa (bias-corrected and accelerated) CI.

    Uses jackknife influence values for the acceleration parameter.

    Steps:
    1. z0 = Phi^{-1}(proportion of t* < t0)
    2. a = sum(L^3) / (6 * sum(L^2)^1.5) from jackknife
    3. Adjusted quantile levels
    4. CI from adjusted percentiles
    """
    from pystatistics.montecarlo._influence import jackknife_influence

    t0 = boot_out.t0
    t = boot_out.t
    k = len(t0)
    R = t.shape[0]
    ci = np.empty((k, 2), dtype=np.float64)

    for j in range(k):
        t_j = t[:, j]
        t0_j = t0[j]

        # Bias correction factor z0
        prop_below = np.sum(t_j < t0_j) / R
        # Clamp to avoid infinite z0
        prop_below = np.clip(prop_below, 1.0 / (2.0 * R), 1.0 - 1.0 / (2.0 * R))
        z0 = sp_stats.norm.ppf(prop_below)

        # Acceleration parameter from jackknife
        L = jackknife_influence(boot_out, j)
        L_sq_sum = np.sum(L ** 2)
        if L_sq_sum > 0:
            a = np.sum(L ** 3) / (6.0 * L_sq_sum ** 1.5)
        else:
            a = 0.0

        # Adjusted quantile levels
        z_lo = sp_stats.norm.ppf(alpha / 2.0)
        z_hi = sp_stats.norm.ppf(1.0 - alpha / 2.0)

        # BCa formula
        def _adj_quantile(z_alpha):
            numer = z0 + z_alpha
            denom = 1.0 - a * numer
            if abs(denom) < 1e-15:
                return 0.5  # degenerate case
            return sp_stats.norm.cdf(z0 + numer / denom)

        alpha1 = _adj_quantile(z_lo)
        alpha2 = _adj_quantile(z_hi)

        # Clamp to valid range
        alpha1 = np.clip(alpha1, 0.5 / R, 1.0 - 0.5 / R)
        alpha2 = np.clip(alpha2, 0.5 / R, 1.0 - 0.5 / R)

        ci[j, 0] = np.quantile(t_j, alpha1)
        ci[j, 1] = np.quantile(t_j, alpha2)

    return ci


def _ci_studentized(
    t0: NDArray,
    t: NDArray,
    alpha: float,
    var_t0: float | None,
    var_t: NDArray,
    index: int,
) -> NDArray:
    """
    Studentized (bootstrap-t) CI.

    For each replicate: z* = (t* - t0) / se*
    CI = [t0 - q*(1-alpha/2) * se_hat, t0 - q*(alpha/2) * se_hat]

    Requires variance estimates for each replicate.
    """
    k = len(t0)
    ci = np.empty((k, 2), dtype=np.float64)

    # Studentized CI computed for the specified index
    t_j = t[:, index]
    t0_j = t0[index]

    se_star = np.sqrt(var_t)

    # Compute bootstrap t-statistics
    # Avoid division by zero
    valid = se_star > 0
    z_star = np.full(len(t_j), np.nan)
    z_star[valid] = (t_j[valid] - t0_j) / se_star[valid]
    z_star = z_star[~np.isnan(z_star)]

    if len(z_star) == 0:
        ci[:, :] = np.nan
        return ci

    q_lo = np.quantile(z_star, alpha / 2.0)
    q_hi = np.quantile(z_star, 1.0 - alpha / 2.0)

    if var_t0 is not None:
        se_hat = np.sqrt(var_t0)
    else:
        se_hat = np.std(t_j, ddof=1)

    # Note the reversal: upper quantile gives lower bound
    ci[index, 0] = t0_j - q_hi * se_hat
    ci[index, 1] = t0_j - q_lo * se_hat

    # Fill other indices with NaN (studentized is per-index)
    for j in range(k):
        if j != index:
            ci[j, :] = np.nan

    return ci
