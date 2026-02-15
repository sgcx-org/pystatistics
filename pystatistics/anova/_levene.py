"""
Levene's test for homogeneity of variances.

Algorithm: Transform y to |y_i - center(group_j)|, then run one-way ANOVA
on the transformed values. center='median' gives the Brown-Forsythe variant
(robust, R's default). center='mean' gives the original Levene test.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from pystatistics.anova._common import LeveneParams


def levene_test_impl(
    y: NDArray,
    group: NDArray,
    *,
    center: str = 'median',
) -> LeveneParams:
    """
    Compute Levene's test (or Brown-Forsythe variant).

    Args:
        y: 1D response array
        group: 1D group labels (same length as y)
        center: 'median' (Brown-Forsythe, default) or 'mean' (original Levene)

    Returns:
        LeveneParams with F statistic, p-value, and degrees of freedom
    """
    if center not in ('mean', 'median'):
        raise ValueError(f"center must be 'mean' or 'median', got {center!r}")

    group_str = np.array([str(v) for v in group])
    levels = sorted(set(group_str))
    k = len(levels)
    n = len(y)

    # Compute centers and transformed values
    center_fn = np.mean if center == 'mean' else np.median
    z = np.empty(n, dtype=np.float64)
    group_vars: dict[str, float] = {}

    for level in levels:
        mask = group_str == level
        y_group = y[mask]
        c = center_fn(y_group)
        z[mask] = np.abs(y_group - c)
        group_vars[level] = float(np.var(y_group, ddof=1))

    # One-way ANOVA on the transformed values (manual, to avoid circular import)
    z_grand_mean = np.mean(z)
    ss_between = 0.0
    ss_within = 0.0

    for level in levels:
        mask = group_str == level
        z_group = z[mask]
        n_j = len(z_group)
        z_mean_j = np.mean(z_group)
        ss_between += n_j * (z_mean_j - z_grand_mean) ** 2
        ss_within += np.sum((z_group - z_mean_j) ** 2)

    df_between = k - 1
    df_within = n - k

    if df_between <= 0 or df_within <= 0 or ss_within == 0:
        f_val = 0.0
        p_val = 1.0
    else:
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        f_val = ms_between / ms_within
        p_val = float(sp_stats.f.sf(f_val, df_between, df_within))

    return LeveneParams(
        f_value=f_val,
        p_value=p_val,
        df_between=df_between,
        df_within=df_within,
        center=center,
        group_vars=group_vars,
    )
