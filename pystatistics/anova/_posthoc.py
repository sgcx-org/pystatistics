"""
Post-hoc pairwise comparison tests.

Tukey HSD:
    Uses the studentized range distribution (scipy.stats.studentized_range)
    to compute simultaneous confidence intervals and adjusted p-values.

Bonferroni:
    Individual t-tests with Bonferroni correction (multiply p by k*(k-1)/2).

Dunnett:
    Many-to-one comparisons against a control group.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from pystatistics.anova._common import PostHocComparison, PostHocParams


def tukey_hsd(
    y: NDArray,
    group: NDArray,
    mse: float,
    df_error: int,
    *,
    factor: str = 'group',
    conf_level: float = 0.95,
) -> PostHocParams:
    """
    Tukey's Honestly Significant Difference test.

    Uses the studentized range distribution for simultaneous inference.
    Matches R's TukeyHSD(aov(...)).

    Args:
        y: 1D response array
        group: 1D group labels
        mse: Mean square error from ANOVA
        df_error: Error degrees of freedom from ANOVA
        factor: Name of the factor being compared
        conf_level: Confidence level for intervals

    Returns:
        PostHocParams with all pairwise comparisons
    """
    group_str = np.array([str(v) for v in group])
    levels = sorted(set(group_str))
    k = len(levels)

    # Group means and sizes
    means: dict[str, float] = {}
    sizes: dict[str, int] = {}
    for level in levels:
        mask = group_str == level
        means[level] = float(np.mean(y[mask]))
        sizes[level] = int(np.sum(mask))

    comparisons: list[PostHocComparison] = []

    for i in range(k):
        for j in range(i + 1, k):
            g1, g2 = levels[i], levels[j]
            diff = means[g2] - means[g1]
            n1, n2 = sizes[g1], sizes[g2]

            se = np.sqrt(mse * (1.0 / n1 + 1.0 / n2) / 2.0)

            # q statistic: diff / (se_tukey) where se_tukey = sqrt(MSE / n_harmonic)
            # But we need q = |diff| / sqrt(MSE * 0.5 * (1/n1 + 1/n2))
            # The studentized range distribution uses q = range / s
            # For Tukey: q = |diff| / sqrt(MSE / 2 * (1/n1 + 1/n2))
            q_stat = np.abs(diff) / se

            # p-value from studentized range distribution
            p_val = float(sp_stats.studentized_range.sf(q_stat, k, df_error))

            # Confidence interval
            q_crit = float(sp_stats.studentized_range.ppf(conf_level, k, df_error))
            margin = q_crit * se

            comparisons.append(PostHocComparison(
                group1=g1,
                group2=g2,
                diff=diff,
                ci_lower=diff - margin,
                ci_upper=diff + margin,
                p_value=min(p_val, 1.0),
                se=se,
            ))

    return PostHocParams(
        method='tukey',
        comparisons=tuple(comparisons),
        conf_level=conf_level,
        factor=factor,
        mse=mse,
        df_error=df_error,
    )


def bonferroni_pairwise(
    y: NDArray,
    group: NDArray,
    mse: float,
    df_error: int,
    *,
    factor: str = 'group',
    conf_level: float = 0.95,
) -> PostHocParams:
    """
    Pairwise t-tests with Bonferroni correction.

    Each comparison uses pooled MSE from the ANOVA. P-values are multiplied
    by the number of comparisons k*(k-1)/2.

    Args:
        y: 1D response array
        group: 1D group labels
        mse: Mean square error from ANOVA
        df_error: Error degrees of freedom from ANOVA
        factor: Name of the factor being compared
        conf_level: Confidence level for intervals

    Returns:
        PostHocParams with all pairwise comparisons
    """
    group_str = np.array([str(v) for v in group])
    levels = sorted(set(group_str))
    k = len(levels)
    n_comparisons = k * (k - 1) // 2

    # Bonferroni-adjusted alpha
    alpha = 1.0 - conf_level
    alpha_adj = alpha / n_comparisons

    means: dict[str, float] = {}
    sizes: dict[str, int] = {}
    for level in levels:
        mask = group_str == level
        means[level] = float(np.mean(y[mask]))
        sizes[level] = int(np.sum(mask))

    comparisons: list[PostHocComparison] = []

    for i in range(k):
        for j in range(i + 1, k):
            g1, g2 = levels[i], levels[j]
            diff = means[g2] - means[g1]
            n1, n2 = sizes[g1], sizes[g2]

            se = np.sqrt(mse * (1.0 / n1 + 1.0 / n2))
            t_stat = diff / se

            # Two-sided p-value, Bonferroni adjusted
            p_raw = 2.0 * float(sp_stats.t.sf(np.abs(t_stat), df_error))
            p_adj = min(p_raw * n_comparisons, 1.0)

            # Bonferroni-adjusted CI
            t_crit = float(sp_stats.t.ppf(1.0 - alpha_adj / 2.0, df_error))
            margin = t_crit * se

            comparisons.append(PostHocComparison(
                group1=g1,
                group2=g2,
                diff=diff,
                ci_lower=diff - margin,
                ci_upper=diff + margin,
                p_value=p_adj,
                se=se,
            ))

    return PostHocParams(
        method='bonferroni',
        comparisons=tuple(comparisons),
        conf_level=conf_level,
        factor=factor,
        mse=mse,
        df_error=df_error,
    )


def dunnett_test(
    y: NDArray,
    group: NDArray,
    mse: float,
    df_error: int,
    control: str,
    *,
    factor: str = 'group',
    conf_level: float = 0.95,
) -> PostHocParams:
    """
    Dunnett's test: compare each treatment group to a control.

    Uses scipy.stats.dunnett when available (scipy >= 1.12), otherwise
    falls back to Bonferroni-corrected t-tests as a conservative approximation.

    Args:
        y: 1D response array
        group: 1D group labels
        mse: Mean square error from ANOVA
        df_error: Error degrees of freedom from ANOVA
        control: Name of the control group
        factor: Name of the factor being compared
        conf_level: Confidence level for intervals

    Returns:
        PostHocParams with treatment-vs-control comparisons
    """
    group_str = np.array([str(v) for v in group])
    levels = sorted(set(group_str))

    if control not in levels:
        raise ValueError(
            f"Control group {control!r} not found in levels: {levels}"
        )

    treatment_levels = [lev for lev in levels if lev != control]
    n_comparisons = len(treatment_levels)

    means: dict[str, float] = {}
    sizes: dict[str, int] = {}
    samples: dict[str, NDArray] = {}
    for level in levels:
        mask = group_str == level
        samples[level] = y[mask]
        means[level] = float(np.mean(y[mask]))
        sizes[level] = int(np.sum(mask))

    comparisons: list[PostHocComparison] = []

    # Try scipy.stats.dunnett (available in scipy >= 1.12)
    try:
        treatment_samples = [samples[lev] for lev in treatment_levels]
        control_sample = samples[control]
        result = sp_stats.dunnett(*treatment_samples, control=control_sample)

        for idx, lev in enumerate(treatment_levels):
            diff = means[lev] - means[control]
            n_ctrl, n_trt = sizes[control], sizes[lev]
            se = np.sqrt(mse * (1.0 / n_ctrl + 1.0 / n_trt))

            ci = result.confidence_interval(confidence_level=conf_level)
            comparisons.append(PostHocComparison(
                group1=control,
                group2=lev,
                diff=diff,
                ci_lower=float(ci.low[idx]),
                ci_upper=float(ci.high[idx]),
                p_value=float(result.pvalue[idx]),
                se=se,
            ))
    except (AttributeError, TypeError):
        # Fallback: Bonferroni-corrected t-tests
        alpha = 1.0 - conf_level
        alpha_adj = alpha / n_comparisons

        for lev in treatment_levels:
            diff = means[lev] - means[control]
            n_ctrl, n_trt = sizes[control], sizes[lev]
            se = np.sqrt(mse * (1.0 / n_ctrl + 1.0 / n_trt))
            t_stat = diff / se

            p_raw = 2.0 * float(sp_stats.t.sf(np.abs(t_stat), df_error))
            p_adj = min(p_raw * n_comparisons, 1.0)

            t_crit = float(sp_stats.t.ppf(1.0 - alpha_adj / 2.0, df_error))
            margin = t_crit * se

            comparisons.append(PostHocComparison(
                group1=control,
                group2=lev,
                diff=diff,
                ci_lower=diff - margin,
                ci_upper=diff + margin,
                p_value=p_adj,
                se=se,
            ))

    return PostHocParams(
        method='dunnett',
        comparisons=tuple(comparisons),
        conf_level=conf_level,
        factor=factor,
        mse=mse,
        df_error=df_error,
    )
