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
from pystatistics.core.exceptions import ValidationError

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from pystatistics.anova._common import PostHocComparison, PostHocParams


def _pair_cohens_d(
    y: NDArray, group_str: NDArray, g1: str, g2: str,
) -> float:
    """Pairwise Cohen's d for groups ``g1`` vs ``g2`` using the pooled SD.

    Signed as ``(mean(g2) - mean(g1)) / s_pooled`` so it carries the same sign
    as the reported ``diff`` (which is also ``mean(g2) - mean(g1)``), with
    ``s_pooled = sqrt(((n1-1) s1^2 + (n2-1) s2^2) / (n1 + n2 - 2))``. This
    matches R's ``effectsize::cohens_d`` (pooled SD, its default).
    """
    a = y[group_str == g1]
    b = y[group_str == g2]
    n1, n2 = len(a), len(b)
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    s_pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return float((np.mean(b) - np.mean(a)) / s_pooled)


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
                cohens_d=_pair_cohens_d(y, group_str, g1, g2),
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
                cohens_d=_pair_cohens_d(y, group_str, g1, g2),
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
        raise ValidationError(
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
                cohens_d=_pair_cohens_d(y, group_str, control, lev),
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
                cohens_d=_pair_cohens_d(y, group_str, control, lev),
            ))

    return PostHocParams(
        method='dunnett',
        comparisons=tuple(comparisons),
        conf_level=conf_level,
        factor=factor,
        mse=mse,
        df_error=df_error,
    )


def games_howell(
    y: NDArray,
    group: NDArray,
    *,
    factor: str = 'group',
    conf_level: float = 0.95,
) -> PostHocParams:
    """
    Games-Howell post-hoc test (unequal variances, unequal n).

    Unlike Tukey HSD (which pools a single MSE and assumes homogeneous
    variances), Games-Howell uses each pair's own group variances and a
    Welch-Satterthwaite degrees-of-freedom, so it is the appropriate all-pairs
    procedure when variances differ (e.g. a significant Levene test). Inference
    uses the studentized-range distribution, matching the canonical formula
    (equivalent to ``rstatix::games_howell_test`` /
    ``PMCMRplus::gamesHowellTest``):

        se_ij  = sqrt( 0.5 * (s_i^2/n_i + s_j^2/n_j) )
        df_ij  = (s_i^2/n_i + s_j^2/n_j)^2
                 / ( (s_i^2/n_i)^2/(n_i-1) + (s_j^2/n_j)^2/(n_j-1) )
        q_ij   = |mean_j - mean_i| / se_ij
        p_ij   = P(Q > q_ij; k, df_ij)          # studentized range, k groups
        CI     = diff ± q_{conf, k, df_ij} * se_ij

    Args:
        y: 1D response array.
        group: 1D group labels.
        factor: Name of the factor being compared.
        conf_level: Confidence level for the simultaneous intervals.

    Returns:
        PostHocParams with all pairwise comparisons. ``mse`` is NaN and
        ``df_error`` is -1 because Games-Howell uses a per-pair variance and
        degrees-of-freedom rather than a single pooled error term.
    """
    group_str = np.array([str(v) for v in group])
    levels = sorted(set(group_str))
    k = len(levels)

    means: dict[str, float] = {}
    variances: dict[str, float] = {}
    sizes: dict[str, int] = {}
    for level in levels:
        vals = y[group_str == level]
        if len(vals) < 2:
            raise ValidationError(
                f"Games-Howell requires >= 2 observations per group; "
                f"group {level!r} has {len(vals)}."
            )
        means[level] = float(np.mean(vals))
        variances[level] = float(np.var(vals, ddof=1))
        sizes[level] = int(len(vals))

    comparisons: list[PostHocComparison] = []
    for i in range(k):
        for j in range(i + 1, k):
            g1, g2 = levels[i], levels[j]
            diff = means[g2] - means[g1]
            vn1 = variances[g1] / sizes[g1]
            vn2 = variances[g2] / sizes[g2]

            se = np.sqrt(0.5 * (vn1 + vn2))
            df = (vn1 + vn2) ** 2 / (
                vn1 ** 2 / (sizes[g1] - 1) + vn2 ** 2 / (sizes[g2] - 1)
            )
            q_stat = np.abs(diff) / se
            p_val = float(sp_stats.studentized_range.sf(q_stat, k, df))
            q_crit = float(sp_stats.studentized_range.ppf(conf_level, k, df))
            margin = q_crit * se

            comparisons.append(PostHocComparison(
                group1=g1,
                group2=g2,
                diff=diff,
                ci_lower=diff - margin,
                ci_upper=diff + margin,
                p_value=min(p_val, 1.0),
                se=se,
                cohens_d=_pair_cohens_d(y, group_str, g1, g2),
            ))

    return PostHocParams(
        method='games-howell',
        comparisons=tuple(comparisons),
        conf_level=conf_level,
        factor=factor,
        mse=float('nan'),
        df_error=-1,
    )
