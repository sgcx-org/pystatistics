"""
ANOVA solver dispatch.

Public API:
    anova_oneway(y, group, ...) -> AnovaSolution
    anova(y, factors, ...) -> AnovaSolution
    anova_rm(y, subject, within, ...) -> AnovaRMSolution
    anova_posthoc(result, ...) -> PostHocSolution
    levene_test(y, group, ...) -> LeveneSolution
"""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result
from pystatistics.core.exceptions import ValidationError
from pystatistics.anova._common import (
    AnovaParams,
    AnovaRMParams,
    AnovaTableRow,
    LeveneParams,
    PostHocParams,
)
from pystatistics.anova._contrasts import build_model_matrix
from pystatistics.anova._ss import compute_ss_type1, compute_ss_type2, compute_ss_type3
from pystatistics.anova._levene import levene_test_impl
from pystatistics.anova._posthoc import tukey_hsd, bonferroni_pairwise, dunnett_test
from pystatistics.anova._repeated import repeated_measures_anova
from pystatistics.anova.design import AnovaDesign
from pystatistics.anova.solution import (
    AnovaSolution,
    AnovaRMSolution,
    LeveneSolution,
    PostHocSolution,
)


def anova_oneway(
    y: Any,
    group: Any,
    *,
    ss_type: int = 1,
) -> AnovaSolution:
    """
    One-way Analysis of Variance.

    Tests whether the means of two or more groups are equal.

    Args:
        y: Response variable (1D numeric array-like)
        group: Group labels (1D array-like, same length as y)
        ss_type: Type of sums of squares (1, 2, or 3). Default 1.
            For one-way ANOVA, all three types give identical results.

    Returns:
        AnovaSolution with ANOVA table, effect sizes, and group means

    Examples:
        >>> result = anova_oneway(y, group)
        >>> print(result.summary())
        >>> result.table[0].f_value  # F statistic for the group effect
        >>> result.eta_squared       # effect sizes
    """
    t0 = time.perf_counter()

    design = AnovaDesign.for_oneway(y, group)
    y_arr = design.y
    group_arr = design.factors['group']

    # Build model matrix
    coding = 'deviation' if ss_type == 3 else 'treatment'
    mm = build_model_matrix(
        {'group': group_arr},
        coding=coding,
        include_intercept=True,
        interactions=None,
    )

    # Compute SS
    rows = _compute_ss(y_arr, mm, ss_type)

    # Group means
    levels = sorted(set(str(v) for v in group_arr))
    group_means_dict: dict[str, float] = {}
    for level in levels:
        mask = group_arr == level
        group_means_dict[level] = float(np.mean(y_arr[mask]))

    # Effect sizes
    residual_row = rows[-1]
    total_ss = sum(row.sum_sq for row in rows)
    eta_sq: dict[str, float] = {}
    partial_eta_sq: dict[str, float] = {}
    for row in rows:
        if row.term != 'Residuals':
            eta_sq[row.term] = row.sum_sq / total_ss if total_ss > 0 else 0.0
            partial_eta_sq[row.term] = (
                row.sum_sq / (row.sum_sq + residual_row.sum_sq)
                if (row.sum_sq + residual_row.sum_sq) > 0 else 0.0
            )

    elapsed = time.perf_counter() - t0

    params = AnovaParams(
        table=tuple(rows),
        ss_type=ss_type,
        n_obs=design.n,
        n_groups={'group': len(levels)},
        grand_mean=float(np.mean(y_arr)),
        group_means={'group': group_means_dict},
        residual_df=residual_row.df,
        residual_ss=residual_row.sum_sq,
        residual_ms=residual_row.mean_sq,
        eta_squared=eta_sq,
        partial_eta_squared=partial_eta_sq,
    )

    result = Result(
        params=params,
        info={
            'ss_type': ss_type,
            'design_type': 'oneway',
            '_y': y_arr,
            '_factor_group': group_arr,
        },
        timing={'total_seconds': elapsed},
        backend_name='cpu',
    )

    return AnovaSolution(_result=result)


def anova(
    y: Any,
    factors: dict[str, Any],
    *,
    covariates: dict[str, Any] | None = None,
    ss_type: int = 2,
    interactions: bool = True,
) -> AnovaSolution:
    """
    Factorial ANOVA or ANCOVA.

    Tests main effects and interactions of two or more factors, with optional
    continuous covariates (ANCOVA).

    Args:
        y: Response variable (1D numeric array-like)
        factors: {name: 1D array of group labels}
        covariates: {name: 1D numeric array} or None
        ss_type: Type of sums of squares (1, 2, or 3). Default 2.
            Type I: sequential (order-dependent)
            Type II: marginal, respects marginality (R's car::Anova default)
            Type III: each term last (requires deviation coding)
        interactions: Whether to include interaction terms. Default True.

    Returns:
        AnovaSolution with ANOVA table, effect sizes, and group means

    Examples:
        >>> result = anova(y, {'A': factor_a, 'B': factor_b})
        >>> print(result.summary())
        >>> result = anova(y, {'treatment': tx}, covariates={'age': age}, ss_type=2)
    """
    t0 = time.perf_counter()

    design = AnovaDesign.for_factorial(y, factors, covariates=covariates)
    y_arr = design.y

    # Build model matrix
    coding = 'deviation' if ss_type == 3 else 'treatment'
    interaction_pairs = None if not interactions else None  # None = auto all pairs

    # Prepare validated covariates as NDArrays
    cov_arrays = None
    if design.covariates is not None:
        cov_arrays = {name: arr for name, arr in design.covariates.items()}

    mm = build_model_matrix(
        design.factors,
        covariates=cov_arrays,
        coding=coding,
        include_intercept=True,
        interactions=interaction_pairs if interactions else [],
    )

    # Compute SS
    rows = _compute_ss(y_arr, mm, ss_type)

    # Group means
    group_means: dict[str, dict[str, float]] = {}
    n_groups: dict[str, int] = {}
    for name, fac_arr in design.factors.items():
        levels = sorted(set(str(v) for v in fac_arr))
        n_groups[name] = len(levels)
        means_dict: dict[str, float] = {}
        for level in levels:
            mask = fac_arr == level
            means_dict[level] = float(np.mean(y_arr[mask]))
        group_means[name] = means_dict

    # Effect sizes
    residual_row = rows[-1]
    total_ss = sum(row.sum_sq for row in rows)
    eta_sq: dict[str, float] = {}
    partial_eta_sq: dict[str, float] = {}
    for row in rows:
        if row.term != 'Residuals':
            eta_sq[row.term] = row.sum_sq / total_ss if total_ss > 0 else 0.0
            partial_eta_sq[row.term] = (
                row.sum_sq / (row.sum_sq + residual_row.sum_sq)
                if (row.sum_sq + residual_row.sum_sq) > 0 else 0.0
            )

    elapsed = time.perf_counter() - t0

    params = AnovaParams(
        table=tuple(rows),
        ss_type=ss_type,
        n_obs=design.n,
        n_groups=n_groups,
        grand_mean=float(np.mean(y_arr)),
        group_means=group_means,
        residual_df=residual_row.df,
        residual_ss=residual_row.sum_sq,
        residual_ms=residual_row.mean_sq,
        eta_squared=eta_sq,
        partial_eta_squared=partial_eta_sq,
    )

    result = Result(
        params=params,
        info={
            'ss_type': ss_type,
            'design_type': 'factorial',
            '_y': y_arr,
            **{f'_factor_{name}': arr for name, arr in design.factors.items()},
        },
        timing={'total_seconds': elapsed},
        backend_name='cpu',
    )

    return AnovaSolution(_result=result)


def anova_rm(
    y: Any,
    subject: Any,
    within: dict[str, Any],
    *,
    between: dict[str, Any] | None = None,
    correction: str = 'auto',
) -> AnovaRMSolution:
    """
    Repeated-measures ANOVA.

    Tests within-subjects effects with optional between-subjects factors
    (mixed design). Includes Mauchly's sphericity test and GG/HF corrections.

    Args:
        y: Response variable (1D, long format)
        subject: Subject identifiers (1D)
        within: {factor_name: 1D condition labels}
        between: {factor_name: 1D group labels} or None
        correction: Sphericity correction:
            'none': no correction
            'gg': Greenhouse-Geisser
            'hf': Huynh-Feldt
            'auto': GG if Mauchly p < 0.05, else none (default)

    Returns:
        AnovaRMSolution with ANOVA table, sphericity, corrected p-values

    Examples:
        >>> result = anova_rm(y, subject=subj, within={'condition': cond})
        >>> print(result.summary())
        >>> result.sphericity[0].gg_epsilon  # GG correction factor
    """
    t0 = time.perf_counter()

    design = AnovaDesign.for_repeated_measures(
        y, subject, within, between=between,
    )

    # Extract validated within factors from design
    within_validated = {}
    for name in within.keys():
        within_validated[name] = design.factors[name]

    between_validated = None
    if between is not None:
        between_validated = {}
        for name in between.keys():
            between_validated[name] = design.factors[name]

    rm_params = repeated_measures_anova(
        design.y,
        design.subject,
        within_validated,
        between=between_validated,
        correction=correction,
    )

    elapsed = time.perf_counter() - t0

    result = Result(
        params=rm_params,
        info={'design_type': 'rm', 'correction': correction},
        timing={'total_seconds': elapsed},
        backend_name='cpu',
    )

    return AnovaRMSolution(_result=result)


def anova_posthoc(
    anova_result: AnovaSolution,
    *,
    method: str = 'tukey',
    factor: str | None = None,
    control: str | None = None,
    conf_level: float = 0.95,
) -> PostHocSolution:
    """
    Post-hoc pairwise comparisons following ANOVA.

    Args:
        anova_result: Result from anova_oneway() or anova()
        method: 'tukey' (default), 'bonferroni', or 'dunnett'
        factor: Which factor to compare (required for factorial, auto for oneway)
        control: Control group name (required for dunnett)
        conf_level: Confidence level (default 0.95)

    Returns:
        PostHocSolution with comparison table and adjusted p-values

    Examples:
        >>> anova_result = anova_oneway(y, group)
        >>> posthoc = anova_posthoc(anova_result, method='tukey')
        >>> print(posthoc.summary())
    """
    t0 = time.perf_counter()

    params = anova_result._result.params
    mse = params.residual_ms
    df_error = params.residual_df

    # Determine factor
    if factor is None:
        if len(params.group_means) == 1:
            factor = list(params.group_means.keys())[0]
        else:
            raise ValueError(
                "Multiple factors present. Specify factor= explicitly. "
                f"Available: {list(params.group_means.keys())}"
            )

    if factor not in params.group_means:
        raise ValueError(
            f"Factor {factor!r} not found. "
            f"Available: {list(params.group_means.keys())}"
        )

    # We need the raw data — reconstruct from the design
    # Since AnovaSolution doesn't store raw data, we need the user to
    # pass the original arrays. For now, extract from the stored info.
    # WORKAROUND: Store y and group arrays in info dict during anova_oneway/anova
    y_arr = anova_result._result.info.get('_y')
    group_arr = anova_result._result.info.get(f'_factor_{factor}')

    if y_arr is None or group_arr is None:
        raise ValueError(
            "Post-hoc tests require original data arrays. "
            "This should have been stored in the ANOVA result."
        )

    if method == 'tukey':
        posthoc_params = tukey_hsd(
            y_arr, group_arr, mse, df_error,
            factor=factor, conf_level=conf_level,
        )
    elif method == 'bonferroni':
        posthoc_params = bonferroni_pairwise(
            y_arr, group_arr, mse, df_error,
            factor=factor, conf_level=conf_level,
        )
    elif method == 'dunnett':
        if control is None:
            raise ValueError("Dunnett test requires control= parameter")
        posthoc_params = dunnett_test(
            y_arr, group_arr, mse, df_error, control,
            factor=factor, conf_level=conf_level,
        )
    else:
        raise ValueError(
            f"Unknown method {method!r}. Use 'tukey', 'bonferroni', or 'dunnett'."
        )

    elapsed = time.perf_counter() - t0

    result = Result(
        params=posthoc_params,
        info={'method': method, 'factor': factor},
        timing={'total_seconds': elapsed},
        backend_name='cpu',
    )

    return PostHocSolution(_result=result)


def levene_test(
    y: Any,
    group: Any,
    *,
    center: str = 'median',
) -> LeveneSolution:
    """
    Levene's test for homogeneity of variances.

    Tests the null hypothesis that all groups have equal variances.
    With center='median' (default), this is the Brown-Forsythe variant
    which is more robust to non-normality.

    Args:
        y: Response variable (1D numeric array-like)
        group: Group labels (1D array-like, same length as y)
        center: 'median' (Brown-Forsythe, default) or 'mean' (original Levene)

    Returns:
        LeveneSolution with F statistic, p-value, and group variances

    Examples:
        >>> result = levene_test(y, group)
        >>> result.p_value > 0.05  # Can't reject equal variances
        >>> print(result.summary())
    """
    t0 = time.perf_counter()

    design = AnovaDesign.for_oneway(y, group)
    y_arr = design.y
    group_arr = design.factors['group']

    levene_params = levene_test_impl(y_arr, group_arr, center=center)

    elapsed = time.perf_counter() - t0

    result = Result(
        params=levene_params,
        info={'center': center},
        timing={'total_seconds': elapsed},
        backend_name='cpu',
    )

    return LeveneSolution(_result=result)


# =====================================================================
# Internal helpers
# =====================================================================


def _compute_ss(
    y: NDArray,
    mm,
    ss_type: int,
) -> list[AnovaTableRow]:
    """Dispatch to the correct SS computation."""
    if ss_type == 1:
        return compute_ss_type1(y, mm)
    elif ss_type == 2:
        return compute_ss_type2(y, mm)
    elif ss_type == 3:
        return compute_ss_type3(y, mm)
    else:
        raise ValueError(f"ss_type must be 1, 2, or 3, got {ss_type}")
