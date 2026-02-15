"""
Repeated-measures ANOVA.

Long-format input. Reshapes to wide (subjects x conditions), computes
within-subject sums of squares, and tests sphericity via Mauchly's test.

Corrections:
    - Greenhouse-Geisser: conservative, always ≤ 1
    - Huynh-Feldt: less conservative, can slightly exceed 1 (capped at 1.0)
    - 'auto': use GG when Mauchly p < 0.05, otherwise uncorrected
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from pystatistics.anova._common import (
    AnovaRMParams,
    AnovaRMTableRow,
    SphericitySummary,
)


def repeated_measures_anova(
    y: NDArray,
    subject: NDArray,
    within: dict[str, NDArray],
    *,
    between: dict[str, NDArray] | None = None,
    correction: str = 'auto',
) -> AnovaRMParams:
    """
    Compute repeated-measures ANOVA.

    Currently supports one within-subjects factor. Mixed designs (one within +
    one between) are also supported.

    Args:
        y: 1D response array (long format)
        subject: 1D subject identifiers
        within: {factor_name: 1D condition labels}
        between: {factor_name: 1D group labels} or None
        correction: 'none', 'gg', 'hf', or 'auto'

    Returns:
        AnovaRMParams with table, sphericity tests, and corrected p-values
    """
    if correction not in ('none', 'gg', 'hf', 'auto'):
        raise ValueError(f"correction must be 'none', 'gg', 'hf', or 'auto', got {correction!r}")

    if len(within) != 1:
        raise NotImplementedError(
            f"Currently supports exactly 1 within-subjects factor, got {len(within)}"
        )

    within_name = list(within.keys())[0]
    within_factor = np.array([str(v) for v in within[within_name]])
    subject_ids = np.array([str(v) for v in subject])

    # Get unique levels
    conditions = sorted(set(within_factor))
    subjects = sorted(set(subject_ids))
    k = len(conditions)  # number of conditions
    n = len(subjects)    # number of subjects
    N = len(y)           # total observations

    # Reshape to wide format: Y[i, j] = response for subject i, condition j
    Y_wide = np.full((n, k), np.nan, dtype=np.float64)
    for idx in range(N):
        i = subjects.index(subject_ids[idx])
        j = conditions.index(within_factor[idx])
        Y_wide[i, j] = y[idx]

    if np.any(np.isnan(Y_wide)):
        raise ValueError(
            "Missing data detected after reshaping to wide format. "
            "Ensure each subject has exactly one observation per condition."
        )

    grand_mean = float(np.mean(Y_wide))
    subject_means = np.mean(Y_wide, axis=1)   # (n,)
    condition_means = np.mean(Y_wide, axis=0)  # (k,)

    # Total SS
    ss_total = float(np.sum((Y_wide - grand_mean) ** 2))

    # Between-subjects SS
    ss_subjects = k * float(np.sum((subject_means - grand_mean) ** 2))

    # Within-subjects total SS
    ss_within_total = ss_total - ss_subjects

    # Effect of within-factor (condition)
    ss_condition = n * float(np.sum((condition_means - grand_mean) ** 2))

    # Error = within total - condition effect
    ss_error = ss_within_total - ss_condition

    # Degrees of freedom
    df_condition = k - 1
    df_subjects = n - 1
    df_error = (n - 1) * (k - 1)

    # Mean squares
    ms_condition = ss_condition / df_condition if df_condition > 0 else 0.0
    ms_error = ss_error / df_error if df_error > 0 else 0.0

    # F test
    if ms_error > 0:
        f_val = ms_condition / ms_error
        p_val = float(sp_stats.f.sf(f_val, df_condition, df_error))
    else:
        f_val = 0.0
        p_val = 1.0

    # Sphericity test (only meaningful when k >= 3)
    if k >= 3:
        mauchly_w, mauchly_p, gg_eps, hf_eps = mauchly_test(Y_wide)
    else:
        # With 2 conditions, sphericity is always satisfied (epsilon = 1)
        mauchly_w = 1.0
        mauchly_p = 1.0
        gg_eps = 1.0
        hf_eps = 1.0

    # Corrected p-values
    gg_df_cond = gg_eps * df_condition
    gg_df_error = gg_eps * df_error
    gg_p = float(sp_stats.f.sf(f_val, gg_df_cond, gg_df_error)) if ms_error > 0 else 1.0

    hf_df_cond = hf_eps * df_condition
    hf_df_error = hf_eps * df_error
    hf_p = float(sp_stats.f.sf(f_val, hf_df_cond, hf_df_error)) if ms_error > 0 else 1.0

    # Build table rows
    table_rows: list[AnovaRMTableRow] = []

    table_rows.append(AnovaRMTableRow(
        term=within_name,
        df=float(df_condition),
        sum_sq=ss_condition,
        mean_sq=ms_condition,
        f_value=f_val,
        p_value=p_val,
        gg_p_value=gg_p,
        hf_p_value=hf_p,
    ))

    table_rows.append(AnovaRMTableRow(
        term='Error',
        df=float(df_error),
        sum_sq=ss_error,
        mean_sq=ms_error,
        f_value=None,
        p_value=None,
        gg_p_value=None,
        hf_p_value=None,
    ))

    # Between-subjects effects (if present)
    between_factors_tuple: tuple[str, ...] = ()
    if between is not None and len(between) > 0:
        between_name = list(between.keys())[0]
        between_factor = np.array([str(v) for v in between[between_name]])
        between_factors_tuple = (between_name,)

        # Map subject to between-group
        subject_group: dict[str, str] = {}
        for idx in range(N):
            sid = subject_ids[idx]
            if sid not in subject_group:
                subject_group[sid] = between_factor[idx]

        between_levels = sorted(set(subject_group.values()))
        n_between = len(between_levels)

        # Between-subjects ANOVA on subject means
        # SS_between_factor = sum over groups of n_g * (group_mean - grand_mean)^2
        ss_between_factor = 0.0
        for blevel in between_levels:
            group_subject_means = [
                subject_means[subjects.index(sid)]
                for sid in subjects if subject_group[sid] == blevel
            ]
            group_mean = np.mean(group_subject_means)
            n_g = len(group_subject_means)
            ss_between_factor += n_g * (group_mean - grand_mean) ** 2

        # Scale by k (number of conditions)
        ss_between_factor *= k

        ss_between_error = ss_subjects - ss_between_factor
        df_between_factor = n_between - 1
        df_between_error = n - n_between

        ms_between_factor = ss_between_factor / df_between_factor if df_between_factor > 0 else 0.0
        ms_between_error = ss_between_error / df_between_error if df_between_error > 0 else 0.0

        if ms_between_error > 0:
            f_between = ms_between_factor / ms_between_error
            p_between = float(sp_stats.f.sf(f_between, df_between_factor, df_between_error))
        else:
            f_between = 0.0
            p_between = 1.0

        # Insert between-subjects rows before within rows
        between_row = AnovaRMTableRow(
            term=between_name,
            df=float(df_between_factor),
            sum_sq=ss_between_factor,
            mean_sq=ms_between_factor,
            f_value=f_between,
            p_value=p_between,
            gg_p_value=None,
            hf_p_value=None,
        )
        between_error_row = AnovaRMTableRow(
            term='Between-Error',
            df=float(df_between_error),
            sum_sq=ss_between_error,
            mean_sq=ms_between_error,
            f_value=None,
            p_value=None,
            gg_p_value=None,
            hf_p_value=None,
        )
        table_rows = [between_row, between_error_row] + table_rows

    # Sphericity summary
    sphericity = (SphericitySummary(
        factor=within_name,
        mauchly_w=mauchly_w,
        p_value=mauchly_p,
        gg_epsilon=gg_eps,
        hf_epsilon=hf_eps,
    ),)

    # Effect sizes
    eta_sq = {within_name: ss_condition / ss_total if ss_total > 0 else 0.0}
    partial_eta_sq = {
        within_name: ss_condition / (ss_condition + ss_error)
        if (ss_condition + ss_error) > 0 else 0.0
    }

    return AnovaRMParams(
        table=tuple(table_rows),
        n_subjects=n,
        n_obs=N,
        within_factors=(within_name,),
        between_factors=between_factors_tuple,
        sphericity=sphericity,
        correction=correction,
        grand_mean=grand_mean,
        eta_squared=eta_sq,
        partial_eta_squared=partial_eta_sq,
    )


def mauchly_test(
    Y_wide: NDArray,
) -> tuple[float, float, float, float]:
    """
    Mauchly's test of sphericity.

    Tests whether the covariance matrix of the orthonormalized contrast
    variables is proportional to the identity matrix.

    Args:
        Y_wide: (n, k) array, subjects x conditions

    Returns:
        (W, p_value, gg_epsilon, hf_epsilon)
    """
    n, k = Y_wide.shape
    p = k - 1  # number of contrasts

    # Helmert-like orthonormal contrast matrix C: (k, k-1)
    C = _helmert_contrasts(k)

    # Transformed data: (n, k-1)
    Y_transformed = Y_wide @ C

    # Covariance matrix of transformed variables
    S = np.cov(Y_transformed, rowvar=False, ddof=1)

    # Mauchly's W = det(S) / (trace(S)/p)^p
    trace_S = np.trace(S)
    det_S = np.linalg.det(S)
    mean_eigenvalue = trace_S / p

    if mean_eigenvalue <= 0:
        return 0.0, 0.0, 1.0 / p, 1.0 / p

    W = det_S / (mean_eigenvalue ** p)
    W = max(0.0, min(1.0, W))  # numerical safety

    # Chi-squared approximation for p-value
    # df = p*(p+1)/2 - 1
    f = 1.0 - (2.0 * p * p + p + 2.0) / (6.0 * p * (n - 1.0))
    df_chi = p * (p + 1) // 2 - 1

    if W > 0 and df_chi > 0:
        chi_sq = -f * (n - 1.0) * np.log(W)
        p_value = float(sp_stats.chi2.sf(chi_sq, df_chi))
    else:
        p_value = 0.0

    # Greenhouse-Geisser epsilon
    gg_eps = _greenhouse_geisser_epsilon(S, p)

    # Huynh-Feldt epsilon
    hf_eps = _huynh_feldt_epsilon(gg_eps, k, n)

    return float(W), p_value, gg_eps, hf_eps


def _helmert_contrasts(k: int) -> NDArray:
    """
    Generate (k, k-1) orthonormal Helmert contrast matrix.

    Used to transform repeated measures data for sphericity testing.
    """
    C = np.zeros((k, k - 1), dtype=np.float64)

    for j in range(k - 1):
        # Helmert contrast j: compare level j+1 to the mean of levels 0..j
        C[:j + 1, j] = -1.0 / (j + 1)
        C[j + 1, j] = 1.0
        # Normalize
        norm = np.sqrt(np.sum(C[:, j] ** 2))
        C[:, j] /= norm

    return C


def _greenhouse_geisser_epsilon(
    S: NDArray,
    p: int,
) -> float:
    """
    Greenhouse-Geisser epsilon correction.

    epsilon = trace(S)^2 / (p * trace(S @ S))

    Where S is the (p x p) covariance matrix of the orthonormalized
    contrast variables.

    Args:
        S: (p, p) covariance matrix
        p: number of contrasts (k - 1)

    Returns:
        epsilon in [1/p, 1]
    """
    trace_S = np.trace(S)
    trace_S2 = np.trace(S @ S)

    if trace_S2 == 0:
        return 1.0 / p

    eps = (trace_S ** 2) / (p * trace_S2)
    return float(max(1.0 / p, min(1.0, eps)))


def _huynh_feldt_epsilon(
    gg_eps: float,
    k: int,
    n: int,
) -> float:
    """
    Huynh-Feldt epsilon correction.

    epsilon_HF = (n * (k-1) * gg_eps - 2) / ((k-1) * (n - 1 - (k-1) * gg_eps))

    Less conservative than GG. Capped at 1.0.

    Args:
        gg_eps: Greenhouse-Geisser epsilon
        k: number of conditions
        n: number of subjects

    Returns:
        epsilon in [gg_eps, 1.0]
    """
    p = k - 1
    numerator = n * p * gg_eps - 2.0
    denominator = p * (n - 1.0 - p * gg_eps)

    if denominator <= 0:
        return 1.0

    hf_eps = numerator / denominator
    return float(max(gg_eps, min(1.0, hf_eps)))
