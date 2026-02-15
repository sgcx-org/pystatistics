"""
Sums of squares computation for ANOVA.

All three SS types work by fitting multiple OLS models via regression.fit()
and comparing residual sums of squares. No new solver math — just model
comparisons.

Type I (Sequential):
    Add terms one at a time. SS(term) = RSS(without) - RSS(with).

Type II (Marginal, respects marginality):
    SS(A) = RSS(model without A but with everything A doesn't contain) - RSS(full).
    An interaction A:B contains both A and B, so when computing SS(A), A:B is also dropped.

Type III (Each term last):
    SS(term) = RSS(full minus just that term) - RSS(full).
    Requires deviation coding so that each term's contribution is interpretable.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from scipy import stats as sp_stats

from pystatistics.anova._common import AnovaTableRow
from pystatistics.anova._contrasts import ModelMatrix


def _fit_rss(X: NDArray, y: NDArray) -> float:
    """
    Fit OLS via regression.fit() and return RSS.

    Uses the regression module directly to maintain consistency.
    For ANOVA we only need the RSS, not the full solution object.
    """
    from pystatistics.regression import fit

    if X.shape[1] == 0:
        # No predictors → RSS is total sum of squares
        return float(np.sum((y - np.mean(y)) ** 2))

    sol = fit(X, y, backend='cpu')
    return sol.rss


def _compute_f_and_p(
    ss: float,
    df: int,
    rss_error: float,
    df_error: int,
) -> tuple[float | None, float | None]:
    """Compute F statistic and p-value from SS components."""
    if df <= 0 or df_error <= 0 or rss_error <= 0:
        return None, None

    ms = ss / df
    ms_error = rss_error / df_error
    if ms_error == 0:
        return None, None

    f_val = ms / ms_error
    p_val = float(sp_stats.f.sf(f_val, df, df_error))
    return f_val, p_val


def compute_ss_type1(
    y: NDArray,
    model_matrix: ModelMatrix,
) -> list[AnovaTableRow]:
    """
    Type I (Sequential) Sums of Squares.

    Terms are added in order. SS(term_k) = RSS(terms 1..k-1) - RSS(terms 1..k).
    Order-dependent for unbalanced designs.

    This matches R's default: anova(lm(y ~ A * B)).
    """
    n = len(y)
    terms = [t for t in model_matrix.term_names if t != 'Intercept']

    # Start with intercept-only model
    X_intercept = np.ones((n, 1), dtype=np.float64)
    rss_prev = _fit_rss(X_intercept, y)
    columns_so_far = [X_intercept]

    rows: list[AnovaTableRow] = []

    for term in terms:
        term_slice = model_matrix.term_slices[term]
        term_cols = model_matrix.X[:, term_slice]
        columns_so_far.append(term_cols)
        X_current = np.hstack(columns_so_far)

        rss_current = _fit_rss(X_current, y)
        ss_term = rss_prev - rss_current
        df_term = model_matrix.term_df[term]

        rows.append(AnovaTableRow(
            term=term,
            df=df_term,
            sum_sq=ss_term,
            mean_sq=ss_term / df_term if df_term > 0 else 0.0,
            f_value=None,  # Filled after residuals are known
            p_value=None,
        ))

        rss_prev = rss_current

    # Residuals
    rss_full = rss_prev
    df_residual = n - model_matrix.p
    ms_residual = rss_full / df_residual if df_residual > 0 else 0.0

    # Now compute F and p for each term
    final_rows = []
    for row in rows:
        f_val, p_val = _compute_f_and_p(
            row.sum_sq, row.df, rss_full, df_residual
        )
        final_rows.append(AnovaTableRow(
            term=row.term,
            df=row.df,
            sum_sq=row.sum_sq,
            mean_sq=row.mean_sq,
            f_value=f_val,
            p_value=p_val,
        ))

    final_rows.append(AnovaTableRow(
        term='Residuals',
        df=df_residual,
        sum_sq=rss_full,
        mean_sq=ms_residual,
        f_value=None,
        p_value=None,
    ))

    return final_rows


def compute_ss_type2(
    y: NDArray,
    model_matrix: ModelMatrix,
) -> list[AnovaTableRow]:
    """
    Type II (Marginal) Sums of Squares.

    For each term, SS is computed by comparing two models:
        - "augmented": all terms that don't contain the target, PLUS the target
        - "reduced": all terms that don't contain the target (without the target)
        SS(term) = RSS(reduced) - RSS(augmented)

    Respects the marginality principle: an interaction A:B "contains" both
    A and B. When testing a main effect A, terms that contain A (like A:B)
    are excluded from both the augmented and reduced models. When testing
    the interaction A:B, nothing contains it, so augmented = full model.

    This matches R's: car::Anova(lm(y ~ A * B), type="II").
    """
    n = len(y)
    terms = [t for t in model_matrix.term_names if t != 'Intercept']

    # Full model RSS (used for residuals and for highest-order terms)
    rss_full = _fit_rss(model_matrix.X, y)
    df_residual = n - model_matrix.p

    rows: list[AnovaTableRow] = []

    for term in terms:
        # Collect terms that DON'T contain the target term
        # These form the "base" set for both reduced and augmented models
        base_cols = []
        for other_term in model_matrix.term_names:
            if other_term == term:
                continue
            if _term_contains(other_term, term):
                continue
            other_slice = model_matrix.term_slices[other_term]
            base_cols.append(model_matrix.X[:, other_slice])

        # Reduced model: base terms only (without target)
        if base_cols:
            X_reduced = np.hstack(base_cols)
        else:
            X_reduced = np.empty((n, 0), dtype=np.float64)

        # Augmented model: base terms + target term
        target_cols = model_matrix.X[:, model_matrix.term_slices[term]]
        if base_cols:
            X_augmented = np.hstack(base_cols + [target_cols])
        else:
            X_augmented = target_cols

        rss_reduced = _fit_rss(X_reduced, y)
        rss_augmented = _fit_rss(X_augmented, y)
        ss_term = rss_reduced - rss_augmented
        df_term = model_matrix.term_df[term]

        f_val, p_val = _compute_f_and_p(ss_term, df_term, rss_full, df_residual)

        rows.append(AnovaTableRow(
            term=term,
            df=df_term,
            sum_sq=ss_term,
            mean_sq=ss_term / df_term if df_term > 0 else 0.0,
            f_value=f_val,
            p_value=p_val,
        ))

    # Residuals
    ms_residual = rss_full / df_residual if df_residual > 0 else 0.0
    rows.append(AnovaTableRow(
        term='Residuals',
        df=df_residual,
        sum_sq=rss_full,
        mean_sq=ms_residual,
        f_value=None,
        p_value=None,
    ))

    return rows


def compute_ss_type3(
    y: NDArray,
    model_matrix: ModelMatrix,
) -> list[AnovaTableRow]:
    """
    Type III Sums of Squares.

    SS(term) = RSS(full minus just that term) - RSS(full).
    Each term is tested as if it were the last one added.

    IMPORTANT: model_matrix must use deviation coding for Type III
    to give meaningful results (otherwise SS depends on cell frequencies).

    This matches R's: car::Anova(lm(y ~ A * B), type="III") with contr.sum.
    """
    n = len(y)
    terms = [t for t in model_matrix.term_names if t != 'Intercept']

    # Full model RSS
    rss_full = _fit_rss(model_matrix.X, y)
    df_full = n - model_matrix.p
    df_residual = df_full

    rows: list[AnovaTableRow] = []

    for term in terms:
        # Build reduced model: drop ONLY this term
        reduced_cols = []
        for other_term in model_matrix.term_names:
            if other_term == term:
                continue
            other_slice = model_matrix.term_slices[other_term]
            reduced_cols.append(model_matrix.X[:, other_slice])

        if reduced_cols:
            X_reduced = np.hstack(reduced_cols)
        else:
            X_reduced = np.empty((n, 0), dtype=np.float64)

        rss_reduced = _fit_rss(X_reduced, y)
        ss_term = rss_reduced - rss_full
        df_term = model_matrix.term_df[term]

        f_val, p_val = _compute_f_and_p(ss_term, df_term, rss_full, df_residual)

        rows.append(AnovaTableRow(
            term=term,
            df=df_term,
            sum_sq=ss_term,
            mean_sq=ss_term / df_term if df_term > 0 else 0.0,
            f_value=f_val,
            p_value=p_val,
        ))

    # Residuals
    ms_residual = rss_full / df_residual if df_residual > 0 else 0.0
    rows.append(AnovaTableRow(
        term='Residuals',
        df=df_residual,
        sum_sq=rss_full,
        mean_sq=ms_residual,
        f_value=None,
        p_value=None,
    ))

    return rows


def _term_contains(candidate: str, target: str) -> bool:
    """
    Check if candidate term contains target term.

    A term 'A:B' contains 'A' and 'B' (it's an interaction of those factors).
    A main effect 'A' does NOT contain itself (we handle that with ==).
    The Intercept contains nothing.

    Used by Type II to respect marginality: when testing A, also drop A:B.
    """
    if ':' not in candidate:
        return False
    parts = candidate.split(':')
    return target in parts
