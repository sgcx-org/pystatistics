"""
Rubin's rules — combine analyses across the ``m`` completed datasets.

Multiple imputation is only complete once the ``m`` separate analyses are pooled
into a single inference. The user fits their model on each completed dataset and
collects, for each quantity of interest, a point estimate ``Q_i`` and its
squared standard error ``U_i`` (the within-imputation variance). ``pool``
combines them with Rubin's (1987) rules:

    Qbar = mean(Q_i)                         pooled point estimate
    Ubar = mean(U_i)                         within-imputation variance
    B    = var(Q_i)        (ddof=1)          between-imputation variance
    T    = Ubar + (1 + 1/m) B                total variance
    SE   = sqrt(T)

with Barnard & Rubin (1999) degrees of freedom (so finite complete-data df are
respected), and the standard derived diagnostics: relative increase in variance
(``riv``), ``lambda``, and the fraction of missing information (``fmi``).

This module is method- and backend-agnostic: it pools numbers, so it is
unchanged by Stage 2 (GPU) or Stage 3 (categorical methods).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from pystatistics.core.exceptions import ValidationError


@dataclass(frozen=True)
class PooledResult:
    """Pooled estimate(s) and inference from Rubin's rules.

    Each field is a scalar when a single quantity was pooled, or a vector
    (one entry per quantity) when several were pooled together.
    """

    estimate: NDArray[np.floating[Any]] | float   # Qbar
    se: NDArray[np.floating[Any]] | float          # sqrt(T)
    df: NDArray[np.floating[Any]] | float          # Barnard-Rubin df
    ci_low: NDArray[np.floating[Any]] | float
    ci_high: NDArray[np.floating[Any]] | float
    within: NDArray[np.floating[Any]] | float      # Ubar
    between: NDArray[np.floating[Any]] | float      # B
    total: NDArray[np.floating[Any]] | float        # T
    riv: NDArray[np.floating[Any]] | float          # relative increase in variance
    lambda_: NDArray[np.floating[Any]] | float      # proportion of variance from missingness
    fmi: NDArray[np.floating[Any]] | float          # fraction of missing information
    m: int
    alpha: float


def pool(
    estimates,
    variances,
    *,
    dfcom: float | None = None,
    alpha: float = 0.05,
) -> PooledResult:
    """Pool ``m`` estimates and their variances with Rubin's rules.

    Parameters
    ----------
    estimates : array-like
        Point estimates. Shape ``(m,)`` for a single quantity, or ``(m, k)`` to
        pool ``k`` quantities at once (e.g. all coefficients of a regression).
    variances : array-like
        Within-imputation variances ``U_i`` — the *squared* standard errors of
        the corresponding estimates. Same shape as ``estimates``.
    dfcom : float, optional
        Complete-data degrees of freedom (e.g. ``n - p`` for a linear model).
        Used for the Barnard-Rubin df adjustment. If None, treated as infinite
        (no small-sample adjustment), which reduces to the classic Rubin df.
    alpha : float
        Significance level for the confidence interval (default 0.05 -> 95% CI).

    Returns
    -------
    PooledResult
    """
    Q = np.asarray(estimates, dtype=np.float64)
    U = np.asarray(variances, dtype=np.float64)

    scalar_input = Q.ndim == 1
    if scalar_input:
        Q = Q[:, None]
        U = U[:, None]
    if Q.ndim != 2:
        raise ValidationError(
            f"estimates must be 1D (m,) or 2D (m, k), got {Q.ndim}D"
        )
    if U.shape != Q.shape:
        raise ValidationError(
            f"variances shape {U.shape} must match estimates shape {Q.shape}"
        )

    m = Q.shape[0]
    if m < 1:
        raise ValidationError("need at least one imputation to pool")
    if np.any(U < 0):
        raise ValidationError("variances must be non-negative (they are squared SEs)")
    if not (0.0 < alpha < 1.0):
        raise ValidationError(f"alpha must be in (0, 1), got {alpha}")

    dfcom_val = np.inf if dfcom is None else float(dfcom)
    if dfcom_val <= 0:
        raise ValidationError(f"dfcom must be positive, got {dfcom}")

    qbar = Q.mean(axis=0)
    ubar = U.mean(axis=0)
    # Between-imputation variance needs m >= 2; with m == 1 there is no
    # between-imputation component (B = 0).
    between = Q.var(axis=0, ddof=1) if m > 1 else np.zeros_like(qbar)

    total = ubar + (1.0 + 1.0 / m) * between
    se = np.sqrt(total)

    with np.errstate(divide="ignore", invalid="ignore"):
        riv = np.where(ubar > 0, (1.0 + 1.0 / m) * between / ubar, 0.0)
        lam = np.where(total > 0, (1.0 + 1.0 / m) * between / total, 0.0)

    df = _barnard_rubin_df(m, lam, dfcom_val)
    fmi = (riv + 2.0 / (df + 3.0)) / (riv + 1.0)

    # t-based CI; scipy handles df = inf as the normal limit.
    tcrit = stats.t.ppf(1.0 - alpha / 2.0, df)
    ci_low = qbar - tcrit * se
    ci_high = qbar + tcrit * se

    def out(x):
        return float(x[0]) if scalar_input else x

    return PooledResult(
        estimate=out(qbar),
        se=out(se),
        df=out(df),
        ci_low=out(ci_low),
        ci_high=out(ci_high),
        within=out(ubar),
        between=out(between),
        total=out(total),
        riv=out(riv),
        lambda_=out(lam),
        fmi=out(fmi),
        m=m,
        alpha=alpha,
    )


def _barnard_rubin_df(m: int, lam: NDArray, dfcom: float) -> NDArray:
    """Barnard & Rubin (1999) degrees of freedom.

    Reduces to the classic Rubin df ``(m-1)/lambda^2`` as ``dfcom -> inf``.
    Handles ``lambda == 0`` (no between-imputation variance) by returning
    ``dfcom`` — the complete-data df — which is the correct limit.
    """
    lam = np.asarray(lam, dtype=np.float64)
    if m <= 1:
        return np.full_like(lam, dfcom)

    # Where lambda == 0, df_old is infinite; guard the division.
    with np.errstate(divide="ignore", invalid="ignore"):
        df_old = np.where(lam > 0, (m - 1) / np.square(lam), np.inf)

    if np.isinf(dfcom):
        return df_old

    df_obs = (dfcom + 1.0) / (dfcom + 3.0) * dfcom * (1.0 - lam)
    with np.errstate(divide="ignore", invalid="ignore"):
        df = np.where(
            np.isinf(df_old),
            df_obs,
            df_old * df_obs / (df_old + df_obs),
        )
    return df
