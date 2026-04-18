"""Monotone missingness: detection and closed-form MLE.

A missingness pattern is *monotone* if the variables can be ordered
such that each observation's missing entries form a contiguous suffix
— equivalently, column $i$ being missing implies every column ordered
after $i$ is also missing for that observation. Longitudinal cohorts
with attrition, panel surveys with dropout, and most sequentially
administered instruments produce monotone patterns.

When the data are monotone, the MVN MLE has a closed form via a chain
of OLS regressions, due originally to Anderson (1957). The algorithm
is $O(v^3 n)$ deterministic work with no iteration, so it dominates
EM and BFGS by orders of magnitude when it applies.

References
----------
Anderson, T. W. (1957). Maximum likelihood estimates for a
multivariate normal distribution when some observations are missing.
JASA, 52(278), 200-203.

Little, R. J. A. & Rubin, D. B. (2002). Statistical Analysis with
Missing Data, 2nd ed. Wiley. Chapter 7.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ValidationError


def is_monotone(data) -> bool:
    """Return True iff the missingness pattern is monotone.

    Parameters
    ----------
    data : array-like, shape (n, v)
        Data matrix with NaN for missing values.

    Returns
    -------
    bool
    """
    return monotone_permutation(data) is not None


def monotone_permutation(data) -> Optional[NDArray]:
    """Find a variable ordering that makes the missingness pattern
    monotone, or return ``None`` if no such ordering exists.

    Algorithm: order columns by number of missing values ascending.
    The pattern is monotone under that order iff each column's set
    of missing-row indices is a subset of the next column's set.
    Equivalent to "at each prefix, the set of rows with at least one
    missing among the prefix columns is the set of rows where the
    *last* prefix column is missing."

    Parameters
    ----------
    data : array-like, shape (n, v)

    Returns
    -------
    order : ndarray of int64, shape (v,) or None
        Permutation that yields monotone patterns, or ``None``.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValidationError("Data must be 2D")

    n, v = data.shape
    missing = np.isnan(data)

    # Missing count per column. Ties broken by original column index
    # for determinism.
    n_missing = missing.sum(axis=0)
    order = np.argsort(n_missing, kind="stable")

    # Nestedness check in the sorted order: for each consecutive pair,
    # the earlier column's missing rows must be a subset of the later
    # column's missing rows. Implemented as a bitwise check on the
    # missing-indicator columns, no Python-level set allocation.
    for i in range(v - 1):
        earlier = missing[:, order[i]]
        later = missing[:, order[i + 1]]
        # earlier ⊆ later  ⇔  earlier AND NOT later is nowhere True
        if np.any(earlier & ~later):
            return None

    return order.astype(np.int64)


def mlest_monotone_closed_form(data) -> tuple[NDArray, NDArray, int]:
    """Closed-form MVN MLE for monotone missingness.

    Works by ordering variables as per ``monotone_permutation`` so that
    missingness is a nested sequence, then performing a chain of OLS
    regressions. Specifically, with variables ordered
    $X_1, X_2, \\ldots, X_v$ such that every observation with $X_k$
    observed also has $X_1, \\ldots, X_{k-1}$ observed:

        1. $\\hat\\mu_1, \\hat\\sigma_{11}$ are the sample mean and
           variance of $X_1$ over all observations (it is never
           missing under the monotone order).
        2. For $k = 2, \\ldots, v$: using observations where
           $X_k$ is observed, fit OLS regression
           $X_k = \\beta_{k,0} + \\sum_{j<k} \\beta_{k,j} X_j + \\epsilon_k$
           with $\\mathrm{Var}(\\epsilon_k) = \\rho_k^2$. Back-
           substitute into the full $(\\mu, \\Sigma)$ via
                $\\hat\\mu_k = \\beta_{k,0} + \\beta_{k,1:}^\\top \\hat\\mu_{1:k-1}$
                $\\hat\\Sigma_{k, 1:k-1} = \\beta_{k,1:}^\\top \\hat\\Sigma_{1:k-1, 1:k-1}$
                $\\hat\\Sigma_{k, k} = \\rho_k^2 + \\beta_{k,1:}^\\top \\hat\\Sigma_{1:k-1, 1:k-1} \\beta_{k,1:}$
        3. Undo the variable permutation to return parameters in the
           user's original variable order.

    Matches Anderson (1957). Runs in $O(v^3 n)$ deterministic work;
    no iteration.

    Parameters
    ----------
    data : array-like, shape (n, v)

    Returns
    -------
    mu : ndarray (v,)
    sigma : ndarray (v, v)
    n_effective : int
        Number of rows that participated in at least one regression
        (all rows that aren't all-missing).

    Raises
    ------
    ValidationError
        If the data are not monotone, or if a regression step has
        fewer observations than predictors (identifiability failure).
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValidationError("Data must be 2D")
    n, v = data.shape

    order = monotone_permutation(data)
    if order is None:
        raise ValidationError(
            "Data are not monotone — no column permutation makes each "
            "observation's missing entries a contiguous suffix. Use "
            "algorithm='em' or 'direct' for general missingness."
        )

    # Reorder columns into the monotone-compatible sequence.
    X = data[:, order]
    # obs_k: boolean mask for observations where column k is observed.
    # Under the monotone order these satisfy obs_1 ⊇ obs_2 ⊇ ... ⊇ obs_v.
    obs_per_col = ~np.isnan(X)  # (n, v)

    mu_ord = np.zeros(v)
    sigma_ord = np.zeros((v, v))

    # Step k=0: univariate mean and variance of the first (never-missing
    # under the monotone order) variable.
    if not obs_per_col[:, 0].all():
        # Even the "most-observed" column has NaNs; this degenerates
        # the monotone case but the algorithm can still handle it if
        # at least some observations are present.
        x1 = X[obs_per_col[:, 0], 0]
    else:
        x1 = X[:, 0]

    if x1.size < 2:
        raise ValidationError(
            "Monotone MLE: fewer than 2 observations for the first "
            "variable in the monotone order; cannot identify its "
            "variance."
        )

    mu_ord[0] = float(np.mean(x1))
    sigma_ord[0, 0] = float(np.mean((x1 - mu_ord[0]) ** 2))

    # Step k=1..v-1: regress X_k on X_0..X_{k-1} using obs_k rows.
    for k in range(1, v):
        rows = obs_per_col[:, k]
        n_k = int(rows.sum())

        if n_k < k + 1:
            raise ValidationError(
                f"Monotone MLE: only {n_k} observations available for "
                f"the regression of variable {int(order[k])} on "
                f"{k} predictors; need at least {k + 1}."
            )

        # Predictors: columns 0..k-1 restricted to the rows where X_k
        # is observed. Under the monotone ordering, all of columns
        # 0..k-1 are observed on these rows too.
        X_pred = X[rows, :k]       # (n_k, k)
        y = X[rows, k]             # (n_k,)

        X_aug = np.column_stack([np.ones(n_k), X_pred])  # (n_k, k+1)
        # OLS via normal equations. Sample sizes here are modest
        # (n_k <= n) and k is bounded by v; use lstsq for stability.
        coef, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        intercept = float(coef[0])
        slopes = coef[1:]                         # (k,)
        resid = y - X_aug @ coef
        # MLE variance (divide by n, not n-k-1) to match the
        # standard-MVN-MLE convention used elsewhere in this module.
        rho2 = float(np.mean(resid ** 2))

        # Back-substitute into (mu, sigma) in the ordered basis.
        mu_prev = mu_ord[:k]
        sigma_prev = sigma_ord[:k, :k]

        mu_ord[k] = intercept + float(slopes @ mu_prev)
        sigma_prev_slopes = sigma_prev @ slopes    # (k,)
        sigma_ord[k, :k] = sigma_prev_slopes
        sigma_ord[:k, k] = sigma_prev_slopes
        sigma_ord[k, k] = rho2 + float(slopes @ sigma_prev_slopes)

    # Undo the permutation so the returned parameters are in the
    # user's original column order.
    inv_order = np.argsort(order)
    mu = mu_ord[inv_order]
    sigma = sigma_ord[np.ix_(inv_order, inv_order)]
    # Symmetrise to wash out any floating-point asymmetry from the
    # scatter-plus-symmetric-slice back-substitution above.
    sigma = 0.5 * (sigma + sigma.T)

    n_effective = int((~np.isnan(data).all(axis=1)).sum())
    return mu, sigma, n_effective
