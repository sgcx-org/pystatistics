"""Starting values for iterative MVN estimation with missing data.

One job: turn an incomplete data matrix into a mean vector and a positive
definite covariance matrix from which an optimizer or EM can start.

Available-case moments
----------------------
* ``mean[j]`` is the average of the observed values of variable ``j``.
* ``cov[j, k]`` is the pairwise-complete covariance: over the rows in which
  both ``j`` and ``k`` are observed, the average product of deviations from
  the two variables' means *over those same rows*. The divisor is the number
  of such rows (the maximum-likelihood divisor, matching the estimand).
  A pair observed together in fewer than two rows has covariance 0.
* ``cov[j, j]`` is therefore the variance of the observed values of ``j``.

Each entry of this matrix is a legitimate moment estimate, but because the
entries come from different subsets of rows the matrix as a whole need not be
positive definite.

Positive-definite start
-----------------------
The start is obtained by shrinking the *correlation* matrix towards the
identity just far enough that its smallest eigenvalue reaches
``MIN_CORRELATION_EIGENVALUE``:

    R_a = (1 - a) R + a I,    a = (tau - lambda_min) / (1 - lambda_min),

and ``a = 0`` (no change at all) when ``lambda_min >= tau`` already. The
variances are left untouched, so the rule is invariant to the units of the
variables. A variance that is not positive and finite (a constant or
once-observed column, which the callers' input checks reject unless forced)
is replaced by 1 and its covariances by 0.

This affects only where the iteration starts. The likelihood is not modified,
so the reported estimate does not depend on it.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import NumericalError, ValidationError

# Smallest eigenvalue allowed for the starting correlation matrix. Three
# orders of magnitude above float32 machine epsilon, so the start factorises
# safely in both float64 and float32.
MIN_CORRELATION_EIGENVALUE = 1e-4


def available_case_moments(data: NDArray[np.float64]
                           ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Available-case mean vector and pairwise-complete covariance matrix.

    Parameters
    ----------
    data : np.ndarray, shape (n_obs, n_vars)
        Data matrix with ``np.nan`` for missing values.

    Returns
    -------
    (mean, cov) : np.ndarray (n_vars,), np.ndarray (n_vars, n_vars)
        ``cov`` is symmetric but not necessarily positive definite.

    Raises
    ------
    ValidationError
        If ``data`` is not 2-D or a variable has no observed value.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValidationError(f"Data must be 2D, got shape {data.shape}")

    observed = ~np.isnan(data)
    w = observed.astype(np.float64)
    n_j = w.sum(axis=0)
    if np.any(n_j == 0):
        raise ValidationError("Data contains columns with all missing values")

    mean = np.where(observed, data, 0.0).sum(axis=0) / n_j

    # Centre on the available-case means first: the pairwise formulas below
    # then subtract small quantities instead of large, nearly equal ones.
    z = np.where(observed, data - mean, 0.0)

    n_jk = w.T @ w                   # rows in which both j and k are observed
    sum_j = z.T @ w                  # [j, k]: sum of z_j over those rows
    cross = z.T @ z                  # [j, k]: sum of z_j * z_k over those rows

    enough = n_jk >= 2
    safe_n = np.where(enough, n_jk, 1.0)
    mean_j = sum_j / safe_n          # mean of z_j over the pair's rows
    cov = np.where(enough, cross / safe_n - mean_j * mean_j.T, 0.0)
    return mean, 0.5 * (cov + cov.T)


def positive_definite_start(cov: NDArray[np.float64],
                            min_eigenvalue: float = MIN_CORRELATION_EIGENVALUE
                            ) -> NDArray[np.float64]:
    """Return a positive definite covariance to start from.

    ``cov`` is returned unchanged when its correlation matrix already has
    smallest eigenvalue at least ``min_eigenvalue``; otherwise the
    correlations are shrunk towards zero by the smallest amount that achieves
    it (see the module docstring).

    Raises
    ------
    ValidationError
        If ``cov`` is not square or ``min_eigenvalue`` is not in (0, 1).
    NumericalError
        If ``cov`` contains non-finite entries.
    """
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValidationError(
            f"cov must be a square matrix, got shape {cov.shape}")
    if not 0.0 < min_eigenvalue < 1.0:
        raise ValidationError(
            f"min_eigenvalue must lie in (0, 1), got {min_eigenvalue!r}")
    if not np.all(np.isfinite(cov)):
        raise NumericalError(
            "Available-case covariance contains non-finite values; cannot "
            "form starting values. Check the data for infinite entries.")

    start = cov.copy()
    var = np.diag(start).copy()
    degenerate = ~(var > 0.0)
    if np.any(degenerate):
        start[degenerate, :] = 0.0
        start[:, degenerate] = 0.0
        var[degenerate] = 1.0
        start[np.diag_indices_from(start)] = var

    sd = np.sqrt(var)
    corr = start / np.outer(sd, sd)
    np.fill_diagonal(corr, 1.0)

    lam_min = float(np.linalg.eigvalsh(corr)[0])
    if lam_min >= min_eigenvalue:
        return start

    # trace(corr) = p, so lam_min <= 1 and the denominator is positive.
    a = (min_eigenvalue - lam_min) / (1.0 - lam_min)
    corr = (1.0 - a) * corr
    np.fill_diagonal(corr, 1.0)
    return corr * np.outer(sd, sd)
