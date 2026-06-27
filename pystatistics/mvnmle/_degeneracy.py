"""
Rank-deficiency (collinearity) detection for fitted MVN covariances.

A rank-deficient input — two or more (near-)collinear variables — has no
interior maximum-likelihood estimate: the multivariate-normal likelihood
increases without bound as the fitted covariance approaches singularity. In
that regime the optimizer's own convergence flag is unreliable (it may stall
and report either success or failure), so the *fitted* covariance must be
inspected directly. This module owns that single check.

The degeneracy signal is the smallest eigenvalue of the correlation matrix
implied by the fitted covariance. Using the correlation matrix rather than
the covariance directly makes the measure scale-invariant: it depends only
on the collinearity structure, not on the units of each variable. That lets
a single threshold separate genuine rank-deficiency from variables that are
merely full-rank but ill-conditioned (e.g. measured on very different
scales), which a raw condition number on the covariance cannot do.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import SingularMatrixError, ValidationError

# Smallest correlation-matrix eigenvalue treated as full-rank. Calibrated
# empirically: rank-deficient fits floor at ~3e-6 (an exactly duplicated or
# affine column), while legitimate ill-conditioned datasets stay at >= 4e-4.
# 1e-5 sits in that gap; for two variables it corresponds to |corr| ~ 0.99999.
DEFAULT_COLLINEARITY_TOL: float = 1e-5


def correlation_min_eigenvalue(sigma: NDArray[np.floating]) -> float:
    """Smallest eigenvalue of the correlation matrix implied by ``sigma``.

    A value at or near zero indicates (near-)collinear variables. The
    measure is scale-invariant. A non-finite ``sigma`` or a non-positive
    variance on the diagonal is itself a degenerate fit and returns ``0.0``.

    Parameters
    ----------
    sigma : ndarray
        Square (p, p) fitted covariance matrix.

    Returns
    -------
    float
        The smallest eigenvalue of the derived correlation matrix, in the
        range [0, p]. ``0.0`` for an already-degenerate ``sigma``.
    """
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValidationError(
            f"sigma must be a square matrix, got shape {sigma.shape}"
        )
    if not np.all(np.isfinite(sigma)):
        return 0.0
    diag = np.diag(sigma)
    if np.any(diag <= 0.0):
        return 0.0
    d = np.sqrt(diag)
    corr = sigma / np.outer(d, d)
    eigvals = np.linalg.eigvalsh(corr)  # ascending order
    return float(eigvals[0])


def check_fitted_covariance(
    sigma: NDArray[np.floating],
    *,
    tol: float = DEFAULT_COLLINEARITY_TOL,
    force: bool = False,
) -> str | None:
    """Guard a fitted covariance against rank-deficiency.

    Parameters
    ----------
    sigma : ndarray
        Fitted covariance matrix to inspect.
    tol : float
        Full-rank threshold on the correlation-matrix minimum eigenvalue.
    force : bool
        When True, a degenerate fit is accepted rather than rejected: the
        function returns a warning message instead of raising.

    Returns
    -------
    str or None
        ``None`` when the fit is full-rank (no action needed). When the fit
        is rank-deficient *and* ``force`` is True, a warning message
        describing the degeneracy — the caller should attach it to the
        result and mark the fit not-converged.

    Raises
    ------
    SingularMatrixError
        When the fit is rank-deficient and ``force`` is False.
    """
    min_eig = correlation_min_eigenvalue(sigma)
    if min_eig >= tol:
        return None

    detail = (
        f"the fitted covariance is rank-deficient (correlation-matrix "
        f"minimum eigenvalue {min_eig:.2e} < {tol:.0e}). The input has "
        f"(near-)collinear variables, so no interior maximum-likelihood "
        f"estimate exists and the reported fit is not a true optimum"
    )
    if force:
        return (
            f"Degenerate fit accepted under force=True: {detail}. "
            f"Treat muhat and sigmahat with caution."
        )
    raise SingularMatrixError(
        f"MVN MLE failed: {detail}. Remove the collinear column(s) before "
        f"fitting, or pass force=True to obtain the (non-converged) result "
        f"anyway. The detection threshold is adjustable via collinearity_tol."
    )
