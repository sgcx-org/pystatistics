"""
Parameter payloads for Generalized Additive Model results.

Each dataclass is a frozen payload describing a fitted GAM or
one of its smooth terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GAMParams:
    """Parameters from a fitted GAM.

    Carries the full numerical output of the penalized iteratively
    re-weighted least squares (P-IRLS) fitting procedure.

    Attributes:
        coefficients: Full coefficient vector (parametric + all basis functions).
        fitted_values: Response-scale predictions (mu_hat).
        linear_predictor: Link-scale predictions (eta_hat = X_aug @ coefficients).
        residuals: Working residuals (y - mu_hat).
        edf: Effective degrees of freedom per smooth term.
        total_edf: Sum of per-smooth edf plus parametric term count.
        scale: Estimated or fixed dispersion parameter.
        gcv: Generalized cross-validation score.
        ubre: Un-biased risk estimator score (= AIC/n for known scale).
        deviance: Model deviance.
        null_deviance: Null-model deviance (intercept only).
        log_likelihood: Maximized log-likelihood.
        aic: Akaike information criterion.
        n_obs: Number of observations used in fitting.
        family_name: Name of the exponential family (e.g. 'gaussian').
        link_name: Name of the link function (e.g. 'identity').
        converged: Whether the P-IRLS algorithm converged.
        n_iter: Number of P-IRLS iterations executed.
        method: Smoothing parameter selection method ('GCV' or 'REML').
    """

    coefficients: NDArray[np.floating[Any]]
    fitted_values: NDArray[np.floating[Any]]
    linear_predictor: NDArray[np.floating[Any]]
    residuals: NDArray[np.floating[Any]]
    edf: NDArray[np.floating[Any]]
    total_edf: float
    scale: float
    gcv: float
    ubre: float
    deviance: float
    null_deviance: float
    log_likelihood: float
    aic: float
    n_obs: int
    family_name: str
    link_name: str
    converged: bool
    n_iter: int
    method: str


@dataclass(frozen=True)
class SmoothInfo:
    """Information about a single smooth term in a fitted GAM.

    One ``SmoothInfo`` is produced per ``s()`` term after fitting.
    It records the basis metadata and the approximate significance
    test for the smooth (following Wood, 2013).

    Attributes:
        term_name: Display name, e.g. ``'s(x1)'``.
        var_name: Bare predictor name, e.g. ``'x1'``.
        basis_type: Basis identifier: ``'cr'`` (cubic regression spline)
            or ``'tp'`` (thin plate regression spline).
        k: Number of basis functions.
        edf: Effective degrees of freedom for this term.
        ref_df: Reference degrees of freedom used in the approximate test.
        chi_sq: Approximate chi-squared (or F) statistic.
        p_value: Approximate p-value from the significance test.
        coef_indices: ``(start, end)`` slice into the full coefficient vector
            identifying which coefficients belong to this term.
    """

    term_name: str
    var_name: str
    basis_type: str
    k: int
    edf: float
    ref_df: float
    chi_sq: float
    p_value: float
    coef_indices: tuple[int, int]
