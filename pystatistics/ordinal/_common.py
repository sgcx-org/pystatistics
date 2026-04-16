"""
Common types for ordinal regression.

Defines OrdinalParams, the frozen dataclass payload for proportional
odds / cumulative link model results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class OrdinalParams:
    """
    Parameter payload for ordinal regression (proportional odds model).

    Stores all fitted quantities from a cumulative link model, matching
    the information available from R's MASS::polr().

    Attributes
    ----------
    coefficients : NDArray
        Slope parameters beta (length p). These are shared across all
        thresholds under the proportional odds assumption.
    thresholds : NDArray
        Cutpoint parameters alpha (length K-1), ordered such that
        alpha_1 < alpha_2 < ... < alpha_{K-1}.
    vcov : NDArray
        Variance-covariance matrix of all parameters, ordered as
        [thresholds, coefficients], shape (K-1+p, K-1+p).
    log_likelihood : float
        Maximized log-likelihood of the fitted model.
    deviance : float
        Residual deviance, equal to -2 * log_likelihood.
    aic : float
        Akaike information criterion, equal to deviance + 2 * n_params.
    n_obs : int
        Number of observations used in fitting.
    n_levels : int
        Number of ordered response categories (K).
    level_names : tuple[str, ...]
        Labels for the K ordered categories.
    n_iter : int
        Number of optimizer iterations completed.
    converged : bool
        Whether the optimizer converged within tolerance.
    method : str
        Link function name used ('logistic', 'probit', or 'cloglog').
    """

    coefficients: NDArray[np.floating[Any]]
    thresholds: NDArray[np.floating[Any]]
    vcov: NDArray[np.floating[Any]]
    log_likelihood: float
    deviance: float
    aic: float
    n_obs: int
    n_levels: int
    level_names: tuple[str, ...]
    n_iter: int
    converged: bool
    method: str
