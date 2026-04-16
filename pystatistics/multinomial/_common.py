"""
Parameter payload for multinomial logistic regression results.

The MultinomialParams dataclass is a frozen payload that holds all
fitted model quantities. It is wrapped by MultinomialSolution for
user-facing access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MultinomialParams:
    """Multinomial logistic regression parameters.

    Matches the output of R's nnet::multinom().

    Attributes:
        coefficient_matrix: Shape (J-1, p) matrix of coefficients.
            One row per non-reference class, one column per predictor
            (including intercept if present in X).
        vcov: Shape ((J-1)*p, (J-1)*p) variance-covariance matrix of
            the flattened coefficient vector.
        fitted_probs: Shape (n, J) matrix of predicted probabilities
            for each observation and class.
        log_likelihood: Maximized log-likelihood of the fitted model.
        deviance: Residual deviance, equal to -2 * log_likelihood.
        null_deviance: Deviance of the intercept-only (null) model,
            equal to -2 * null_log_likelihood.
        aic: Akaike Information Criterion, equal to
            -2 * log_likelihood + 2 * n_parameters.
        n_obs: Number of observations used in fitting.
        n_classes: Number of response classes J.
        class_names: Tuple of class labels in order, with the last
            element being the reference class.
        feature_names: Tuple of predictor names matching columns of X.
        n_iter: Number of optimizer iterations performed.
        converged: Whether the optimizer converged within tolerance.
    """

    coefficient_matrix: NDArray[np.floating[Any]]
    vcov: NDArray[np.floating[Any]]
    fitted_probs: NDArray[np.floating[Any]]
    log_likelihood: float
    deviance: float
    null_deviance: float
    aic: float
    n_obs: int
    n_classes: int
    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    n_iter: int
    converged: bool
