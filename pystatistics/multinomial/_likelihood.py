"""
Core likelihood functions for multinomial logistic regression.

Implements the negative log-likelihood and its gradient for the
multinomial logit (softmax) model. Uses the log-sum-exp trick
for numerical stability.

Model:
    P(Y = j | x) = exp(x' beta_j) / sum_k exp(x' beta_k)
    with beta_J = 0 (reference class, last class).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _compute_log_probs(
    params_flat: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
    n_classes: int,
) -> NDArray[np.floating[Any]]:
    """Compute log-probabilities using the log-sum-exp trick.

    Args:
        params_flat: Flattened parameter vector of length (J-1) * p.
        X: Design matrix of shape (n, p).
        n_classes: Total number of classes J.

    Returns:
        Log-probability matrix of shape (n, J). Each row sums to
        approximately 0 in log-space (i.e. probabilities sum to 1).
    """
    n, p = X.shape
    n_nonref = n_classes - 1

    # Reshape flat params to (J-1, p) coefficient matrix
    beta = params_flat.reshape(n_nonref, p)

    # Linear predictors: (n, J-1) for non-reference classes
    eta_nonref = X @ beta.T

    # Full eta matrix: append zeros for the reference class
    eta = np.zeros((n, n_classes), dtype=np.float64)
    eta[:, :n_nonref] = eta_nonref
    # eta[:, -1] = 0 already (reference class)

    # Log-sum-exp trick for numerical stability
    eta_max = np.max(eta, axis=1, keepdims=True)
    shifted = eta - eta_max
    log_sum_exp = eta_max + np.log(
        np.sum(np.exp(shifted), axis=1, keepdims=True)
    )
    log_probs = eta - log_sum_exp

    return log_probs


def multinomial_negloglik(
    params_flat: NDArray[np.floating[Any]],
    y_onehot: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
    n_classes: int,
) -> float:
    """Negative log-likelihood for multinomial logistic regression.

    Computes -sum_i sum_j y_{ij} * log(pi_{ij}) where pi is the
    predicted probability from the softmax model.

    Args:
        params_flat: Flattened parameter vector of length (J-1) * p,
            where J is the number of classes and p is the number of
            predictors (including intercept if present).
        y_onehot: One-hot encoded response matrix of shape (n, J).
        X: Design matrix of shape (n, p).
        n_classes: Total number of classes J.

    Returns:
        Scalar negative log-likelihood value.
    """
    log_probs = _compute_log_probs(params_flat, X, n_classes)

    # Negative log-likelihood: -sum of y * log(pi)
    negloglik = -np.sum(y_onehot * log_probs)

    return float(negloglik)


def multinomial_gradient(
    params_flat: NDArray[np.floating[Any]],
    y_onehot: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
    n_classes: int,
) -> NDArray[np.floating[Any]]:
    """Gradient of the negative log-likelihood.

    For each non-reference class j:
        d(-loglik)/d(beta_j) = -X' (y_j - pi_j)

    where pi_j is the vector of predicted probabilities for class j
    across all observations.

    Args:
        params_flat: Flattened parameter vector of length (J-1) * p.
        y_onehot: One-hot encoded response matrix of shape (n, J).
        X: Design matrix of shape (n, p).
        n_classes: Total number of classes J.

    Returns:
        Flattened gradient vector of length (J-1) * p.
    """
    n_nonref = n_classes - 1
    log_probs = _compute_log_probs(params_flat, X, n_classes)
    probs = np.exp(log_probs)

    # Residuals for non-reference classes: y_j - pi_j
    residuals = y_onehot[:, :n_nonref] - probs[:, :n_nonref]

    # Gradient for each non-ref class: -X' @ residuals_j
    # Shape: (J-1, p)
    grad_matrix = -(X.T @ residuals).T

    return grad_matrix.ravel()


def compute_probs(
    params_flat: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
    n_classes: int,
) -> NDArray[np.floating[Any]]:
    """Compute predicted probabilities from fitted parameters.

    Args:
        params_flat: Flattened parameter vector of length (J-1) * p.
        X: Design matrix of shape (n, p).
        n_classes: Total number of classes J.

    Returns:
        Probability matrix of shape (n, J). Each row sums to 1.
    """
    log_probs = _compute_log_probs(params_flat, X, n_classes)
    return np.exp(log_probs)
