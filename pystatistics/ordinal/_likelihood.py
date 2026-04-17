"""
Cumulative link model likelihood and gradient.

Implements the negative log-likelihood and its analytical gradient for
the proportional odds model (and probit / cloglog variants). Uses an
unconstrained parameterization for thresholds to allow standard
unconstrained optimization.

Threshold parameterization
--------------------------
To enforce alpha_1 < alpha_2 < ... < alpha_{K-1} without box constraints,
we optimize over raw parameters:
    raw_1 = alpha_1
    raw_j = log(alpha_j - alpha_{j-1})   for j = 2, ..., K-1

The inverse mapping is:
    alpha_1 = raw_1
    alpha_j = alpha_{j-1} + exp(raw_j)   for j = 2, ..., K-1

References
----------
    Agresti, A. (2010). Analysis of Ordinal Categorical Data (2nd ed.)
    McCullagh, P. (1980). Regression models for ordinal data. JRSS-B.
    Venables, W. N. & Ripley, B. D. (2002). Modern Applied Statistics with S.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatistics.regression.families import Link


# -- Complementary log-log link (not in families.py) ----------------------

class CLogLogLink(Link):
    """
    Complementary log-log link: g(mu) = log(-log(1 - mu)).

    This is the default link for the complementary log-log model, an
    asymmetric alternative to logit/probit for cumulative link models.
    """

    @property
    def name(self) -> str:
        """Return link function name."""
        return 'cloglog'

    def link(self, mu: NDArray) -> NDArray:
        """
        Compute g(mu) = log(-log(1 - mu)).

        Args:
            mu: Array of probabilities in (0, 1).

        Returns:
            Linear predictor values.
        """
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return np.log(-np.log(1 - mu))

    def linkinv(self, eta: NDArray) -> NDArray:
        """
        Compute g^{-1}(eta) = 1 - exp(-exp(eta)).

        Args:
            eta: Array of linear predictor values.

        Returns:
            Probability values in (0, 1).
        """
        eta = np.clip(eta, -500, 500)
        return 1.0 - np.exp(-np.exp(eta))

    def mu_eta(self, eta: NDArray) -> NDArray:
        """
        Compute d(mu)/d(eta) = exp(eta) * exp(-exp(eta)).

        Args:
            eta: Array of linear predictor values.

        Returns:
            Derivative of inverse link.
        """
        eta = np.clip(eta, -500, 500)
        exp_eta = np.exp(eta)
        return np.maximum(exp_eta * np.exp(-exp_eta), 1e-10)


# -- Threshold transforms -------------------------------------------------

def raw_to_thresholds(raw: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """
    Convert unconstrained raw parameters to ordered thresholds.

    Args:
        raw: Unconstrained parameters of length K-1. raw[0] is alpha_1,
            raw[j] for j >= 1 is log(alpha_{j+1} - alpha_j).

    Returns:
        Ordered thresholds alpha of length K-1 where
        alpha_1 < alpha_2 < ... < alpha_{K-1}.
    """
    n_thresh = len(raw)
    alpha = np.empty(n_thresh)
    alpha[0] = raw[0]
    for j in range(1, n_thresh):
        alpha[j] = alpha[j - 1] + np.exp(raw[j])
    return alpha


def thresholds_to_raw(alpha: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """
    Convert ordered thresholds to unconstrained raw parameters.

    Args:
        alpha: Ordered thresholds of length K-1, strictly increasing.

    Returns:
        Unconstrained parameters of length K-1.
    """
    n_thresh = len(alpha)
    raw = np.empty(n_thresh)
    raw[0] = alpha[0]
    for j in range(1, n_thresh):
        diff = alpha[j] - alpha[j - 1]
        raw[j] = np.log(max(diff, 1e-15))
    return raw


# -- Negative log-likelihood -----------------------------------------------

def cumulative_negloglik(
    params: NDArray[np.floating[Any]],
    y_codes: NDArray[np.integer[Any]],
    X: NDArray[np.floating[Any]],
    link: Link,
    n_levels: int,
) -> float:
    """
    Negative log-likelihood for the cumulative link model.

    The model is P(Y <= j | x) = g^{-1}(alpha_j - x'beta), where g^{-1}
    is the inverse link function. Category probabilities are obtained by
    differencing cumulative probabilities.

    Args:
        params: Concatenated parameter vector [raw_thresholds, beta],
            length (K-1 + p). Raw thresholds use the unconstrained
            parameterization.
        y_codes: Integer response codes 0, 1, ..., K-1 of length n.
        X: Design matrix of shape (n, p). Must NOT include an intercept.
        link: Link function instance (LogitLink, ProbitLink, CLogLogLink).
        n_levels: Number of ordered categories K.

    Returns:
        Negative log-likelihood (scalar). Minimizing this fits the model.
    """
    n_thresh = n_levels - 1
    raw_thresh = params[:n_thresh]
    beta = params[n_thresh:]

    alpha = raw_to_thresholds(raw_thresh)
    eta = X @ beta  # (n,)

    # Previously this function contained an n-long Python loop that
    # called `link.linkinv(np.atleast_1d(scalar))` twice per observation
    # to fill in `prob[i]` one element at a time. On MASS::housing
    # (n=1681) that was 3362 scalar linkinv calls × ~30 optimizer steps =
    # ~100k scalar calls per fit, each paying numpy's per-call overhead.
    # `_cumulative_probs_vectorized` already computes the full (n, K)
    # category-probability matrix in one shot; just index it.
    cat_probs = _cumulative_probs_vectorized(alpha, eta, link, n_levels)
    prob = cat_probs[np.arange(len(y_codes)), y_codes]
    prob = np.maximum(prob, 1e-15)
    return -np.sum(np.log(prob))


def _cumulative_probs_vectorized(
    alpha: NDArray[np.floating[Any]],
    eta: NDArray[np.floating[Any]],
    link: Link,
    n_levels: int,
) -> NDArray[np.floating[Any]]:
    """
    Compute category probabilities for all observations (vectorized).

    Args:
        alpha: Ordered thresholds, shape (K-1,).
        eta: Linear predictor x'beta, shape (n,).
        link: Link function instance.
        n_levels: Number of categories K.

    Returns:
        Category probabilities, shape (n, K). Row i gives
        P(Y=0|x_i), P(Y=1|x_i), ..., P(Y=K-1|x_i).
    """
    n = len(eta)
    n_thresh = n_levels - 1

    # Cumulative probabilities F(alpha_j - eta) for j = 0, ..., K-2
    # Shape: (n, K-1)
    cum_args = alpha[np.newaxis, :] - eta[:, np.newaxis]  # (n, K-1)
    cum_probs = link.linkinv(cum_args)  # (n, K-1)

    # Category probabilities by differencing
    # P(Y=0) = cum[0], P(Y=j) = cum[j] - cum[j-1], P(Y=K-1) = 1 - cum[K-2]
    cat_probs = np.empty((n, n_levels))
    cat_probs[:, 0] = cum_probs[:, 0]
    for j in range(1, n_thresh):
        cat_probs[:, j] = cum_probs[:, j] - cum_probs[:, j - 1]
    cat_probs[:, n_levels - 1] = 1.0 - cum_probs[:, n_thresh - 1]

    return cat_probs


# -- Gradient of negative log-likelihood -----------------------------------

def cumulative_gradient(
    params: NDArray[np.floating[Any]],
    y_codes: NDArray[np.integer[Any]],
    X: NDArray[np.floating[Any]],
    link: Link,
    n_levels: int,
) -> NDArray[np.floating[Any]]:
    """
    Gradient of the negative log-likelihood w.r.t. [raw_thresholds, beta].

    Uses the chain rule through the unconstrained threshold parameterization
    and the link function derivative (mu_eta).

    Args:
        params: Concatenated [raw_thresholds, beta], length (K-1 + p).
        y_codes: Integer response codes 0, ..., K-1, length n.
        X: Design matrix (n, p), no intercept.
        link: Link function instance.
        n_levels: Number of categories K.

    Returns:
        Gradient vector of length (K-1 + p).
    """
    n_thresh = n_levels - 1
    n_params = len(params)
    p = n_params - n_thresh

    raw_thresh = params[:n_thresh]
    beta = params[n_thresh:]

    alpha = raw_to_thresholds(raw_thresh)
    eta = X @ beta  # (n,)

    # Cumulative arguments and their derivatives
    # cum_arg[i, j] = alpha_j - eta_i
    cum_args = alpha[np.newaxis, :] - eta[:, np.newaxis]  # (n, K-1)
    cum_probs = link.linkinv(cum_args)  # F(alpha_j - eta_i)
    cum_deriv = link.mu_eta(cum_args)   # f(alpha_j - eta_i) = dF/d(arg)

    # Category probabilities
    cat_probs = _cumulative_probs_vectorized(alpha, eta, link, n_levels)
    cat_probs = np.maximum(cat_probs, 1e-15)

    n = len(y_codes)
    obs_idx = np.arange(n)
    # w_i = 1 / P(Y = y_i | x_i)
    w = 1.0 / cat_probs[obs_idx, y_codes]

    # Gradient w.r.t. alpha_j (ordered thresholds):
    # d(-loglik)/d(alpha_j) = -sum over i where y_i involves alpha_j
    # For threshold j: it appears in P(Y=j) and P(Y=j+1)
    # d P(Y=j)/d(alpha_j) = f(alpha_j - eta_i)   [positive contribution]
    # d P(Y=j+1)/d(alpha_j) = -f(alpha_j - eta_i) [negative contribution]
    grad_alpha = np.zeros(n_thresh)
    for j in range(n_thresh):
        # Observations where y_i == j: dP(Y=j)/d(alpha_j) = +f(alpha_j - eta_i)
        mask_j = (y_codes == j)
        if np.any(mask_j):
            grad_alpha[j] += np.sum(w[mask_j] * cum_deriv[mask_j, j])
        # Observations where y_i == j+1: dP(Y=j+1)/d(alpha_j) = -f(alpha_j - eta_i)
        mask_j1 = (y_codes == j + 1)
        if np.any(mask_j1):
            grad_alpha[j] -= np.sum(w[mask_j1] * cum_deriv[mask_j1, j])

    # Negate for negative log-likelihood
    grad_alpha = -grad_alpha

    # Chain rule: d/d(raw) via Jacobian of raw -> alpha transform
    # d(alpha_j)/d(raw_k):
    #   For k=0: affects alpha_0, alpha_1, ..., alpha_{K-2} (all >= k)
    #   For k>=1: d(alpha_j)/d(raw_k) = exp(raw_k) for j >= k, else 0
    grad_raw = np.zeros(n_thresh)
    # Cumulative sum from the end: grad_raw[k] = sum_{j>=k} grad_alpha[j] * d(alpha_j)/d(raw_k)
    # For k=0: d(alpha_j)/d(raw_0) = 1 for all j >= 0
    # For k>=1: d(alpha_j)/d(raw_k) = exp(raw_k) for all j >= k
    cumsum_grad_alpha = np.cumsum(grad_alpha[::-1])[::-1]
    grad_raw[0] = cumsum_grad_alpha[0]
    for k in range(1, n_thresh):
        grad_raw[k] = cumsum_grad_alpha[k] * np.exp(raw_thresh[k])

    # Gradient w.r.t. beta:
    # dP(Y=j|x)/d(beta) = -f(alpha_j - eta) * x  +  f(alpha_{j-1} - eta) * x
    # (negative sign because eta = x'beta enters as alpha_j - x'beta)
    grad_beta = np.zeros(p)
    for j in range(n_levels):
        mask = (y_codes == j)
        if not np.any(mask):
            continue
        w_masked = w[mask]
        X_masked = X[mask]

        # Upper boundary contribution: -f(alpha_j - eta) * x  (for j < K-1)
        if j < n_thresh:
            contrib_upper = -cum_deriv[mask, j]
        else:
            contrib_upper = np.zeros(np.sum(mask))

        # Lower boundary contribution: +f(alpha_{j-1} - eta) * x  (for j > 0)
        if j > 0:
            contrib_lower = cum_deriv[mask, j - 1]
        else:
            contrib_lower = np.zeros(np.sum(mask))

        grad_beta += X_masked.T @ (w_masked * (contrib_upper + contrib_lower))

    # Negate for negative log-likelihood
    grad_beta = -grad_beta

    return np.concatenate([grad_raw, grad_beta])
