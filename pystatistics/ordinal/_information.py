"""
Observed information and variance-covariance for the cumulative link model.

The optimizer works in the unconstrained *raw* (log-gap) threshold
parameterization (see ``_likelihood.py``); statisticians and R's
``MASS::polr`` report the variance-covariance in the *natural* threshold
parameterization ``[alpha_1, ..., alpha_{K-1}, beta]``. This module owns the
two jobs that connect them:

1. Build the observed information (Hessian of the negative log-likelihood)
   at a parameter point by forward-differencing the *analytic* gradient.
   This is what ``MASS::polr`` does (it finite-differences the deviance);
   differencing the analytic vector gradient costs ``d + 1`` gradient
   evaluations instead of the ``d * (d + 1)`` an element-wise
   ``approx_fprime`` Hessian pays, where ``d = K - 1 + p``.

2. Map that raw-coordinate covariance to natural threshold coordinates via
   the delta method, so the reported threshold standard errors — and the
   posterior draw the MICE ``polr`` method takes over ``[alpha, beta]`` —
   are on the same scale as ``MASS::polr``.

At the maximum-likelihood estimate the raw-coordinate Hessian satisfies
``H_raw = J^T H_nat J`` exactly (the nonlinear-reparameterization correction
term is proportional to the score, which vanishes at the optimum), so
``J H_raw^{-1} J^T`` equals the natural-coordinate observed-information
inverse — i.e. exactly what ``MASS::polr`` reports.

References
----------
    Venables, W. N. & Ripley, B. D. (2002). Modern Applied Statistics with S.
    Agresti, A. (2010). Analysis of Ordinal Categorical Data (2nd ed.)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatistics.regression.families import Link
from pystatistics.ordinal._likelihood import cumulative_gradient


def raw_to_natural_jacobian(
    raw_thresh: NDArray[np.floating[Any]],
    n_params: int,
) -> NDArray[np.floating[Any]]:
    """
    Jacobian d[alpha, beta] / d[raw_thresh, beta], shape (n_params, n_params).

    The raw -> natural threshold map is alpha_0 = raw_0,
    alpha_j = alpha_{j-1} + exp(raw_j) for j >= 1, so
    d(alpha_j)/d(raw_0) = 1 for all j, and d(alpha_j)/d(raw_k) = exp(raw_k)
    for j >= k (k >= 1), else 0. The slope block is the identity.

    Args:
        raw_thresh: Raw (unconstrained) threshold parameters, length K-1.
        n_params: Total parameter count K-1+p (thresholds then slopes).

    Returns:
        Lower-block-triangular Jacobian of shape (n_params, n_params).
    """
    n_thresh = len(raw_thresh)
    jac = np.eye(n_params)
    jac[:n_thresh, 0] = 1.0
    for k in range(1, n_thresh):
        jac[k:n_thresh, k] = np.exp(raw_thresh[k])
    return jac


def observed_information(
    params: NDArray[np.floating[Any]],
    y_codes: NDArray[np.integer[Any]],
    X: NDArray[np.floating[Any]],
    link: Link,
    n_levels: int,
    grad0: NDArray[np.floating[Any]] | None = None,
    ridge: float = 0.0,
) -> NDArray[np.floating[Any]]:
    """
    Observed information (Hessian of the negative log-likelihood) in raw
    coordinates, via forward differences of the analytic gradient.

    Args:
        params: Raw-parameterization point [raw_thresholds, beta], length d.
        y_codes: Integer response codes 0, ..., K-1, length n.
        X: Design matrix (n, p), no intercept.
        link: Link function instance.
        n_levels: Number of categories K.
        grad0: Analytic gradient at ``params``, if already computed. Recomputed
            when None. Must be the gradient at the *same* ``ridge`` so the
            forward difference is consistent.
        ridge: Optional L2 (ridge) penalty coefficient on the slopes beta. The
            penalty adds ``ridge`` to each diagonal entry of the beta block;
            because the Hessian is differenced from the penalized gradient, this
            happens automatically when ``ridge`` is threaded through. Default
            0.0 reproduces the unpenalized observed information.

    Returns:
        Symmetric (d, d) Hessian of the (penalized) negative log-likelihood.
    """
    d = len(params)
    if grad0 is None:
        grad0 = cumulative_gradient(params, y_codes, X, link, n_levels, ridge)

    # Same step scaling as scipy's approx_fprime, but applied to the analytic
    # vector gradient so the whole Hessian costs d gradient evaluations.
    step = np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(params))

    hess = np.empty((d, d))
    for k in range(d):
        shifted = params.copy()
        shifted[k] += step[k]
        grad_k = cumulative_gradient(shifted, y_codes, X, link, n_levels, ridge)
        hess[:, k] = (grad_k - grad0) / step[k]

    # Forward differences are not perfectly symmetric; symmetrize.
    return 0.5 * (hess + hess.T)


def vcov_natural(
    hess_raw: NDArray[np.floating[Any]],
    raw_thresh: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """
    Natural-coordinate variance-covariance from a raw-coordinate Hessian.

    Inverts the observed information and applies the delta-method transform to
    natural threshold coordinates, matching ``MASS::polr``'s reported vcov.

    Args:
        hess_raw: Observed information in raw coordinates, shape (d, d).
        raw_thresh: Raw threshold parameters at the estimate, length K-1.

    Returns:
        Variance-covariance in natural coordinates [alpha, beta], shape (d, d).

    Raises:
        numpy.linalg.LinAlgError: If the observed information is singular
            (e.g. under data separation) and cannot be inverted.
    """
    vcov_raw = np.linalg.inv(hess_raw)
    jac = raw_to_natural_jacobian(raw_thresh, hess_raw.shape[0])
    return jac @ vcov_raw @ jac.T
