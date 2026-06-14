"""
Shared multivariate-normal posterior draw for the categorical methods.

The categorical imputation methods (logreg, polyreg, polr) all follow the same
Bayesian recipe R's ``mice`` uses: fit the model, then draw the parameter vector
once from its asymptotic posterior ``N(theta_hat, V)`` before predicting and
sampling. Drawing the parameters — rather than reusing the point estimate —
injects the between-imputation variability that makes multiple imputation
produce valid standard errors.

``V`` is a covariance (or inverse-information) matrix, positive definite in
theory but occasionally marginally indefinite in finite precision, so the
Cholesky factor is taken with the same jittered, eigenvalue-clipped fallback as
the numeric path (reused from ``_linreg``).
"""

from __future__ import annotations

import numpy as np

from pystatistics.mice.methods._linreg import _safe_cholesky


def mvn_draw(mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one sample from ``N(mean, cov)`` via its Cholesky factor.

    Parameters
    ----------
    mean : (d,) array
        Posterior mean (the point estimate).
    cov : (d, d) array
        Posterior covariance of the estimate.
    rng : numpy.random.Generator
        Sole randomness source.
    """
    mean = np.asarray(mean, dtype=np.float64).ravel()
    L = _safe_cholesky(np.asarray(cov, dtype=np.float64))
    return mean + L @ rng.standard_normal(mean.shape[0])


def sample_categories(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one class index per row from a (n, K) matrix of class probabilities.

    Robust to small negative probabilities (which can arise, e.g., from drawn
    ordinal thresholds that are not perfectly ordered): negatives are clipped and
    each row renormalised before inverse-CDF sampling.
    """
    probs = np.clip(np.asarray(probs, dtype=np.float64), 0.0, None)
    row_sums = probs.sum(axis=1, keepdims=True)
    # Degenerate rows (all-zero) fall back to uniform.
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    probs = probs / row_sums

    cdf = np.cumsum(probs, axis=1)
    u = rng.random(probs.shape[0])
    # Smallest index whose cumulative probability reaches u.
    return (cdf >= u[:, None]).argmax(axis=1).astype(np.intp)


def marginal_indices(
    y_obs: np.ndarray, n_mis: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample ``n_mis`` class indices from the observed empirical distribution.

    Used as the documented fallback when a categorical model fit fails to
    converge mid-sweep (see the categorical methods). It preserves the observed
    marginal distribution of the column; the next iteration retries the full
    conditional model, so the fallback is local and self-correcting.
    """
    return rng.choice(
        np.asarray(y_obs, dtype=np.intp), size=int(n_mis), replace=True
    )
