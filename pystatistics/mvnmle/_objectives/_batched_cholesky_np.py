"""NumPy kernel for the forward-Cholesky missing-data MVN objective.

One job: evaluate

    F(mu, Sigma) = sum_k [ n_k log|Sigma_k| + tr(Sigma_k^{-1} M_k) ]   (= -2 log L
                                                   up to the 2*pi constant)

and its closed-form gradient with respect to the log-Cholesky parameters, over
all missingness patterns at once, in float64 NumPy. It is the PyTorch-free
counterpart of ``_batched_cholesky.py`` and consumes the same padded
per-pattern constants (:class:`BatchedConstants`), so both implementations
evaluate the same function of the same statistics.

Notation (pattern k observes the variable subset O_k, with selector E_k):

    Sigma   = L L^T,   L lower triangular with positive diagonal
    Sigma_k = E_k Sigma E_k^T          (principal submatrix on O_k)
    delta_k = ybar_k - E_k mu
    M_k     = C_k + n_k delta_k delta_k^T   (scatter about the current mean)
    S_k     = Sigma_k^{-1}

Gradient. With G = sum_k E_k^T (n_k S_k - S_k M_k S_k) E_k  (= dF/dSigma,
symmetric), the chain rule through Sigma = L L^T gives dF/dL = tril(2 G L).
The diagonal of L is parameterised as exp(.), so its entries are scaled by
diag(L). The mean gradient is  -2 sum_k n_k E_k^T S_k delta_k.

Padding. Observed sub-blocks sit in the top-left of a ``(P, v, v)`` batch.
Padded diagonal entries of Sigma_k are 1 and padded entries of M_k are 0, so
padding contributes log|I| = 0 to the log-determinant and nothing to the
trace; no masked reductions are needed.

Parameter layout (identical to ``CholeskyParameterization``):

    theta = [ mu (p) | log diag(L) (p) | offdiag(L), row-major lower triangle ]
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ._batched_cholesky import BatchedConstants, chunk_bounds

_MAX_RIDGE_TRIES = 5


def unpack_cholesky_np(theta: NDArray[np.float64], n_vars: int
                       ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split ``theta`` into the mean and the lower-triangular factor ``L``.

    Returns ``(mu, L)`` with ``L`` of shape ``(n_vars, n_vars)``; the
    covariance is ``L @ L.T``.
    """
    p = n_vars
    mu = theta[:p]
    L = np.zeros((p, p), dtype=np.float64)
    L[np.diag_indices(p)] = np.exp(theta[p:2 * p])
    if p > 1:
        L[np.tril_indices(p, k=-1)] = theta[2 * p:]
    return mu, L


def _batched_cholesky_with_ridge(sig: NDArray[np.float64], eps: float
                                 ) -> NDArray[np.float64]:
    """Batched Cholesky; on failure add an escalating ridge and retry.

    ``np.linalg.cholesky`` raises for the whole batch if any matrix is not
    positive definite, so the ridge (starting at ``10 * eps`` and growing
    tenfold per retry, cumulatively) is applied to every matrix in the batch
    — the same policy as the PyTorch kernel.

    Raises
    ------
    np.linalg.LinAlgError
        If the batch is still not positive definite after the last retry.
    """
    try:
        return np.linalg.cholesky(sig)
    except np.linalg.LinAlgError:
        pass
    ridge = eps
    eye = np.eye(sig.shape[-1], dtype=sig.dtype)
    for _ in range(_MAX_RIDGE_TRIES):
        ridge *= 10.0
        sig = sig + ridge * eye
        try:
            return np.linalg.cholesky(sig)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError(
        f"batched Cholesky failed for a batch of {sig.shape[0]} pattern "
        f"covariance blocks after ridge escalation to {ridge:.2e}")


def _batched_tril_inverse(L: NDArray[np.float64]) -> NDArray[np.float64]:
    """Invert a batch of lower-triangular matrices by forward substitution.

    Row i of ``W = L^{-1}`` is ``(e_i - L[i, :i] @ W[:i, :]) / L[i, i]``; the
    recursion runs over the (small) matrix dimension and is vectorised over
    the batch, so it costs ``v`` NumPy calls regardless of the pattern count.
    """
    v = L.shape[-1]
    W = np.zeros_like(L)
    diag = np.diagonal(L, axis1=-2, axis2=-1)          # (P, v)
    W[:, 0, 0] = 1.0 / diag[:, 0]
    for i in range(1, v):
        row = np.matmul(L[:, i:i + 1, :i], W[:, :i, :i])[:, 0, :]   # (P, i)
        W[:, i, :i] = -row / diag[:, i:i + 1]
        W[:, i, i] = 1.0 / diag[:, i]
    return W


def _chunk_terms(mu, sigma, consts: BatchedConstants, s: int, e: int,
                 eps: float):
    """Per-chunk Cholesky factors and the quantities built from them.

    Returns ``(n_k, delta, M, logdet, W, maskf, mask_outer)`` for patterns
    ``s:e``, where ``W = L_k^{-1}`` so that ``S_k = W^T W``.
    """
    idx = consts.obs_idx[s:e]
    mask = consts.obs_mask[s:e]
    n_k = consts.n_k[s:e]
    maskf = mask.astype(np.float64)
    mask_outer = maskf[:, :, None] * maskf[:, None, :]

    # Gather Sigma_k; padded slots get a unit diagonal, observed slots the
    # (theta-independent) scale-relative jitter eps * diag_scale.
    sig = sigma[idx[:, :, None], idx[:, None, :]] * mask_outer
    diag_add = eps * consts.diag_scale[s:e] + (1.0 - maskf)
    v = idx.shape[1]
    sig[:, np.arange(v), np.arange(v)] += diag_add

    delta = (consts.ybar[s:e] - mu[idx] * maskf) * maskf
    M = (consts.c[s:e]
         + n_k[:, None, None] * (delta[:, :, None] * delta[:, None, :])
         ) * mask_outer

    Lk = _batched_cholesky_with_ridge(sig, eps)
    logdet = 2.0 * np.log(np.diagonal(Lk, axis1=-2, axis2=-1)).sum(-1)
    W = _batched_tril_inverse(Lk)
    return n_k, delta, M, logdet, W, maskf, mask_outer


def neg2_loglik_np(theta: NDArray[np.float64], consts: BatchedConstants,
                   n_vars: int, eps: float,
                   chunk_size: Optional[int] = None) -> float:
    """Objective value ``F`` (= -2 log L without the 2*pi constant)."""
    mu, L = unpack_cholesky_np(theta, n_vars)
    sigma = L @ L.T
    total = 0.0
    for s, e in chunk_bounds(consts.obs_idx.shape[0], chunk_size):
        n_k, _, M, logdet, W, _, _ = _chunk_terms(mu, sigma, consts, s, e, eps)
        # tr(S_k M_k) = tr(W M W^T) = sum((W M) * W)
        trace = (np.matmul(W, M) * W).sum((-2, -1))
        total += float((n_k * logdet + trace).sum())
    return total


def value_and_gradient_np(theta: NDArray[np.float64], consts: BatchedConstants,
                          n_vars: int, eps: float,
                          chunk_size: Optional[int] = None
                          ) -> Tuple[float, NDArray[np.float64]]:
    """Objective value and its gradient with respect to ``theta``.

    Both come from the same per-chunk Cholesky factors, so the pair costs one
    factorisation per chunk.
    """
    p = n_vars
    mu, L = unpack_cholesky_np(theta, p)
    sigma = L @ L.T
    G = np.zeros(p * p, dtype=np.float64)       # dF/dSigma, flattened
    g_mu = np.zeros(p, dtype=np.float64)
    total = 0.0
    for s, e in chunk_bounds(consts.obs_idx.shape[0], chunk_size):
        n_k, delta, M, logdet, W, maskf, mask_outer = _chunk_terms(
            mu, sigma, consts, s, e, eps)
        Sinv = np.matmul(W.transpose(0, 2, 1), W)
        SM = np.matmul(Sinv, M)
        trace = np.trace(SM, axis1=-2, axis2=-1)
        total += float((n_k * logdet + trace).sum())

        G_k = (n_k[:, None, None] * Sinv - np.matmul(SM, Sinv)) * mask_outer
        gmu_k = (-2.0 * n_k[:, None]
                 * np.matmul(Sinv, delta[:, :, None])[:, :, 0]) * maskf

        # Scatter-add back to the full (p, p) / (p,) arrays. Padded slots
        # carry exact zeros, so their (arbitrary) index 0 receives nothing.
        idx = consts.obs_idx[s:e]
        flat = (idx[:, :, None] * p + idx[:, None, :]).ravel()
        G += np.bincount(flat, weights=G_k.ravel(), minlength=p * p)
        g_mu += np.bincount(idx.ravel(), weights=gmu_k.ravel(), minlength=p)

    G = G.reshape(p, p)
    dL = 2.0 * (G @ L)                           # lower triangle is dF/dL
    grad = np.empty_like(theta, dtype=np.float64)
    grad[:p] = g_mu
    grad[p:2 * p] = np.diagonal(dL) * np.diagonal(L)
    if p > 1:
        grad[2 * p:] = dL[np.tril_indices(p, k=-1)]
    return total, grad
