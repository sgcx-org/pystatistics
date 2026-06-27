"""Batched per-pattern objective for the GPU forward-Cholesky MLE.

One job: evaluate the missing-data multivariate-normal objective
``f = sum_k [ n_k * log|Sigma_k| + tr(Sigma_k^{-1} M_k) ]`` over *all*
missingness patterns at once, with a single batched Cholesky and a single
batched triangular solve, instead of a Python loop over patterns.

This is the batched replacement for the per-pattern loop that previously lived
in ``gpu_fp32.py`` / ``gpu_fp64.py``. The math is identical; only the execution
is vectorised (the loop made the GPU launch one tiny kernel per pattern, which
dominated runtime once the pattern count grew into the thousands).

Key trick — pattern contributions depend on the parameters only through ``mu``
and ``Sigma``; the per-pattern data enters solely through fixed sufficient
statistics that are precomputed once:

    n_k    : number of observations with pattern k
    ybar_k : mean of the observed rows for pattern k        (padded to v_obs_max)
    T2_k   : sum_i y_i y_i^T over pattern k's observed rows (padded to v_obs_max)

so that, writing delta_k = ybar_k - mu_k,

    M_k = sum_i (y_i - mu_k)(y_i - mu_k)^T = C_k + n_k delta_k delta_k^T,
    C_k = sum_i (y_i - ybar_k)(y_i - ybar_k)^T   (the centered scatter).

C_k is precomputed once in FP64; delta_k is small (both terms are on the same
scale), so this avoids the catastrophic cancellation that the raw form
``T2_k - n_k mu_k mu_k^T`` suffers in FP32 — important on the FP32 (Metal /
consumer-GPU) path. It is also cheaper: one outer product instead of three.

Padding: observed sub-blocks are placed in the top-left of a ``(P, v, v)`` batch.
Padded rows/cols of ``Sigma_k`` get a unit diagonal (so the batched Cholesky sees
identity blocks → ``log|I| = 0`` and no contribution), and padded entries of
``M_k`` are zero (so the trace term ignores them). No masking of the reductions
is then required.
"""

from __future__ import annotations

from pystatistics.core.exceptions import ValidationError

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.compute.linalg import (
    batched_tri_inv_series,
    use_blocked_inverse,
)


@dataclass(frozen=True)
class BatchedConstants:
    """Per-pattern sufficient statistics, padded to ``v_obs_max`` (NumPy)."""

    obs_idx: NDArray[np.int64]    # (P, v_obs_max) column indices, padded with 0
    obs_mask: NDArray[np.bool_]   # (P, v_obs_max) True for real observed slots
    n_k: NDArray[np.float64]      # (P,) observations per pattern
    ybar: NDArray[np.float64]     # (P, v_obs_max) observed means, padded with 0
    c: NDArray[np.float64]        # (P, v_obs_max, v_obs_max) centered scatter
    v_obs_max: int


def build_batched_constants(patterns, n_vars: int) -> BatchedConstants:
    """Precompute padded per-pattern sufficient statistics from PatternData.

    Patterns with zero observed variables are kept as all-padding rows (they
    contribute nothing); this keeps indexing uniform.

    Raises
    ------
    ValueError
        If there are no patterns, or every pattern has zero observed variables.
    """
    if len(patterns) == 0:
        raise ValidationError("no missingness patterns to build constants from")
    P = len(patterns)
    v_obs_max = max(len(p.observed_indices) for p in patterns)
    if v_obs_max == 0:
        raise ValidationError("every pattern has zero observed variables")

    obs_idx = np.zeros((P, v_obs_max), dtype=np.int64)
    obs_mask = np.zeros((P, v_obs_max), dtype=bool)
    n_k = np.zeros(P, dtype=np.float64)
    ybar = np.zeros((P, v_obs_max), dtype=np.float64)
    c = np.zeros((P, v_obs_max, v_obs_max), dtype=np.float64)

    for k, pat in enumerate(patterns):
        v = len(pat.observed_indices)
        if v == 0:
            continue
        n = float(pat.n_obs)
        d = np.asarray(pat.data, dtype=np.float64)  # (n_obs, v)
        mean = d.sum(axis=0) / n
        dc = d - mean                               # centered (no cancellation)
        obs_idx[k, :v] = pat.observed_indices
        obs_mask[k, :v] = True
        n_k[k] = n
        ybar[k, :v] = mean
        c[k, :v, :v] = dc.T @ dc

    return BatchedConstants(obs_idx=obs_idx, obs_mask=obs_mask, n_k=n_k,
                            ybar=ybar, c=c, v_obs_max=v_obs_max)


def to_torch(consts: BatchedConstants, torch, device, dtype) -> dict:
    """Move constants onto ``device`` with compute ``dtype`` (idx stays long)."""
    return {
        "obs_idx": torch.as_tensor(consts.obs_idx, device=device, dtype=torch.long),
        "obs_mask": torch.as_tensor(consts.obs_mask, device=device, dtype=torch.bool),
        "n_k": torch.as_tensor(consts.n_k, device=device, dtype=dtype),
        "ybar": torch.as_tensor(consts.ybar, device=device, dtype=dtype),
        "c": torch.as_tensor(consts.c, device=device, dtype=dtype),
    }


def unpack_cholesky(torch, theta, n_vars: int):
    """Reconstruct ``(mu, Sigma)`` from a standard-Cholesky parameter vector.

    The single torch implementation of the standard-Cholesky reconstruction,
    shared by the FP32 and FP64 GPU objectives so the two cannot drift apart.
    It reproduces ``CholeskyParameterization.unpack`` (the canonical NumPy
    reference) up to floating-point rounding, in whatever dtype/device ``theta``
    carries, while preserving the autograd graph.

    Parameter layout (matching the canonical parameterization)::

        theta = [ mu (n) | log(diag L) (n) | offdiag(L) ]

    The off-diagonal block is laid out **row-major** — the ordering produced by
    ``numpy.tril_indices(n, k=-1)`` and, identically, by
    ``torch.tril_indices(n, n, offset=-1)``. The reconstruction uses the torch
    form so the placement is the same as the reference; a hand-rolled
    column-major loop here is what previously made the FP64 path disagree with
    the canonical Sigma for ``n_vars >= 3``.

    L is built functionally (``diag_embed`` for the diagonal, out-of-place
    ``index_put`` for the off-diagonals) rather than via ``.diagonal().copy_()``
    plus in-place assignment, so there is no reliance on view/in-place aliasing
    and the result is safe to differentiate through.

    Parameters
    ----------
    torch : module
        The imported ``torch`` module (injected; no hidden import).
    theta : torch.Tensor, shape (n_params,)
        Standard-Cholesky parameter vector on the active device/dtype.
    n_vars : int
        Number of variables (dimension of Sigma).

    Returns
    -------
    (mu, sigma) : tuple of torch.Tensor
        ``mu`` shape ``(n_vars,)``; ``sigma`` shape ``(n_vars, n_vars)``,
        symmetric.
    """
    n = n_vars
    mu = theta[:n]

    # Diagonal of L (exponentiated for an unconstrained, positive parameter).
    L = torch.diag_embed(torch.exp(theta[n:2 * n]))

    # Off-diagonals: row-major placement matching the canonical reference.
    if n > 1:
        tril = torch.tril_indices(n, n, offset=-1, device=theta.device)
        L = L.index_put((tril[0], tril[1]), theta[2 * n:])

    sigma = L @ L.T
    return mu, 0.5 * (sigma + sigma.T)


def _batched_cholesky_with_ridge(torch, sigma_b, eps: float, max_tries: int = 5):
    """Batched Cholesky, escalating a ridge on any non-PD matrix in the batch.

    Uses ``cholesky_ex`` (no exception) to detect failures per matrix, then adds
    an escalating ridge to the *whole* batch diagonal and retries — mirroring the
    EM backend's ridge fallback for indefinite per-pattern submatrices.
    """
    ridge = eps
    P, v, _ = sigma_b.shape
    eye = torch.eye(v, device=sigma_b.device, dtype=sigma_b.dtype)
    L, info = torch.linalg.cholesky_ex(sigma_b)
    for _ in range(max_tries):
        if not bool((info > 0).any()):
            return L
        ridge *= 10.0
        sigma_b = sigma_b + ridge * eye
        L, info = torch.linalg.cholesky_ex(sigma_b)
    raise np.linalg.LinAlgError(
        f"batched Cholesky failed for {int((info > 0).sum())}/{P} patterns "
        f"after ridge escalation to {ridge:.2e}")


def _trace_sigma_inv_m(torch, L, M, method: str = "auto"):
    """tr(Sigma_k^{-1} M_k) for the whole batch, Sigma_k = L_k L_k^T.

    When ``use_blocked_inverse`` (MPS by default), form Sigma^{-1} via the matmul-only
    blocked inverse (Metal's triangular solve is pathologically slow); otherwise
    use two triangular solves (well optimised on CUDA/CPU). ``method`` forces the
    choice for ablation.
    """
    if use_blocked_inverse(L, method):
        W = batched_tri_inv_series(L)                           # L^{-1}
        sigma_inv = W.transpose(-1, -2) @ W             # Sigma^{-1} = L^-T L^-1
        return (sigma_inv * M).sum((-2, -1))
    Y = torch.linalg.solve_triangular(L, M, upper=False)
    X = torch.linalg.solve_triangular(L.transpose(-1, -2), Y, upper=True)
    return torch.diagonal(X, dim1=-2, dim2=-1).sum(-1)


def _sigma_inv(torch, L, method: str = "auto"):
    """Explicit per-pattern inverse covariance from its Cholesky factor.

    Uses the matmul-only series inverse when ``use_blocked_inverse`` (Metal's
    triangular-solve / inverse family is slow); otherwise inverts the triangular
    factor with ``solve_triangular`` against the identity and forms
    Sigma^{-1} = L^{-T} L^{-1}. (``cholesky_inverse`` is not implemented on MPS,
    so this portable solve-family form is used on the non-blocked path.)
    """
    if use_blocked_inverse(L, method):
        W = batched_tri_inv_series(L)
        return W.transpose(-1, -2) @ W
    eye = torch.eye(L.shape[-1], device=L.device, dtype=L.dtype).expand_as(L).contiguous()
    Linv = torch.linalg.solve_triangular(L, eye, upper=False)
    return Linv.transpose(-1, -2) @ Linv


def analytic_value_and_gradient(torch, theta, unpack, consts: dict, eps: float,
                                chunk_size=None, method: str = "auto"):
    """Objective value AND gradient of ``-2 log L`` in a *single* device pass.

    The per-pattern objective $n_k\\log|\\Sigma_k| + \\mathrm{tr}(\\Sigma_k^{-1}M_k)$
    has the closed-form partials
        dF/dSigma_k = n_k Sigma_k^{-1} - Sigma_k^{-1} M_k Sigma_k^{-1},
        dF/dmu_k    = -2 n_k Sigma_k^{-1} (ybar_k - mu_k),
    both expressible from the *forward* inverse covariance alone. We form these,
    then backpropagate them through only the cheap, autodiff-friendly part of the
    map---the gather of the observed sub-blocks and the ``theta -> (mu, Sigma)``
    reconstruction---so no gradient flows through ``cholesky`` or the matrix
    inverse. On Metal this replaces a ~20s Cholesky-backward (at p=100, tens of
    thousands of patterns) with milliseconds; results are identical to
    reverse-mode autodiff.

    The objective value is read off the SAME per-chunk intermediates the gradient
    already forms: ``logdet`` from the Cholesky factor ``L`` and the trace from the
    explicit ``Sinv`` (``tr(Sigma^{-1} M) = sum(Sinv * M)``), so it costs only two
    extra reductions and no extra factorisation. Returning both lets the optimiser
    driver fold value and gradient into one host<->device sync per evaluation
    instead of two (the dominant end-to-end cost on MPS at large p).

    ``unpack`` is a callable ``theta -> (mu, sigma)``. Returns ``(value, grad)`` as
    device tensors (a 0-d scalar and a ``(n_params,)`` vector) so the caller can
    coalesce them into a single device->host transfer.
    """
    P = consts["obs_idx"].shape[0]
    total_grad = None
    total_val = None
    for s, e in chunk_bounds(P, chunk_size):
        cc = _slice_consts(consts, s, e)
        idx = cc["obs_idx"]
        mask = cc["obs_mask"]
        n_k = cc["n_k"]
        ybar = cc["ybar"]
        c = cc["c"]

        mu, sigma = unpack(theta)                 # autodiff-tracked (cheap map)
        mu_k = mu[idx]                             # (Pc, v) tracked
        sig = sigma[idx[:, :, None], idx[:, None, :]]   # (Pc, v, v) tracked
        dtype = sigma.dtype
        maskf = mask.to(dtype)
        mo = (mask[:, :, None] & mask[:, None, :]).to(dtype)

        # Closed-form upstream gradients (no autodiff through chol/inverse).
        with torch.no_grad():
            mu_kd = mu_k * maskf
            sig_full = sig * mo + torch.diag_embed(eps * maskf + (1.0 - maskf))
            delta = (ybar - mu_kd) * maskf
            nb = n_k.view(-1, 1, 1)
            M = (c + nb * (delta[:, :, None] * delta[:, None, :])) * mo
            L = _batched_cholesky_with_ridge(torch, sig_full, eps)
            Sinv = _sigma_inv(torch, L, method)
            g_sig = (nb * Sinv - (Sinv @ M) @ Sinv) * mo            # dF/d(gathered sig)
            g_mu = (-2.0 * n_k.view(-1, 1)
                    * (Sinv @ delta[:, :, None]).squeeze(-1)) * maskf  # dF/d(gathered mu_k)
            # Objective value from the same intermediates (no extra factorisation):
            #   F = sum_k [ n_k log|Sigma_k| + tr(Sigma_k^{-1} M_k) ].
            logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
            val = (n_k * logdet + (Sinv * M).sum((-2, -1))).sum()

        g = torch.autograd.grad(outputs=[sig, mu_k], inputs=theta,
                                grad_outputs=[g_sig, g_mu])[0]
        total_grad = g if total_grad is None else total_grad + g
        total_val = val if total_val is None else total_val + val
    return total_val, total_grad


def analytic_gradient(torch, theta, unpack, consts: dict, eps: float,
                      chunk_size=None, method: str = "auto"):
    """Gradient of ``-2 log L`` w.r.t. ``theta`` via the closed-form matrix
    gradient, avoiding automatic differentiation through ``cholesky``.

    Thin wrapper over :func:`analytic_value_and_gradient` that discards the
    objective value, kept as the gradient-only entry point for callers that do
    not also need the value (e.g. the standalone ``compute_gradient`` path and the
    Hessian setup). The two share one implementation so the closed-form gradient
    cannot drift between them. ``unpack`` is a callable ``theta -> (mu, sigma)``.
    Returns the gradient tensor.
    """
    _, grad = analytic_value_and_gradient(
        torch, theta, unpack, consts, eps, chunk_size, method)
    return grad


def batched_neg2_loglik(torch, mu, sigma, consts: dict, eps: float,
                        method: str = "auto"):
    """Return the scalar ``-2 log L`` summed over all patterns (differentiable).

    Parameters
    ----------
    mu : (n_vars,) tensor   — current mean
    sigma : (n_vars, n_vars) tensor — current covariance
    consts : dict of torch tensors from :func:`to_torch`
    eps : float — diagonal jitter added to the real observed block (FP32/FP64)
    """
    idx = consts["obs_idx"]            # (P, v)
    mask = consts["obs_mask"]          # (P, v)
    n_k = consts["n_k"]                # (P,)
    ybar = consts["ybar"]              # (P, v)
    c = consts["c"]                    # (P, v, v) centered scatter
    dtype = sigma.dtype
    P, v = idx.shape

    maskf = mask.to(dtype)                                  # (P, v)
    mask_outer = (mask[:, :, None] & mask[:, None, :]).to(dtype)  # (P, v, v)

    # Gather mu_k and the observed sub-blocks of Sigma; zero the padding.
    mu_k = mu[idx] * maskf                                  # (P, v)
    sig = sigma[idx[:, :, None], idx[:, None, :]] * mask_outer  # (P, v, v)
    # Real diagonal gets +eps (matches the looped objective); padded diagonal
    # gets +1 so the batched Cholesky sees identity on padding.
    diag_add = eps * maskf + (1.0 - maskf)                  # (P, v)
    sig = sig + torch.diag_embed(diag_add)

    # M_k = C_k + n_k delta delta^T, delta = ybar - mu_k (cancellation-free).
    delta = (ybar - mu_k) * maskf                           # (P, v)
    nb = n_k.view(P, 1, 1)
    M = (c + nb * (delta[:, :, None] * delta[:, None, :])) * mask_outer  # (P, v, v)

    L = _batched_cholesky_with_ridge(torch, sig, eps)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)  # (P,)
    trace = _trace_sigma_inv_m(torch, L, M, method)         # (P,)

    return (n_k * logdet + trace).sum()


# --- Pattern chunking -------------------------------------------------------
#
# The objective is a sum over patterns, and its per-pattern memory is O(P*v^2)
# because the batched tensors (sig, M, L, ...) span all P patterns at once. For
# wide data with many distinct patterns (e.g. p=100 survey data with tens of
# thousands of patterns) that can exceed GPU memory. Since the sum decomposes
# over patterns, we evaluate it in chunks of patterns, bounding peak memory to
# one chunk's tensors. For the gradient we accumulate per chunk
# (grad of a sum = sum of grads), freeing each chunk's autograd graph before the
# next, so the gradient is memory-bounded too.

def chunk_bounds(n_patterns: int, chunk_size):
    """Return [(start, end), ...] partitioning patterns into chunks.

    ``chunk_size`` falsy/non-positive/>= n_patterns means a single chunk
    (no chunking).
    """
    if not chunk_size or chunk_size <= 0 or chunk_size >= n_patterns:
        return [(0, n_patterns)]
    return [(s, min(s + chunk_size, n_patterns))
            for s in range(0, n_patterns, chunk_size)]


def _slice_consts(consts: dict, s: int, e: int) -> dict:
    return {k: v[s:e] for k, v in consts.items()}


def auto_chunk_size(n_vars: int, dtype_bytes: int, budget_bytes: int = 1 << 31):
    """Patterns per chunk so a chunk's batched (chunk, v, v) tensors — and their
    autograd-saved copies — fit within ``budget_bytes`` (default 2 GiB).

    Conservative: assumes ~12 simultaneous (chunk, v, v) buffers. Returns a
    positive int; for small ``v`` it is large enough that typical pattern counts
    fall in a single chunk (no looping overhead).
    """
    per_pattern = max(1, n_vars * n_vars * dtype_bytes * 12)
    return max(1, int(budget_bytes // per_pattern))


def objective_value(torch, mu, sigma, consts: dict, eps: float, chunk_size=None,
                    method: str = "auto"):
    """Sum ``-2 log L`` over all patterns, evaluated in pattern chunks.

    Intended for the forward (value) path under ``torch.no_grad()``; peak memory
    is one chunk's tensors.
    """
    P = consts["obs_idx"].shape[0]
    total = None
    for s, e in chunk_bounds(P, chunk_size):
        part = batched_neg2_loglik(torch, mu, sigma, _slice_consts(consts, s, e),
                                   eps, method)
        total = part if total is None else total + part
    return total


def accumulate_gradient(torch, theta_gpu, unpack, consts: dict, eps: float,
                        chunk_size=None, method: str = "auto"):
    """Gradient of ``-2 log L`` w.r.t. ``theta_gpu`` via per-chunk backprop.

    Each chunk re-unpacks ``theta_gpu`` to ``(mu, sigma)``, computes its own
    contribution, and backpropagates it — accumulating into ``theta_gpu.grad``
    and freeing that chunk's graph before the next. Peak memory is one chunk's
    batched tensors rather than the whole pattern set. ``unpack`` is a callable
    ``theta -> (mu, sigma)``. Returns ``(grad_tensor, objective_value_float)``.
    """
    P = consts["obs_idx"].shape[0]
    theta_gpu.grad = None
    obj = 0.0
    for s, e in chunk_bounds(P, chunk_size):
        mu, sigma = unpack(theta_gpu)
        part = batched_neg2_loglik(torch, mu, sigma, _slice_consts(consts, s, e),
                                   eps, method)
        part.backward()
        obj += float(part.detach())
    return theta_gpu.grad, obj
