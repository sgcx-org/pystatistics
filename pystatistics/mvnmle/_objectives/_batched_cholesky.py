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

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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
        raise ValueError("no missingness patterns to build constants from")
    P = len(patterns)
    v_obs_max = max(len(p.observed_indices) for p in patterns)
    if v_obs_max == 0:
        raise ValueError("every pattern has zero observed variables")

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


def _tri_inv_blocked(torch, L):
    """Inverse of a batched lower-triangular matrix using matmul only.

    Divide and conquer: inv([[A,0],[B,C]]) = [[A^-1, 0], [-C^-1 B A^-1, C^-1]],
    recursing to a closed-form 2x2 (and 1x1) base. This touches no triangular
    solve / inverse kernel — only matmul, slicing and elementwise ops — which is
    the operation set Apple Metal (MPS) executes fast (its triangular-solve and
    inverse kernels are ~300x slower than its batched matmul/Cholesky). It is
    numerically stable (back-substitution-equivalent; no growing matrix powers)
    and exact to floating precision, validated across condition numbers up to
    correlation 0.99. ``L`` is (..., v, v) lower-triangular with positive
    diagonal.
    """
    v = L.shape[-1]
    if v == 1:
        return 1.0 / L
    if v == 2:
        inv_a = 1.0 / L[..., 0, 0]
        inv_c = 1.0 / L[..., 1, 1]
        off = -L[..., 1, 0] * inv_a * inv_c
        zero = torch.zeros_like(inv_a)
        row0 = torch.stack([inv_a, zero], dim=-1)
        row1 = torch.stack([off, inv_c], dim=-1)
        return torch.stack([row0, row1], dim=-2)
    h = v // 2
    A = L[..., :h, :h].contiguous()
    C = L[..., h:, h:].contiguous()
    B = L[..., h:, :h].contiguous()
    Ai = _tri_inv_blocked(torch, A)
    Ci = _tri_inv_blocked(torch, C)
    BL = -Ci @ (B @ Ai)
    z = torch.zeros(L.shape[:-2] + (h, v - h), device=L.device, dtype=L.dtype)
    return torch.cat([torch.cat([Ai, z], dim=-1),
                      torch.cat([BL, Ci], dim=-1)], dim=-2)


def _trace_sigma_inv_m(torch, L, M):
    """tr(Sigma_k^{-1} M_k) for the whole batch, Sigma_k = L_k L_k^T.

    On MPS, form Sigma^{-1} via the matmul-only blocked inverse (Metal's
    triangular solve is pathologically slow). Elsewhere (CUDA/CPU) use two
    triangular solves, which are well optimised there.
    """
    if L.device.type == "mps":
        W = _tri_inv_blocked(torch, L)                  # L^{-1}
        sigma_inv = W.transpose(-1, -2) @ W             # Sigma^{-1} = L^-T L^-1
        return (sigma_inv * M).sum((-2, -1))
    Y = torch.linalg.solve_triangular(L, M, upper=False)
    X = torch.linalg.solve_triangular(L.transpose(-1, -2), Y, upper=True)
    return torch.diagonal(X, dim1=-2, dim2=-1).sum(-1)


def batched_neg2_loglik(torch, mu, sigma, consts: dict, eps: float):
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
    trace = _trace_sigma_inv_m(torch, L, M)                 # (P,)

    return (n_k * logdet + trace).sum()
