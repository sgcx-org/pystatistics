"""
Triangular-factor inversion for batched GPU linear algebra.

Apple Metal (MPS) executes batched ``matmul``/``cholesky`` fast, but its
triangular-solve and inverse kernels are pathologically slow (~100-300x the
matmul throughput on small batched systems). The portable remedy is to invert a
triangular Cholesky factor with a **matmul-only** block recursion and apply it by
matmul, instead of ``solve_triangular``/``inv``.

This primitive is shared: MVNMLE (the missing-data MVN objective) and MICE (the
batched imputation draw) both factor a small Gram/covariance per step and need
its inverse. Keeping one implementation here avoids re-deriving the trick per
module (it previously lived, siloed, inside ``mvnmle``).

Determinism: pure matmul/slice/elementwise — no RNG, no nondeterministic kernel.
"""

from __future__ import annotations


def use_blocked_inverse(L, method: str = "auto") -> bool:
    """Whether to use the matmul-only blocked inverse for factor ``L``.

    ``'auto'`` (default) uses it on MPS (where the triangular-solve/inverse family
    is slow) and ``solve_triangular`` elsewhere (well optimised on CUDA/CPU).
    ``'blocked'``/``'solve'`` force the choice (for benchmarking/ablation).
    """
    if method == "blocked":
        return True
    if method == "solve":
        return False
    return L.device.type == "mps"


def _tri_inv_series_value(L):
    """Value of ``L^-1`` for batched lower-triangular ``L`` via the exact finite
    Neumann series of its nilpotent part, summed by doubling, then one Newton
    step. Pure value (no autograd contract — see :func:`batched_tri_inv_series`).

    ``L = D(I+N)`` with ``N = D^-1·(strictly-lower part)`` nilpotent (``N^p = 0``),
    so ``(I+N)^-1 = sum_{k=0}^{p-1} (-N)^k`` is exact and finite; summed by
    doubling (``S_{2t} = S_t + P^t S_t``) it costs ~``log2(p)`` *large* matmuls. In
    FP32 the bare series loses accuracy for ill-conditioned ``L``; one Newton step
    ``X <- X(2I - L X)`` restores it to the FP32 floor (matching the block
    inverse) for condition numbers up to ~1e6.
    """
    import torch

    *batch, p, _ = L.shape
    dinv = 1.0 / torch.diagonal(L, dim1=-2, dim2=-1)              # (..., p)
    eye = torch.eye(p, dtype=L.dtype, device=L.device).expand(*batch, p, p)
    P = eye - dinv.unsqueeze(-1) * L                             # -(D^-1 L - I) = -N
    S = eye.clone()                                             # sum_{k=0}^{0}
    Pp = P
    t = 1
    while t < p:
        S = S + Pp @ S                                         # double the terms
        t *= 2
        if t < p:
            Pp = Pp @ Pp                                       # P^t (skip last, unused)
    X = S * dinv.unsqueeze(-2)                                  # (I+N)^-1 D^-1 = L^-1
    eye2 = 2.0 * torch.eye(p, dtype=L.dtype, device=L.device)
    return X @ (eye2 - L @ X)                                   # one Newton step


def batched_tri_inv_series(L):
    """Inverse of a batched lower-triangular ``L`` (positive diagonal), matmul-only
    and **autograd-safe**. ~log2(p) large batched matmuls (Neumann doubling of the
    nilpotent part + Newton), which is what makes it fast on MPS inside a per-step
    loop — MPS runs few large matmuls well, many tiny kernels (e.g. a block
    recursion, or `solve_triangular`) poorly.

    The value comes from :func:`_tri_inv_series_value` (series + Newton). The bare
    series/Newton expression is *not* differentiable to the true inverse gradient,
    so we wrap it in one **differentiable Newton step from a detached, already-
    accurate iterate** ``Yd``::

        Yd = L^-1 (computed graph-free)   ->   return Yd (2I - tril(L) Yd)

    The forward value is inverse-accurate (one more Newton from an accurate ``Yd``);
    and because ``Yd`` is constant, the gradient is exactly the matrix-inverse VJP
    ``-Yd^T ḡ Yd^T`` — correct without a custom ``autograd.Function`` (so the module
    keeps its lazy ``torch`` import). ``tril(L)`` confines the gradient to ``L``'s
    lower triangle, matching ``solve_triangular`` semantics for a triangular factor.
    """
    import torch

    Yd = _tri_inv_series_value(L.detach())                     # accurate L^-1, no graph
    eye = torch.eye(L.shape[-1], dtype=L.dtype, device=L.device)
    return Yd @ (2.0 * eye - torch.tril(L) @ Yd)               # differentiable Newton step
