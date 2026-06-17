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


def batched_tri_inv_series(L, refine_steps: int = 1):
    """Inverse of a batched lower-triangular ``L`` (positive diagonal) via the
    exact finite Neumann series of its nilpotent part, summed by doubling, then
    Newton-refined. A faster matmul-only alternative to :func:`batched_tri_inv`.

    ``L = D(I+N)`` with ``N = D^-1·(strictly-lower part)``, which is nilpotent
    (``N^p = 0``), so ``(I+N)^-1 = sum_{k=0}^{p-1} (-N)^k`` is exact and finite.
    Summed by doubling (``S_{2t} = S_t + P^t S_t``, ``P^{2t} = (P^t)^2``) it costs
    ~``log2(p)`` *large* batched matmuls — far fewer launches than the block
    recursion's ``O(p)`` tiny ops, which is what makes it fast on MPS inside a
    per-step loop (MPS executes few large matmuls well; many tiny kernels poorly).

    In FP32 the bare series loses accuracy for ill-conditioned ``L`` (cancellation
    among ``O(1)`` terms). One Newton step ``X <- X(2I - L X)`` (quadratic
    convergence) restores it to the FP32 floor, matching :func:`batched_tri_inv`;
    a single step suffices for condition numbers up to ~1e6 (beyond which the
    result is precision-limited, not iteration-limited). Pure matmul/elementwise —
    no triangular-solve/inverse kernel, no RNG.
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
    for _ in range(refine_steps):
        X = X @ (eye2 - L @ X)
    return X


def batched_tri_inv(L):
    """Inverse of a batched lower-triangular matrix using matmul only.

    Divide and conquer: ``inv([[A,0],[B,C]]) = [[A^-1, 0], [-C^-1 B A^-1, C^-1]]``,
    recursing to a closed-form 2x2 (and 1x1) base. Touches no triangular-solve or
    inverse kernel — only matmul, slicing and elementwise ops — which is the
    operation set MPS executes fast. Numerically stable (back-substitution-
    equivalent; no growing matrix powers) and exact to floating precision. ``L``
    is ``(..., v, v)`` lower-triangular with positive diagonal.
    """
    import torch

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
    Ai = batched_tri_inv(A)
    Ci = batched_tri_inv(C)
    BL = -Ci @ (B @ Ai)
    z = torch.zeros(L.shape[:-2] + (h, v - h), device=L.device, dtype=L.dtype)
    return torch.cat([torch.cat([Ai, z], dim=-1),
                      torch.cat([BL, Ci], dim=-1)], dim=-2)
