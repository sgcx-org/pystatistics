"""
Batched symmetric-positive-definite apply primitives for the GPU MICE methods.

The categorical methods (logreg, polyreg, polr) all reduce, per Newton/IRLS step
and for the posterior draw, to applying the inverse of a small SPD matrix
``G = L L'`` (a reweighted Gram or a multinomial block Hessian) batched over the
``m`` chains. The two operations they need are:

  * ``solve_spd(L, B) = G^{-1} B``        — the Newton/IRLS update and beta_hat
  * ``apply_inv_factor_T(L, z) = L^{-T} z`` — the draw factor ``A`` with ``A A' = G^{-1}``

Both are device-split through ``core.compute.linalg``: on MPS (whose triangular
solve is ~250x slower than its matmul) the factor is inverted with the matmul-
series inverse and applied by matmul; on CUDA/CPU the fast ``solve_triangular`` is
used directly. Keeping these here (rather than inside any one method module)
avoids coupling polyreg/polr to logreg for what is generic linear algebra.

Determinism: pure matmul / triangular solve — no RNG.
"""

from __future__ import annotations

from pystatistics.core.compute.linalg import (
    batched_tri_inv_series,
    use_blocked_inverse,
)


def solve_spd(L, B):
    """``G^{-1} B`` for SPD ``G = L L'`` (``L`` lower-triangular), batched over the
    leading dim. MPS: matmul-series inverse; CUDA/CPU: two triangular solves."""
    import torch

    if use_blocked_inverse(L):
        Linv = batched_tri_inv_series(L)
        return Linv.transpose(1, 2) @ (Linv @ B)
    fwd = torch.linalg.solve_triangular(L, B, upper=False)
    return torch.linalg.solve_triangular(L.transpose(1, 2), fwd, upper=True)


def apply_inv_factor_T(L, z):
    """``L^{-T} z`` — the posterior-draw factor ``A`` with ``A A' = G^{-1}``."""
    import torch

    if use_blocked_inverse(L):
        Linv = batched_tri_inv_series(L)
        return Linv.transpose(1, 2) @ z
    return torch.linalg.solve_triangular(L.transpose(1, 2), z, upper=True)
