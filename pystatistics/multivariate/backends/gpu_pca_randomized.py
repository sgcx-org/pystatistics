"""Randomized truncated SVD PCA for GPU (Halko–Martinsson–Tropp).

This is the *only* genuinely on-device PCA path for Apple Silicon (Metal/MPS):
every X-sized operation — the random projection ``Y = XΩ``, the power-iteration
matmuls, and the CholeskyQR2 orthonormalization — runs on Metal through
``matmul`` + ``cholesky``, the two linear-algebra kernels Metal implements
natively. The only host work is a tiny ``l × l`` eigendecomposition where
``l = n_components + oversample ≈ 20`` — kilobytes and microseconds, a
deliberate step, not a silent fallback.

Why not ``torch.linalg.qr`` / ``svd`` / ``eigh`` on MPS (torch 2.x):
  - reduced-QR materializes a full ``n × n`` Q (≈149 GiB at n=200k) and OOMs;
  - ``svd`` silently falls back to the CPU;
  - ``eigh`` raises ``NotImplementedError``.
CholeskyQR2 orthonormalizes with ``matmul`` + an ``l × l`` Cholesky only, both
Metal-native, which is the whole reason a naive SVD-of-X port failed on MPS.

The algorithm is device-agnostic, so it is also selectable on CUDA (it does not
change, and does not alter, the CUDA ``svd``/``gram``/``auto`` defaults).

Accuracy / reproducibility
--------------------------
Randomized SVD is *approximate* (top-k only) and draws a random Gaussian
sketch. ``oversample`` and ``n_iter`` are the accuracy knobs; ``oversample=10``,
``n_iter=4`` reproduce the validated ~1e-7 error. The sketch is seeded
(injectable ``seed``, default fixed) so a given input yields the same result
every call (Coding Bible Rule 6).

Two-tier validation: GPU is validated against CPU at the ``GPU_FP32`` tier
(rtol 1e-4, atol 1e-5). The path was validated against the fp64 numpy reference
at ~1e-7 actual error across tall / wide / square shapes.
"""

from __future__ import annotations

import numpy as np

from pystatistics.core.exceptions import NumericalError
from pystatistics.multivariate._common import PCASolution
from pystatistics.multivariate.backends.gpu_pca import (
    _MIN_EIG_RATIO_FP32,
    _MIN_EIG_RATIO_FP64,
    _finalize,
    _fix_sign_convention_gpu,
    _prepare_X_gpu,
)


def _cholesky_qr(Y):
    """One Cholesky-QR pass: orthonormalize the columns of ``Y`` (n×l) on-device.

    Forms the ``l × l`` Gram ``G = YᵀY`` (matmul, Metal-native), Cholesky-factors
    it (Metal-native), and recovers ``Q = Y L⁻ᵀ`` by triangular solve, giving
    ``QᵀQ ≈ I``. A single fp32 pass can lose orthogonality (error up to ~1e-3 for
    large n or a wide sketch); :func:`_cholesky_qr2` runs it twice.

    A nearly-collinear sketch makes ``G`` only borderline positive-definite and
    the plain Cholesky can fail; we retry once with a small diagonal shift
    (shifted Cholesky-QR) so the orthonormalizer degrades gracefully instead of
    raising. The condition gate in :func:`_pca_gpu_randomized` is what refuses
    genuinely ill-conditioned data — the shift only covers round-off-level
    indefiniteness.
    """
    import torch

    G = Y.T @ Y                                          # l × l (matmul)
    try:
        L = torch.linalg.cholesky(G)                     # on-device
    except Exception:
        diag_max = float(torch.diagonal(G).abs().max().cpu())
        shift = 1e-6 * diag_max if diag_max > 0.0 else 1e-6
        eye = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        L = torch.linalg.cholesky(G + shift * eye)
    return torch.linalg.solve_triangular(L, Y.T, upper=False).T


def _cholesky_qr2(Y):
    """CholeskyQR2: two Cholesky-QR passes restore fp32-tier orthonormality."""
    return _cholesky_qr(_cholesky_qr(Y))


def _pca_gpu_randomized(
    X_arr,
    center,
    scale,
    n_components,
    col_means_cpu,
    scale_values_cpu,
    var_names,
    device,
    use_fp64,
    force,
    *,
    oversample: int = 10,
    n_iter: int = 4,
    seed: int = 0,
    device_resident: bool = False,
) -> PCASolution:
    """Randomized truncated SVD PCA (HMT) with CholeskyQR2 orthonormalization.

    Heavy ops (random projection, power-iteration matmuls, CholeskyQR2) stay
    on-device; only the ``l × l`` eigendecomposition runs on the host. Returns a
    :class:`PCASolution` with R's sign convention (largest-abs element per
    column positive), matching the CPU/SVD path.
    """
    import torch

    n, p = X_arr.shape
    X_gpu, dtype = _prepare_X_gpu(
        X_arr, center, scale, col_means_cpu, scale_values_cpu, device, use_fp64,
    )

    rank = min(n, p)
    sketch_width = min(n_components + oversample, rank)

    # NON-DETERMINISTIC: randomized SVD draws a random Gaussian sketch Ω. The
    # seed is injectable and defaults to a fixed value, so a given input yields
    # the same result on every call (Rule 6); pass a different ``seed`` to draw
    # an independent sketch. A *local* Generator is used so there is no global
    # RNG state (Rule 5). GPU RNG is not bit-identical to the CPU RNG — the
    # result is statistically equivalent, not bitwise-equal, to a CPU run.
    gen = torch.Generator(device=X_gpu.device)
    gen.manual_seed(int(seed))
    omega = torch.randn(
        p, sketch_width, device=X_gpu.device, dtype=dtype, generator=gen,
    )

    # Range finder + power iterations. Every orthonormalization is CholeskyQR2
    # (matmul + l×l Cholesky), never torch.linalg.qr (which OOMs on MPS).
    q_basis = _cholesky_qr2(X_gpu @ omega)               # n × l
    for _ in range(n_iter):
        q_basis = _cholesky_qr2(X_gpu.T @ q_basis)       # p × l
        q_basis = _cholesky_qr2(X_gpu @ q_basis)         # n × l

    b_small = q_basis.T @ X_gpu                          # l × p

    # SVD of the small B (l×p) via the eig of B Bᵀ (l×l). eigh has no Metal
    # kernel (it raises on MPS), so this tiny decomposition runs on the host —
    # the one deliberate CPU step. Eigenvalues are the squared singular values
    # of B, which equal those of the centered/scaled X within the captured
    # subspace.
    gram_small = (b_small @ b_small.T).cpu()             # l × l on host
    eigvals, eigvecs = torch.linalg.eigh(gram_small)     # ascending
    eigvals = torch.flip(eigvals, dims=[0])              # → descending
    eigvecs = torch.flip(eigvecs, dims=[1])
    eigvals = torch.clamp(eigvals, min=0.0)

    # Condition gate (fp32 squared-condition risk inside CholeskyQR), analogous
    # to the Gram path's _MIN_EIG_RATIO gate in gpu_pca.py. Over the captured
    # l-subspace cond(X)² ≈ σ²_max / σ²_min; refuse when the ratio is past the
    # precision threshold (cond(X) ≳ 1e3 for fp32) unless force=True.
    if not force:
        max_eig = float(eigvals[0])
        min_eig = float(eigvals[-1])
        ratio_threshold = _MIN_EIG_RATIO_FP64 if use_fp64 else _MIN_EIG_RATIO_FP32
        if max_eig <= 0.0:
            raise NumericalError(
                "GPU PCA (randomized path): all captured singular values are "
                "zero — the data is degenerate."
            )
        if min_eig <= ratio_threshold * max_eig:
            cond_x_est = float(
                np.sqrt(max_eig / max(min_eig, max_eig * 1e-300))
            )
            precision_name = "FP64" if use_fp64 else "FP32"
            raise NumericalError(
                f"GPU PCA (randomized path): estimated cond(X) ≈ "
                f"{cond_x_est:.2e} exceeds the safe threshold for "
                f"{precision_name} (ratio σ²_min/σ²_max = "
                f"{min_eig / max_eig:.2e} vs. required > "
                f"{ratio_threshold:.0e}). CholeskyQR squares the condition "
                f"number; at this precision the orthonormalization is "
                f"unreliable. Options: backend='cpu' (always safe), "
                f"backend='gpu_fp64' on CUDA, or force=True (bypass the check "
                f"— numerical results will be unreliable)."
            )

    sing_vals = torch.sqrt(eigvals)                      # singular values of X_c
    eigvecs_gpu = eigvecs.to(device=X_gpu.device, dtype=dtype)
    sing_vals_gpu = sing_vals.to(device=X_gpu.device, dtype=dtype)
    # Right singular vectors of the centered/scaled X: V = Bᵀ W / σ.
    rotation_gpu = (b_small.T @ eigvecs_gpu) / sing_vals_gpu.clamp_min(
        1e-30
    ).unsqueeze(0)                                       # p × l
    rotation_gpu = _fix_sign_convention_gpu(rotation_gpu)

    sdev_gpu = sing_vals_gpu / np.sqrt(n - 1)
    scores_gpu = X_gpu @ rotation_gpu

    return _finalize(
        sdev_gpu, rotation_gpu, scores_gpu, n_components,
        col_means_cpu, scale_values_cpu, var_names, n, p,
        device_resident=device_resident, method="randomized",
    )
