"""GPU backend for PCA.

Two algorithmic paths:

    method='svd'   — SVD of X on GPU. Always safe; O(min(n, p) · max(n, p)²)
                     cost dominated by cuSOLVER's gesvdj which has real
                     sequential dependencies (Householder bidiagonalization
                     + QR iteration). Gains a modest ~3× over multi-threaded
                     LAPACK dgesdd on consumer hardware.

    method='gram'  — Eigendecomposition of the Gram matrix X'X. This turns
                     the problem into two GPU sweet spots: one big GEMM to
                     form X'X, and a symmetric eigendecomp (torch.linalg.eigh)
                     on a p×p matrix. For tall-skinny well-conditioned data
                     (n ≫ p) the speedup is large (30–100×+ on an RTX 5070
                     Ti). Cost: the condition number squares, so cond(X'X)
                     = cond(X)². Refuses unless cond(X'X) is safe for the
                     current precision (matching the OLS Cholesky
                     condition-number gate in ``gpu.py``). Pass
                     ``force=True`` to bypass the check.

    method='auto'  — Prefer 'gram' when n > 2p (Gram wins on tall-skinny
                     shapes) and the condition check passes; else 'svd'.

All compute stays on GPU until final transfer of the small result
tensors (sdev, rotation, scores). TF32 is deliberately not enabled —
its 10-bit mantissa gives ~1e-3 per-op precision which composes past
the ``GPU_FP32`` (rtol=1e-4) tier guarantee.

Two-tier validation (README "Design Philosophy"):
    CPU is validated against R to rtol = 1e-10.
    GPU is validated against CPU at GPU_FP32 tolerance (rtol = 1e-4,
    atol = 1e-5). Divergence at that scale is by design.

FP32 is the default on GPU. Consumer NVIDIA cards have deliberately
crippled FP64 throughput (RTX 5070 Ti is ~1/64× FP32 rate on FP64),
so forcing FP64 on GPU is slower than single-threaded numpy LAPACK
on most shapes. FP64 remains available on CUDA for users who need
machine-precision CPU parity (`use_fp64=True`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatistics.multivariate._common import PCAResult

# Condition-number gate for the Gram-matrix path.
#
# cond(X'X) = cond(X)^2. After forming X'X in float precision `eps`,
# eigenvalues smaller than `max_eig * eps` are noise. We refuse the
# Gram path when min_eig / max_eig is below the following ratios.
# These are chosen to preserve ~4 usable digits of precision on the
# smallest eigenvalue, the same engineering margin the OLS Cholesky
# path uses (`GPU_CONDITION_THRESHOLD = 1e6` on cond(X) → 1e12 on
# cond(X'X) → 1e-12 ratio for FP64).
_MIN_EIG_RATIO_FP64 = 1.0e-12   # cond(X) <= 1e6 safe
_MIN_EIG_RATIO_FP32 = 1.0e-6    # cond(X) <= 1e3 safe


def _fix_sign_convention_gpu(rotation_gpu):
    """Enforce R's sign convention on GPU loadings.

    For each column, the element with the largest absolute value is
    made positive. Pure GPU: one argmax + gather + sign + broadcast
    multiply — no host-device round trip.
    """
    import torch
    abs_rot = torch.abs(rotation_gpu)
    # Indices of max-abs entry per column:
    idx = torch.argmax(abs_rot, dim=0)
    # Gather the signed value at that (row, col) pair:
    col_range = torch.arange(rotation_gpu.shape[1], device=rotation_gpu.device)
    signs = torch.sign(rotation_gpu[idx, col_range])
    # A zero singular value (rare) would give sign 0; treat as +1.
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return rotation_gpu * signs.unsqueeze(0)


def _prepare_X_gpu(X_arr, center, scale, col_means_cpu, scale_values_cpu,
                   device, use_fp64):
    """Stream X to GPU, center and/or scale.

    Two entry paths:
      1. ``X_arr`` is a numpy array — pay the host↔device transfer now
         (pageable H2D ≈ 66 ms / GB on PCIe 4.0 x16 on consumer hardware).
      2. ``X_arr`` is already a ``torch.Tensor`` on the requested device —
         the caller materialized a GPU DataSource via ``DataSource.to()``
         and the transfer was paid once, up front. We reach the compute
         ceiling (≈ 5 ms total for the Gram path on 1M × 100) with zero
         per-call H2D cost.

    This second path is the whole point of the device-resident
    DataSource API. Without it, every successive fit on the same data
    re-pays 66 ms / GB of transfer that vastly exceeds the actual
    compute time.
    """
    import torch
    torch_device = torch.device(device)
    dtype = torch.float64 if use_fp64 else torch.float32

    if isinstance(X_arr, torch.Tensor):
        # Already on some device. Move if necessary (user may have
        # constructed a CUDA:0 DataSource and asked us to run on
        # CUDA:1, unusual but valid) and cast to the fit dtype.
        X_gpu = X_arr.to(device=torch_device, dtype=dtype)
    else:
        # numpy → GPU. The per-call transfer. See the docstring above
        # for why this cost dominates single-call wall time at scale.
        X_gpu = torch.as_tensor(X_arr, device=torch_device, dtype=dtype)

    # col_means / scale_values may arrive as either numpy arrays (CPU
    # path) or GPU tensors (the top-level ``pca()`` kept them device-
    # resident to avoid extra D2H syncs). Handle both.
    if center:
        if isinstance(col_means_cpu, torch.Tensor):
            means_gpu = col_means_cpu.to(device=torch_device, dtype=dtype)
        else:
            means_gpu = torch.as_tensor(
                col_means_cpu, device=torch_device, dtype=dtype,
            )
        X_gpu = X_gpu - means_gpu
    if scale and scale_values_cpu is not None:
        if isinstance(scale_values_cpu, torch.Tensor):
            sd_gpu = scale_values_cpu.to(device=torch_device, dtype=dtype)
        else:
            sd_gpu = torch.as_tensor(
                scale_values_cpu, device=torch_device, dtype=dtype,
            )
        X_gpu = X_gpu / sd_gpu
    return X_gpu, dtype


def _finalize(sdev_gpu, rotation_gpu, scores_gpu, n_components,
              col_means_cpu, scale_values_cpu, var_names, n, p):
    """Truncate, transfer, build PCAResult. Shared.

    Transfers scores / rotation / sdev / center / scale back to host
    at their *native* compute dtype — does NOT force-promote FP32 to
    FP64. Reason: promoting a 400 MB FP32 scores tensor to FP64 on the
    GPU doubles the D2H payload to 800 MB, adding ~140 ms of PCIe
    transfer to report "precision" we never actually had (the fit ran
    in FP32). At the project's GPU_FP32 tolerance tier this is
    correct behavior: the GPU path's precision ceiling is FP32, and
    zero-padding with FP64 bits doesn't change that. Users who want
    FP64 scores should run with ``use_fp64=True``.
    """
    import torch
    sdev_gpu = sdev_gpu[:n_components]
    rotation_gpu = rotation_gpu[:, :n_components]
    scores_gpu = scores_gpu[:, :n_components]
    # .cpu() returns a tensor of the same dtype; .numpy() views it as
    # numpy with the matching dtype (float32 stays float32).
    sdev = sdev_gpu.cpu().numpy()
    rotation = rotation_gpu.cpu().numpy()
    scores = scores_gpu.cpu().numpy()

    if isinstance(col_means_cpu, torch.Tensor):
        col_means_cpu = col_means_cpu.detach().cpu().numpy()
    if isinstance(scale_values_cpu, torch.Tensor):
        scale_values_cpu = scale_values_cpu.detach().cpu().numpy()

    return PCAResult(
        sdev=sdev,
        rotation=rotation,
        center=col_means_cpu,
        scale=scale_values_cpu,
        x=scores,
        n_obs=n,
        n_vars=p,
        var_names=var_names,
    )


def _pca_gpu_svd(
    X_arr, center, scale, n_components,
    col_means_cpu, scale_values_cpu, var_names,
    device, use_fp64,
) -> PCAResult:
    """SVD-based PCA on GPU. Always safe, moderate GPU speedup."""
    import torch
    n, p = X_arr.shape
    X_gpu, _ = _prepare_X_gpu(
        X_arr, center, scale, col_means_cpu, scale_values_cpu,
        device, use_fp64,
    )
    # Thin SVD on GPU.
    _U_gpu, S_gpu, Vt_gpu = torch.linalg.svd(X_gpu, full_matrices=False)
    rotation_gpu = Vt_gpu.T                                     # (p, k)
    rotation_gpu = _fix_sign_convention_gpu(rotation_gpu)
    # scores = X_centered @ V_signfixed (use sign-fixed rotation so
    # scores match reported rotation, matching the CPU path).
    scores_gpu = X_gpu @ rotation_gpu
    sdev_gpu = S_gpu / np.sqrt(n - 1)
    return _finalize(
        sdev_gpu, rotation_gpu, scores_gpu, n_components,
        col_means_cpu, scale_values_cpu, var_names, n, p,
    )


def _pca_gpu_gram(
    X_arr, center, scale, n_components,
    col_means_cpu, scale_values_cpu, var_names,
    device, use_fp64, force,
) -> PCAResult:
    """Gram-matrix PCA on GPU: eigendecompose X'X.

    Algorithmic choice: one big GEMM to form ``X'X`` (cuBLAS's sweet
    spot), one symmetric eigendecomposition on a p×p matrix
    (``torch.linalg.eigh``, much faster and more GPU-friendly than
    the iterative SVD on the full n×p X), one final GEMM for scores.
    For tall-skinny well-conditioned data this is typically 30–100×+
    faster than the SVD path on the same GPU.

    Condition-number gate: cond(X'X) = cond(X)². We compute the ratio
    ``λ_min / λ_max`` from the eigendecomposition and refuse unless
    it is above a precision-dependent threshold. For FP32 the
    threshold is 1e-6 (cond(X) ≤ 1000); for FP64 it is 1e-12
    (cond(X) ≤ 10⁶, matching the OLS Cholesky gate). ``force=True``
    bypasses the check — use at your own risk.
    """
    import torch
    from pystatistics.core.exceptions import NumericalError

    n, p = X_arr.shape
    X_gpu, dtype = _prepare_X_gpu(
        X_arr, center, scale, col_means_cpu, scale_values_cpu,
        device, use_fp64,
    )

    # X'X — the one big GEMM.
    G = X_gpu.T @ X_gpu                                         # (p, p)
    # Symmetric eigendecomp. eigh returns ascending eigenvalues.
    eigvals, eigvecs = torch.linalg.eigh(G)

    # Condition-number gate. Pull the two relevant scalars in ONE
    # D2H transfer so we don't pay two sync points (each sync can
    # cost ~40 ms on PCIe 4.0 even for a single scalar because it
    # blocks until prior kernels drain).
    edge_pair = torch.stack(
        [eigvals[0], eigvals[-1]]
    ).to(torch.float64).cpu().numpy()
    min_eig, max_eig = float(edge_pair[0]), float(edge_pair[1])
    if not force:
        ratio_threshold = _MIN_EIG_RATIO_FP64 if use_fp64 else _MIN_EIG_RATIO_FP32
        # max_eig should be positive for non-degenerate data.
        if max_eig <= 0.0:
            raise NumericalError(
                "GPU PCA (Gram path): all eigenvalues of X'X are "
                "non-positive — data is degenerate. Use method='svd'."
            )
        # min_eig can be slightly negative due to FP round-off; treat
        # any magnitude <= ratio_threshold * max_eig as ill-conditioned.
        if min_eig <= ratio_threshold * max_eig:
            # Estimate cond(X) = sqrt(cond(X'X)) for the error message.
            cond_XtX = max_eig / max(abs(min_eig), max_eig * 1e-300)
            cond_X_est = float(np.sqrt(cond_XtX))
            precision_name = "FP64" if use_fp64 else "FP32"
            raise NumericalError(
                f"GPU PCA (Gram path): estimated cond(X) ≈ {cond_X_est:.2e} "
                f"exceeds the safe threshold for {precision_name} "
                f"(ratio λ_min/λ_max = {min_eig/max_eig:.2e} vs. "
                f"required > {ratio_threshold:.0e}). The Gram path "
                f"squares the condition number and cannot recover small "
                f"eigenvalues at this precision. Options: use "
                f"method='svd' (always safe), use_fp64=True (raises "
                f"the threshold), or force=True (bypass the check — "
                f"numerical results will be unreliable)."
            )

    # Reverse to descending order (SVD convention).
    eigvals = torch.flip(eigvals, dims=[0])
    eigvecs = torch.flip(eigvecs, dims=[1])
    # Clamp tiny negatives (numerical noise) to zero before sqrt.
    eigvals_clamped = torch.clamp(eigvals, min=0.0)
    sdev_gpu = torch.sqrt(eigvals_clamped / (n - 1))

    rotation_gpu = _fix_sign_convention_gpu(eigvecs)
    scores_gpu = X_gpu @ rotation_gpu

    return _finalize(
        sdev_gpu, rotation_gpu, scores_gpu, n_components,
        col_means_cpu, scale_values_cpu, var_names, n, p,
    )


def pca_gpu(
    X_arr: NDArray,
    *,
    center: bool,
    scale: bool,
    n_components: int,
    col_means_cpu: NDArray,
    scale_values_cpu: NDArray | None,
    var_names: tuple[str, ...] | None,
    device: str = "cuda",
    use_fp64: bool = False,
    method: str = "svd",
    force: bool = False,
) -> PCAResult:
    """Fit PCA on GPU. Dispatches on ``method``.

    Parameters
    ----------
    X_arr : NDArray
        Data matrix (n, p), already validated and finite-checked.
    center, scale : bool
    n_components : int
        Already validated to lie in [1, min(n, p)].
    col_means_cpu, scale_values_cpu :
        Pre-computed by the CPU gateway.
    var_names : tuple or None
    device : {'cuda', 'mps'}
    use_fp64 : bool
        FP64 on CUDA yes, MPS raises (MPS has no FP64 SVD).
    method : {'svd', 'gram', 'auto'}
        'svd' always-safe SVD of X (default).
        'gram' eigendecompose X'X; raises on ill-conditioned data
            unless ``force=True``.
        'auto' uses 'gram' when n > 2p AND condition check passes;
            falls back to 'svd' otherwise.
    force : bool
        Bypass the Gram-path condition check. Numerical results will
        be unreliable for truly ill-conditioned inputs.
    """
    import torch

    if device == "mps" and use_fp64:
        raise RuntimeError(
            "GPU PCA: MPS does not support FP64. Use use_fp64=False or "
            "backend='cpu'."
        )

    # NOTE on TF32: we deliberately do NOT enable
    # ``torch.backends.cuda.matmul.allow_tf32``. TF32's 10-bit mantissa
    # gives ~1e-3 precision per operation; composed across the
    # ``scores = X_centered @ V`` matmul that dominates this kernel,
    # the per-element error exceeds the ``GPU_FP32`` tier (rtol=1e-4,
    # atol=1e-5) the project guarantees for GPU-vs-CPU agreement.

    if method not in ("svd", "gram", "auto"):
        from pystatistics.core.exceptions import ValidationError
        raise ValidationError(
            f"method: must be 'svd', 'gram', or 'auto', got {method!r}"
        )

    n, p = X_arr.shape

    if method == "svd":
        return _pca_gpu_svd(
            X_arr, center, scale, n_components,
            col_means_cpu, scale_values_cpu, var_names,
            device, use_fp64,
        )
    if method == "gram":
        return _pca_gpu_gram(
            X_arr, center, scale, n_components,
            col_means_cpu, scale_values_cpu, var_names,
            device, use_fp64, force,
        )

    # method == 'auto'. Use Gram when it is likely to help: n > 2p
    # (tall-skinny — the regime where the squared condition number
    # tradeoff pays off) and the condition check passes. If Gram
    # raises NumericalError, fall back to SVD — this is the auto
    # path's explicit fallback contract.
    if n > 2 * p:
        try:
            return _pca_gpu_gram(
                X_arr, center, scale, n_components,
                col_means_cpu, scale_values_cpu, var_names,
                device, use_fp64, force=False,
            )
        except Exception:
            # Any Gram-path failure (condition, degenerate, numerical)
            # → fall back to the always-safe SVD path.
            pass
    return _pca_gpu_svd(
        X_arr, center, scale, n_components,
        col_means_cpu, scale_values_cpu, var_names,
        device, use_fp64,
    )


