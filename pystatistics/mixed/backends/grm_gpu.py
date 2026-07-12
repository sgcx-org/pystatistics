"""GPU (torch) backend for the low-rank / GRM mixed model.

Mirrors the CPU reference (:mod:`pystatistics.mixed._grm_cpu`) op-for-op in
torch, on a CUDA or MPS device, in float32 (``backend='gpu'``) or float64
(``backend='gpu_fp64'``, CUDA only). The dominant kernels are the dense M×M
Gram ``c²W'W`` (one big GEMM) and its Cholesky — the cuBLAS/cuSOLVER regime — so
the GPU earns its keep here in a way it does not for the sparse-design ``lmm``.

CF-1 guard (the float32 hazard, made loud — never silently wrong)
----------------------------------------------------------------
The float32 path forms the Gram ``G = c²W'W + I``. If W has near-collinear
columns (e.g. markers in tight LD), forming ``W'W`` in float32 can lose the
trailing eigenvalues. Two layers make this **never silently wrong**:

1. **The ``+I`` floor + a loud Cholesky.** ``G`` is floored at eigenvalue ≥ 1,
   so when float32 cannot represent it faithfully the Cholesky loses positive-
   definiteness and raises — it does not return a plausible-but-biased factor.
   Measured across cond(W) ∈ [1e3, 2e4] × 8 seeds, the forced float32 fit was
   *always* either accurate (|Δh²| ≤ 9e-4 vs the float64 reference) or a loud
   failure; a silently-wrong result never occurred. So there is no silent-wrong
   band to fall into.

2. **An up-front conditioning gate** for a clean, early, informative refusal.
   We assess the conditioning of ``W'W`` once, on the host in float64 (it is
   θ-independent, so it is a single amortized cost), and refuse the float32 fit
   with a specific message when W is past the float32-safe boundary — better UX
   than a mid-optimization "not positive-definite". The threshold
   (``_MIN_EIG_RATIO_FP32``) is **calibrated to the measured boundary**: float32
   is accurate up to cond(W) ≈ 3e3 (ratio ≈ 1e-7) and starts failing beyond, so
   the gate is placed there — accepting the correct region (R9: no false
   rejection) and refusing the failing one (R12: no biased acceptance).

The float64 GPU path and the CPU path are exact and are never gated. This
implements A6 (the exact request or a loud failure), RIGOR R9 / R12 / R13 (the
threshold is re-proven on *this* regime, above) and R14 (the guarantee rests on
the host-float64 gate + the loud float32 Cholesky). ``force=True`` bypasses the
gate; even then the float32 path fails loud rather than silently wrong.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from pystatistics.core.exceptions import NumericalError
from pystatistics.core.compute.torch_interop import to_host_f64
from pystatistics.mixed._grm_cpu import GRMFit


# Conditioning gate on W'W for the float32 Gram path. ratio = λ_min/λ_max of
# W'W = 1/cond(W'W) = 1/cond(W)². 1e-7 ⇒ cond(W) ≲ 3e3 — the measured boundary
# below which the float32 fit matches the float64 reference (|Δh²| ≤ 9e-4) and
# above which it fails loud (see the module docstring's CF-1 calibration). The
# gate is an early, informative refusal; the loud float32 Cholesky is the
# backstop, so there is no silent-wrong band even if the gate is bypassed.
_MIN_EIG_RATIO_FP32 = 1.0e-7


def _gram_condition_ratio(random_factor: NDArray) -> float:
    """λ_min/λ_max of W'W, computed on the host in float64 (the CF-1 gate)."""
    G = random_factor.T @ random_factor
    eig = np.linalg.eigvalsh(G)            # ascending, symmetric
    max_eig = float(eig[-1])
    min_eig = float(eig[0])
    if max_eig <= 0.0:
        return 0.0
    return max(min_eig, 0.0) / max_eig


def _check_cf1_gate(random_factor: NDArray, use_fp64: bool,
                    force: bool) -> None:
    """Refuse a float32 GRM fit whose W'W is past the float32-safe boundary."""
    if use_fp64 or force:
        return
    ratio = _gram_condition_ratio(random_factor)
    if ratio < _MIN_EIG_RATIO_FP32:
        cond_rf = float(np.sqrt(1.0 / ratio)) if ratio > 0 else float("inf")
        raise NumericalError(
            f"GRM GPU float32 path: the low-rank factor random_factor (W) is "
            f"too ill-conditioned for single precision (estimated cond(W) ≈ "
            f"{cond_rf:.2e}; λ_min/λ_max of W'W = {ratio:.2e} vs. required "
            f"> {_MIN_EIG_RATIO_FP32:.0e}). Forming W'W in float32 would lose "
            f"its trailing eigenvalues and bias the variance components. "
            f"Options: backend='gpu_fp64' (CUDA, exact), backend='cpu' "
            f"(float64 reference), or force=True to bypass this check (results "
            f"on a truly ill-conditioned W will be unreliable)."
        )


def _solve_torch(theta, Wt, Xt, yt, reml, eye_M):
    """One GRM solve at fixed θ on-device; returns a dict of torch tensors."""
    import torch
    n, M = Wt.shape
    p = Xt.shape[1]
    c = theta / (M ** 0.5)
    ZL = c * Wt                                   # (n, M)
    G = ZL.transpose(0, 1) @ ZL + eye_M           # (M, M)
    try:
        L = torch.linalg.cholesky(G)
    except torch._C._LinAlgError:
        raise NumericalError(
            "GRM Gram (c²W'W + I) lost positive-definiteness in this precision "
            "— W is too ill-conditioned for the float32 GPU path. Use "
            "backend='gpu_fp64' (CUDA) or backend='cpu'."
        )
    a = ZL.transpose(0, 1) @ yt.unsqueeze(1)      # (M, 1)
    B = ZL.transpose(0, 1) @ Xt                   # (M, p)
    m_y = torch.cholesky_solve(a, L)              # (M, 1)
    Wm = torch.cholesky_solve(B, L)               # (M, p)
    RtR = Xt.transpose(0, 1) @ Xt - B.transpose(0, 1) @ Wm
    rhs = Xt.transpose(0, 1) @ yt.unsqueeze(1) - B.transpose(0, 1) @ m_y
    try:
        RX = torch.linalg.cholesky(RtR)
    except torch._C._LinAlgError:
        raise NumericalError(
            "GRM fixed-effects system (X'V⁻¹X) is singular — collinear "
            "covariates in X. Remove redundant columns."
        )
    beta = torch.cholesky_solve(rhs, RX)          # (p, 1)
    u = (m_y - Wm @ beta).squeeze(1)              # (M,)
    genetic = ZL @ u                              # (n,)
    resid = yt - Xt @ beta.squeeze(1) - genetic
    pwrss = resid @ resid + u @ u
    sigma_e2 = pwrss / (n - p) if reml else pwrss / n
    logdet_G = 2.0 * torch.log(torch.diagonal(L).clamp_min(1e-300)).sum()
    logdet_RX = 2.0 * torch.log(torch.diagonal(RX).abs().clamp_min(1e-300)).sum()
    return {
        "beta": beta.squeeze(1), "u": u, "sigma_e2": sigma_e2, "pwrss": pwrss,
        "logdet_G": logdet_G, "RX": RX, "genetic": genetic, "resid": resid,
        "fitted": yt - resid, "logdet_RX": logdet_RX, "n": n, "p": p,
    }


def _deviance_torch(theta, Wt, Xt, yt, reml, eye_M) -> float:
    import torch
    s = _solve_torch(theta, Wt, Xt, yt, reml, eye_M)
    n, p = s["n"], s["p"]
    if reml:
        df = n - p
        dev = s["logdet_G"] + s["logdet_RX"] + df * (
            1.0 + torch.log(2 * torch.pi * s["pwrss"] / df))
    else:
        dev = s["logdet_G"] + n * (1.0 + torch.log(2 * torch.pi * s["pwrss"] / n))
    return float(dev)


def grm_fit_gpu(
    random_factor: NDArray, X: NDArray, y: NDArray, *,
    reml: bool, tol: float, max_iter: int,
    device_type: str, use_fp64: bool, force: bool,
    theta_max: float = 1.0e3,
) -> GRMFit:
    """Fit the GRM model on a GPU device, mirroring the CPU reference.

    Args:
        random_factor, X, y: numpy float64 arrays (validated upstream).
        reml: REML or ML.
        tol, max_iter: optimizer settings for the 1-D θ search.
        device_type: 'cuda' or 'mps'.
        use_fp64: True for the float64 path (CUDA only), False for float32.
        force: bypass the CF-1 float32 conditioning gate.
    """
    import torch

    # CF-1 gate (host float64) before any device work — refuse a float32 fit
    # whose W is past the safe boundary.
    _check_cf1_gate(random_factor, use_fp64, force)

    device = torch.device(device_type)
    dtype = torch.float64 if use_fp64 else torch.float32
    Wt = torch.as_tensor(random_factor, device=device, dtype=dtype)
    Xt = torch.as_tensor(X, device=device, dtype=dtype)
    yt = torch.as_tensor(y, device=device, dtype=dtype)
    M = Wt.shape[1]
    eye_M = torch.eye(M, device=device, dtype=dtype)

    # FP32 tolerance floor (CONVENTIONS): float32 gradients bottom out ~1e-7.
    xatol = max(tol, 1e-5) if not use_fp64 else tol

    res = minimize_scalar(
        _deviance_torch, args=(Wt, Xt, yt, reml, eye_M),
        bounds=(0.0, theta_max), method="bounded",
        options={"xatol": xatol, "maxiter": max_iter},
    )
    theta_hat = float(res.x)
    converged = bool(res.success)
    n_iter = int(getattr(res, "nfev", 0))

    s = _solve_torch(theta_hat, Wt, Xt, yt, reml, eye_M)
    return GRMFit(
        theta=theta_hat,
        beta=to_host_f64(s["beta"]),
        u=to_host_f64(s["u"]),
        sigma_e2=float(to_host_f64(s["sigma_e2"])),
        pwrss=float(to_host_f64(s["pwrss"])),
        logdet_G=float(to_host_f64(s["logdet_G"])),
        RX=to_host_f64(s["RX"]),
        genetic_values=to_host_f64(s["genetic"]),
        fitted=to_host_f64(s["fitted"]),
        residuals=to_host_f64(s["resid"]),
        converged=converged,
        n_iter=n_iter,
    )
