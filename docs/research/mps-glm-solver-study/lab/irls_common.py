"""Shared IRLS helpers for the MPS-fp32 solver study (binomial logit only).

One job: the device-agnostic numerics every driver needs — sigmoid, binomial
deviance, the float32 device setup, and the *host float64* Newton-decrement gate
that decides whether a stopped fp32 fit reached the optimum (accept) or stalled
off it (fail loud). This gate is the A6 contract made measurable; it is the same
relative-Newton-decrement idea used in pystatistics' gpu_glm, recomputed here in
float64 on the host from the returned coefficients.
"""

from __future__ import annotations

import os
import numpy as np
import torch

# Device is selectable so the IDENTICAL solver code runs on MPS (Mac) and CUDA
# (Forge) -- the whole point of the cross-check. Default MPS.
_DEV = os.environ.get("PYSTATS_LAB_DEVICE", "mps")
DEVICE = torch.device(_DEV)
F32 = torch.float32


def sync():
    """Device barrier for honest wall timing."""
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    elif DEVICE.type == "cuda":
        torch.cuda.synchronize()

# Accept a stopped fp32 fit iff its relative Newton decrement is below this.
# Same constant pystatistics uses on the GPU GLM fp32 path.
FP32_REL_DECREMENT_TOL = 1e-6


def to_dev(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x_np, dtype=np.float32)).to(DEVICE)


def sigmoid(eta: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(eta)


def binom_deviance_dev(y: torch.Tensor, eta: torch.Tensor) -> float:
    """Binomial(logit) deviance on device, returned as a host float."""
    mu = torch.sigmoid(eta).clamp(1e-7, 1 - 1e-7)
    dev = -2.0 * (y * torch.log(mu) + (1 - y) * torch.log1p(-mu)).sum()
    return float(dev.item())


def working_wz(y: torch.Tensor, eta: torch.Tensor):
    """IRLS weights w = mu(1-mu) (floored) and working response z = eta+(y-mu)/w."""
    mu = torch.sigmoid(eta)
    w = torch.clamp(mu * (1 - mu), min=1e-20)
    z = eta + (y - mu) / w
    return w, z


def host_rel_newton_decrement(X_np: np.ndarray, y_np: np.ndarray,
                              coef: np.ndarray) -> float:
    """Relative Newton decrement λ²/(|deviance|+0.1) in float64 on the host.

    λ² = Uᵀ(XᵀWX)⁻¹U with U the score and W the binomial IRLS weights, all in
    float64. ≈ the fraction of deviance still on the table: ~0 at a true optimum,
    O(1e-1)+ when the coefficients sit off it. Returns inf if XᵀWX is singular
    (also a refuse-worthy state).
    """
    X = X_np.astype(np.float64, copy=False)
    y = y_np.astype(np.float64, copy=False)
    coef = coef.astype(np.float64, copy=False)
    eta = X @ coef
    mu = 1.0 / (1.0 + np.exp(-eta))
    muc = np.clip(mu, 1e-12, 1 - 1e-12)
    dev = -2.0 * (y * np.log(muc) + (1 - y) * np.log1p(-muc)).sum()
    score = X.T @ (y - mu)
    w = np.maximum(mu * (1 - mu), 1e-30)
    XtWX = X.T @ (w[:, None] * X)
    try:
        step = np.linalg.solve(XtWX, score)
    except np.linalg.LinAlgError:
        return float("inf")
    lam2 = float(score @ step)
    return lam2 / (abs(dev) + 0.1)


def coef_max_rel_err(coef: np.ndarray, ref: np.ndarray, cov: slice) -> float:
    """Max relative error on the identifiable covariate coefficients."""
    c = coef[cov].astype(np.float64)
    r = ref[cov].astype(np.float64)
    denom = np.maximum(np.abs(r), 1e-8)
    return float(np.max(np.abs(c - r) / denom))


def rel_step(b_new: torch.Tensor, b_old: torch.Tensor) -> float:
    num = torch.max(torch.abs(b_new - b_old))
    den = torch.max(torch.abs(b_new)).clamp(min=1e-8)
    return float((num / den).item())
