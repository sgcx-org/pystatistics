"""
Batched Bayesian logistic-regression imputation on the GPU (MICE ``logreg``).

GPU counterpart of ``methods/logreg.py``, vectorized over the ``m`` imputation
chains as the leading batch dimension. At a sweep step every chain fits the
*same* binary target on the *same* observed rows but with different predictor
values (each chain imputed the other columns differently), so we run ``m``
independent logistic fits at once with batched kernels.

Model (identical to the CPU path, R mice's default for two-level factors):

  * ridge-stabilised IRLS (Newton on the concave logistic log-likelihood). Each
    step forms the reweighted Gram ``G = X'WX + ridge·I`` (batched matmul) and
    solves ``G·delta = X'(y - p)``. The solve goes through ``cholesky_ex`` plus
    the matmul-series triangular inverse on MPS (``solve_triangular`` is ~250x
    slower there) and ``solve_triangular`` on CUDA/CPU — the shared device split
    in ``core.compute.linalg``. Iteration is bounded (cap 50) with per-chain
    convergence freezing: a chain stops updating once ``max|delta| < tol``.
  * posterior draw ``beta* ~ N(beta_hat, (X'WX)^{-1})`` — the normal approximation
    around the MLE that R's ``logreg`` uses. With ``G = L L'`` the draw is
    ``beta* = beta_hat + L^{-T} z`` (``Var = L^{-T} L^{-1} = G^{-1}``). Logistic
    has no dispersion parameter, so unlike the Gaussian draw there is no sigma /
    chi-square term.
  * predict ``p = sigmoid(X_mis beta*)`` and sample 0/1 by ``u < p``.

The target arrives as 0/1 class indices (the sweep maps the column's two category
codes to indices) and this returns 0/1 indices, batched (m, n_mis).

Like the CPU method there is no marginal-draw fallback: the ridge plus the linear-
predictor clip keep the fit finite even under quasi-/complete separation (where
beta diverges but the predicted probabilities — what we sample from — saturate
cleanly), and genuine degeneracy is caught by the backend's end-of-sweep
non-finite guard. All randomness flows through the passed generator (Rule 6).
Results match the CPU reference distributionally at the GPU/FP32 tolerance tier.
"""

from __future__ import annotations

from pystatistics.mice.backends._gpu_linreg import (
    add_intercept,
    discrete_glm_compute_dtype,
    _cholesky_ridged,
)
from pystatistics.mice.backends._gpu_spd import apply_inv_factor_T, solve_spd

# Mirrors the CPU logreg constants (methods/logreg.py): relative ridge on X'WX
# (also tames separation), Newton cap and tolerance, and the linear-predictor
# clip that keeps exp() from overflowing and weights strictly positive.
_RIDGE = 1e-5
_MAX_IRLS_ITER = 50
_IRLS_TOL = 1e-8
_ETA_CLIP = 30.0
_W_FLOOR = 1e-9


def _ridged_gram(Xa, w, ridge_diag):
    """Reweighted ridged Gram ``X'WX + ridge·I`` per chain. ``w``: (m, n)."""
    import torch

    XtWX = Xa.transpose(1, 2) @ (w.unsqueeze(-1) * Xa)        # (m, k, k)
    eye = torch.eye(Xa.shape[2], dtype=Xa.dtype, device=Xa.device)
    return XtWX + ridge_diag[:, None, None] * eye


def batched_logistic_irls(y, Xa):
    """Batched ridge-stabilised IRLS. ``y`` (m, n_obs) 0/1, ``Xa`` (m, n_obs, k)
    WITH intercept. Returns ``(beta_hat (m, k), L (m, k, k))`` — the coefficient
    estimate and the lower Cholesky of the ridged Gram at ``beta_hat`` (for the
    posterior covariance ``G^{-1}``)."""
    import torch

    m, n, k = Xa.shape
    dtype, device = Xa.dtype, Xa.device
    beta = torch.zeros((m, k), dtype=dtype, device=device)

    # Relative ridge per chain: _RIDGE * mean_k(sum_n Xa^2)/n (matches the CPU
    # scalar diag_scale, computed independently per chain).
    diag_scale = (Xa * Xa).sum(dim=1).mean(dim=1) / max(n, 1)        # (m,)
    ridge_diag = _RIDGE * diag_scale.clamp_min(1e-12)               # (m,)

    converged = torch.zeros(m, dtype=torch.bool, device=device)
    for _ in range(_MAX_IRLS_ITER):
        eta = (Xa @ beta.unsqueeze(-1)).squeeze(-1).clamp(-_ETA_CLIP, _ETA_CLIP)
        p = torch.sigmoid(eta)
        w = (p * (1.0 - p)).clamp_min(_W_FLOOR)
        grad = Xa.transpose(1, 2) @ (y - p).unsqueeze(-1)          # (m, k, 1)
        L = _cholesky_ridged(_ridged_gram(Xa, w, ridge_diag))
        delta = solve_spd(L, grad).squeeze(-1)                     # (m, k)

        small = delta.abs().amax(dim=1) < _IRLS_TOL
        apply = ~converged
        # Freeze converged chains so they stop drifting; matches the CPU break.
        beta = beta + apply.unsqueeze(-1) * delta
        converged = converged | (apply & small)
        if bool(converged.all()):
            break

    # Posterior covariance at beta_hat: one final Gram/Cholesky at the estimate.
    eta = (Xa @ beta.unsqueeze(-1)).squeeze(-1).clamp(-_ETA_CLIP, _ETA_CLIP)
    p = torch.sigmoid(eta)
    w = (p * (1.0 - p)).clamp_min(_W_FLOOR)
    L = _cholesky_ridged(_ridged_gram(Xa, w, ridge_diag))
    return beta, L


def gpu_logreg_impute(y_obs, X_obs, X_mis, gen, *, donors=None, n_classes=None):
    """Batched Bayesian logistic-regression imputation (R ``logreg``).

    Parameters mirror the other GPU methods but batched: ``y_obs`` (m, n_obs) of
    0/1 class indices, ``X_obs`` (m, n_obs, q), ``X_mis`` (m, n_mis, q). Returns
    (m, n_mis) of 0/1 indices. ``donors`` and ``n_classes`` are accepted and
    ignored (binary is always 2-class) so the backend calls every categorical
    method with one uniform signature.

    The fit and draw run in FP64 where the device supports it
    (``discrete_glm_compute_dtype``): under (quasi-)separation the FP32 IRLS Gram
    goes near-singular, the Cholesky/solve returns a non-finite estimate, and the
    Bernoulli draw silently collapses every cell to 0 (``u < NaN`` is False) —
    which then corrupts every column using this one as a predictor. A non-finite
    predicted probability now yields NaN (never a silent 0), so a genuinely
    degenerate fit surfaces via the backend's end-of-sweep guard (Rule 1).
    """
    import torch

    out_dtype = X_obs.dtype
    compute_dtype = discrete_glm_compute_dtype(X_obs.device, out_dtype)
    X_obs = X_obs.to(compute_dtype)
    X_mis = X_mis.to(compute_dtype)
    y_obs = y_obs.to(compute_dtype)

    Xa = add_intercept(X_obs)
    beta_hat, L = batched_logistic_irls(y_obs, Xa)
    m, k = beta_hat.shape

    z = torch.randn((m, k), generator=gen, dtype=Xa.dtype, device=Xa.device)
    beta_star = beta_hat + apply_inv_factor_T(L, z.unsqueeze(-1)).squeeze(-1)

    Xa_mis = add_intercept(X_mis)
    eta = (Xa_mis @ beta_star.unsqueeze(-1)).squeeze(-1).clamp(-_ETA_CLIP, _ETA_CLIP)
    p = torch.sigmoid(eta)
    u = torch.rand(p.shape, generator=gen, dtype=p.dtype, device=p.device)
    out = (u < p).to(out_dtype)
    # Fail loud (Rule 1): non-finite p -> NaN, not a silent 0.
    return torch.where(torch.isfinite(p), out, torch.full_like(out, float("nan")))
