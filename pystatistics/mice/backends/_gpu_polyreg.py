"""
Batched multinomial-logit imputation on the GPU (MICE ``polyreg``).

GPU counterpart of ``methods/polyreg.py``, vectorized over the ``m`` imputation
chains. R mice's default for unordered factors with >2 levels: fit a multinomial
logit of the K-level target on the predictors, draw the coefficient block once
from its posterior normal approximation, predict class probabilities for the
missing rows, and sample a category per row.

Model (identical to the CPU path / R ``nnet::multinom``): softmax with the LAST
class as the reference (``beta_ref = 0``), coefficients ``(K-1, P)``. The CPU path
fits with L-BFGS-B; here we use **batched Newton** on the convex multinomial NLL —
same unique MLE — because Newton is the GPU-friendly, bounded-iteration shape the
logreg path already established. Each step forms the multinomial block Hessian

    H[j, k] = X' diag(W_jk) X,   W_jj = p_j(1-p_j),   W_jk = -p_j p_k   (j != k)

over the ``K-1`` non-reference classes (class-major block layout matching
``coef.ravel()``), and solves ``H delta = grad`` via ``cholesky_ex`` + the shared
device-split SPD apply (matmul-series inverse on MPS). Per-chain convergence
freezing; bounded iterations.

Posterior draw ``beta* ~ N(beta_hat, H^{-1})`` (the vcov R uses), then softmax
prediction and inverse-CDF category sampling — the tensor counterparts of
``_draw.mvn_draw`` and ``_draw.sample_categories``.

Within one sweep step every chain shares the same target column, so ``K`` is fixed
across the batch — the "ragged K" concern only arises when batching *different*
columns together, which the sweep never does. All randomness flows through the
passed generator (Rule 6). Matches the CPU reference distributionally at the
GPU/FP32 tolerance tier.

Like the CPU method, a marginal-distribution fallback covers a non-converged fit;
here the per-step Cholesky is sync-free, so a genuinely degenerate fit surfaces as
a non-finite imputation caught by the backend's end-of-sweep guard.
"""

from __future__ import annotations

from pystatistics.core.exceptions import ValidationError

from pystatistics.mice.backends._gpu_linreg import (
    add_intercept,
    discrete_glm_compute_dtype,
    _cholesky_ridged,
)
from pystatistics.mice.backends._gpu_spd import apply_inv_factor_T, solve_spd

# Relative ridge on the Hessian (matches the numeric/logreg stabiliser); also
# tames separation, where the unpenalised multinomial MLE diverges.
_RIDGE = 1e-5
_MAX_NEWTON_ITER = 100
_NEWTON_TOL = 1e-8


def _log_probs(eta_nonref):
    """Full log-softmax with an appended zero column for the reference (last)
    class. ``eta_nonref`` (m, n, K-1) -> log-probs (m, n, K)."""
    import torch

    m, n, _ = eta_nonref.shape
    zeros = torch.zeros((m, n, 1), dtype=eta_nonref.dtype, device=eta_nonref.device)
    eta = torch.cat([eta_nonref, zeros], dim=2)
    return eta - torch.logsumexp(eta, dim=2, keepdim=True)


def _multinomial_hessian(Xa, pnr, ridge_diag):
    """Ridged block Hessian of the multinomial NLL. ``Xa`` (m, n, P), ``pnr``
    (m, n, K-1) non-reference probabilities. Returns (m, (K-1)P, (K-1)P) in the
    class-major block layout that matches ``coef.ravel()``."""
    import torch

    m, n, P = Xa.shape
    knr = pnr.shape[2]
    d = knr * P
    H = torch.zeros((m, d, d), dtype=Xa.dtype, device=Xa.device)
    for j in range(knr):
        pj = pnr[:, :, j]
        for k in range(j, knr):
            w = pj * (1.0 - pj) if j == k else -pj * pnr[:, :, k]
            block = (Xa * w.unsqueeze(-1)).transpose(1, 2) @ Xa       # (m, P, P)
            H[:, j * P:(j + 1) * P, k * P:(k + 1) * P] = block
            if k != j:
                H[:, k * P:(k + 1) * P, j * P:(j + 1) * P] = block.transpose(1, 2)
    eye = torch.eye(d, dtype=Xa.dtype, device=Xa.device)
    return H + ridge_diag[:, None, None] * eye


def batched_multinomial_newton(y_onehot, Xa, n_classes):
    """Batched Newton on the multinomial NLL. ``y_onehot`` (m, n, K), ``Xa``
    (m, n, P) WITH intercept. Returns ``(beta_hat (m, K-1, P), L)`` — the
    coefficient block and the Cholesky of the ridged Hessian at ``beta_hat``
    (for the posterior covariance ``H^{-1}``)."""
    import torch

    m, n, P = Xa.shape
    knr = n_classes - 1
    d = knr * P
    dtype, device = Xa.dtype, Xa.device
    beta = torch.zeros((m, knr, P), dtype=dtype, device=device)

    diag_scale = (Xa * Xa).sum(dim=1).mean(dim=1) / max(n, 1)         # (m,)
    ridge_diag = _RIDGE * diag_scale.clamp_min(1e-12)                # (m,)
    y_nr = y_onehot[:, :, :knr]

    converged = torch.zeros(m, dtype=torch.bool, device=device)
    for _ in range(_MAX_NEWTON_ITER):
        eta_nr = Xa @ beta.transpose(1, 2)                          # (m, n, K-1)
        pnr = _log_probs(eta_nr).exp()[:, :, :knr]
        resid = y_nr - pnr                                          # (m, n, K-1)
        # log-lik gradient, class-major: (m, P, K-1) -> (m, K-1, P) -> (m, d, 1)
        grad = (Xa.transpose(1, 2) @ resid).transpose(1, 2).reshape(m, d, 1)
        L = _cholesky_ridged(_multinomial_hessian(Xa, pnr, ridge_diag))
        delta = solve_spd(L, grad).squeeze(-1)                      # (m, d)

        small = delta.abs().amax(dim=1) < _NEWTON_TOL
        apply = ~converged
        beta = beta + apply[:, None, None] * delta.reshape(m, knr, P)
        converged = converged | (apply & small)
        if bool(converged.all()):
            break

    eta_nr = Xa @ beta.transpose(1, 2)
    pnr = _log_probs(eta_nr).exp()[:, :, :knr]
    L = _cholesky_ridged(_multinomial_hessian(Xa, pnr, ridge_diag))
    return beta, L


def _sample_categories(probs, gen):
    """Inverse-CDF sample one class per row from a (m, n, K) probability tensor.
    Robust to tiny negatives from an indefinite draw (clip + renormalise), the
    batched counterpart of ``_draw.sample_categories``. Returns (m, n) of class
    indices as a float tensor.

    Fail loud (Rule 1): a row with any non-finite probability yields ``NaN``,
    never a silent category 0 (under NaN, ``(cdf >= u)`` is all-False and
    ``argmax`` returns 0). Emitting NaN lets a degenerate fit reach the backend's
    end-of-sweep non-finite guard. Finite rows are unaffected."""
    import torch

    finite_row = torch.isfinite(probs).all(dim=2)                  # (m, n)
    safe = torch.where(finite_row.unsqueeze(2), probs, torch.zeros_like(probs))
    safe = safe.clamp_min(0.0)
    safe = safe / safe.sum(dim=2, keepdim=True).clamp_min(torch.finfo(safe.dtype).tiny)
    cdf = safe.cumsum(dim=2)
    u = torch.rand(
        safe.shape[:2] + (1,), generator=gen, dtype=safe.dtype, device=safe.device
    )
    idx = (cdf >= u).to(torch.int64).argmax(dim=2).to(safe.dtype)
    return torch.where(finite_row, idx, torch.full_like(idx, float("nan")))


def gpu_polyreg_impute(y_obs, X_obs, X_mis, gen, *, donors=None, n_classes=None):
    """Batched multinomial-logit imputation (R ``polyreg``).

    ``y_obs`` (m, n_obs) of 0..K-1 class indices, ``X_obs`` (m, n_obs, q),
    ``X_mis`` (m, n_mis, q); ``n_classes`` = K is required (passed by the sweep
    from the column's level count). Returns (m, n_mis) of 0..K-1 indices.
    ``donors`` is accepted and ignored (uniform method signature).
    """
    import torch

    if n_classes is None:
        raise ValidationError("gpu_polyreg_impute requires n_classes (number of levels)")
    K = int(n_classes)

    out_dtype = X_obs.dtype
    compute_dtype = discrete_glm_compute_dtype(X_obs.device, out_dtype)
    X_obs = X_obs.to(compute_dtype)
    X_mis = X_mis.to(compute_dtype)

    Xa = add_intercept(X_obs)
    m, n, P = Xa.shape
    knr = K - 1
    d = knr * P
    y_onehot = torch.zeros((m, n, K), dtype=Xa.dtype, device=Xa.device)
    y_onehot.scatter_(2, y_obs.to(torch.int64).unsqueeze(-1), 1.0)

    beta_hat, L = batched_multinomial_newton(y_onehot, Xa, K)
    z = torch.randn((m, d), generator=gen, dtype=Xa.dtype, device=Xa.device)
    beta_star = beta_hat.reshape(m, d) + apply_inv_factor_T(L, z.unsqueeze(-1)).squeeze(-1)
    beta_star = beta_star.reshape(m, knr, P)

    Xa_mis = add_intercept(X_mis)
    eta_nr = Xa_mis @ beta_star.transpose(1, 2)
    probs = _log_probs(eta_nr).exp()
    return _sample_categories(probs, gen).to(out_dtype)
