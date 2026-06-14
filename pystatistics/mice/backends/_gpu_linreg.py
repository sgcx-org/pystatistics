"""
Batched Bayesian linear-regression posterior draw on the GPU.

This is the GPU counterpart of ``methods/_linreg.py``: the same Bayesian draw,
but vectorized over the ``m`` imputation chains as the leading batch dimension.
At a given sweep step every chain is fitting the *same* target column on the
*same* observed rows, but with different predictor values (each chain imputed
the other columns differently), so we solve ``m`` independent linear systems at
once with batched cuBLAS/cuSOLVER kernels.

Posterior draw (identical model to the CPU path):

    beta_hat = (X'X + ridge)^{-1} X'y                 batched solve
    df       = max(n_obs - n_params, 1)               integer, shared by chains
    sigma*   = sqrt( RSS / chi2(df) )                 chi2 = sum of df squared N(0,1)
    beta*    = beta_hat + sigma* * L z                L L' = (X'X + ridge)^{-1}

The chi-square is built from standard normals (``df`` is a non-negative integer)
so the only randomness source is the seeded ``torch.Generator`` — no reliance on
``torch.distributions``, which ignores explicit generators.

All randomness flows through the passed generator (CLAUDE.md Rule 6). Results
match the CPU reference distributionally, at the GPU/FP32 tolerance tier — not
bit-for-bit (different RNG, single precision).
"""

from __future__ import annotations

from dataclasses import dataclass

# Matches the CPU ridge (methods/_linreg.py) — a tiny relative penalty keeping
# the batched Gram matrices invertible under FP32.
_DEFAULT_RIDGE = 1e-5


@dataclass
class BatchedLinRegDraw:
    """Batched point estimate + posterior draw (leading dim = m chains)."""

    beta_hat: "object"    # (m, q+1) tensor
    beta_draw: "object"   # (m, q+1) tensor
    sigma_draw: "object"  # (m,) tensor


def add_intercept(X):
    """Prepend a ones column to a batched predictor tensor (m, n, q) -> (m,n,q+1)."""
    import torch

    m, n, _ = X.shape
    ones = torch.ones((m, n, 1), dtype=X.dtype, device=X.device)
    return torch.cat([ones, X], dim=2)


def batched_bayes_linreg_draw(
    y_obs,
    X_obs,
    gen,
    ridge: float = _DEFAULT_RIDGE,
) -> BatchedLinRegDraw:
    """Draw once from the Gaussian linear-model posterior for every chain.

    Parameters
    ----------
    y_obs : (m, n_obs) tensor
        Observed responses (identical across chains, batched for uniform ops).
    X_obs : (m, n_obs, q) tensor
        Observed predictors WITHOUT an intercept column (added internally).
    gen : torch.Generator
        Sole randomness source, on the same device as the tensors.
    ridge : float
        Relative ridge penalty on the diagonal of X'X.
    """
    import torch

    Xa = add_intercept(X_obs)                       # (m, n_obs, q+1)
    m, n_obs, n_params = Xa.shape
    dtype, device = Xa.dtype, Xa.device

    Xt = Xa.transpose(1, 2)                          # (m, q+1, n_obs)
    XtX = Xt @ Xa                                    # (m, q+1, q+1)

    diag = torch.diagonal(XtX, dim1=1, dim2=2)       # (m, q+1)
    diag_mean = diag.mean(dim=1).clamp_min(torch.finfo(dtype).tiny)  # (m,)
    eye = torch.eye(n_params, dtype=dtype, device=device)
    XtX_ridge = XtX + ridge * diag_mean[:, None, None] * eye

    Xty = Xt @ y_obs.unsqueeze(-1)                   # (m, q+1, 1)
    beta_hat = torch.linalg.solve(XtX_ridge, Xty).squeeze(-1)  # (m, q+1)

    resid = y_obs - (Xa @ beta_hat.unsqueeze(-1)).squeeze(-1)  # (m, n_obs)
    rss = (resid * resid).sum(dim=1)                 # (m,)
    df = max(n_obs - n_params, 1)

    # chi2(df) = sum of df squared standard normals (df is an integer).
    z_chi = torch.randn((m, df), generator=gen, dtype=dtype, device=device)
    chi = (z_chi * z_chi).sum(dim=1).clamp_min(torch.finfo(dtype).tiny)  # (m,)
    sigma_draw = torch.where(
        rss > 0, torch.sqrt(rss / chi), torch.zeros_like(rss)
    )                                                # (m,)

    V = torch.linalg.inv(XtX_ridge)                  # (m, q+1, q+1)
    L = _batched_safe_cholesky(V)                    # (m, q+1, q+1), L L' = V
    z_beta = torch.randn((m, n_params), generator=gen, dtype=dtype, device=device)
    beta_draw = beta_hat + sigma_draw[:, None] * (L @ z_beta.unsqueeze(-1)).squeeze(-1)

    return BatchedLinRegDraw(
        beta_hat=beta_hat, beta_draw=beta_draw, sigma_draw=sigma_draw
    )


def _batched_safe_cholesky(V):
    """Lower Cholesky of each (symmetric PSD) matrix in a batch, with jitter.

    ``V`` is a batch of inverse-Gram matrices — positive definite in exact
    arithmetic, but FP32 rounding can leave one marginally indefinite. We
    symmetrize and add escalating jitter to the *whole batch* (cheap, keeps the
    op batched) until ``cholesky_ex`` reports success for every matrix.
    """
    import torch

    Vs = 0.5 * (V + V.transpose(1, 2))
    eye = torch.eye(Vs.shape[1], dtype=Vs.dtype, device=Vs.device)
    diag = torch.diagonal(Vs, dim1=1, dim2=2)
    scale = diag.mean().clamp_min(1.0).item()

    jitter = 0.0
    for _ in range(8):
        M = Vs if jitter == 0.0 else Vs + jitter * eye
        L, info = torch.linalg.cholesky_ex(M)
        if not torch.any(info):
            return L
        jitter = scale * (1e-8 if jitter == 0.0 else 10.0 * jitter)

    # Documented last resort (Rule 1): eigenvalue-clipped PSD factor. Reached
    # only if a matrix is badly degenerate despite ridge + jitter.
    w, Q = torch.linalg.eigh(Vs)
    w = w.clamp_min(0.0)
    return Q @ torch.diag_embed(torch.sqrt(w))
