"""
Batched Bayesian linear-regression posterior draw on the GPU.

This is the GPU counterpart of ``methods/_linreg.py``: the same Bayesian draw,
but vectorized over the ``m`` imputation chains as the leading batch dimension.
At a given sweep step every chain is fitting the *same* target column on the
*same* observed rows, but with different predictor values (each chain imputed
the other columns differently), so we solve ``m`` independent linear systems at
once with batched kernels. This path is shared by the CUDA and MPS devices.

Posterior draw (identical model to the CPU path), via a single Cholesky of the
ridged Gram ``G = X'X + ridge`` — no matrix inverse, no ``eigh``:

    G = L L'                                          batched Cholesky
    beta_hat = G^{-1} X'y = L^{-T} L^{-1} X'y         forward + back substitution
    df       = max(n_obs - n_params, 1)               integer, shared by chains
    sigma*   = sqrt( RSS / chi2(df) )                 chi2 = sum of df squared N(0,1)
    beta*    = beta_hat + sigma* * (L^{-T} z)         A = L^{-T}, A A' = G^{-1}

The two back-substitutions share ``L'`` and are stacked into one solve. The
chi-square is built from standard normals (``df`` is a non-negative integer) so
the only randomness source is the seeded ``torch.Generator`` — no reliance on
``torch.distributions``, which ignores explicit generators.

All randomness flows through the passed generator (CLAUDE.md Rule 6). Results
match the CPU reference distributionally, at the GPU/FP32 tolerance tier — not
bit-for-bit (different RNG, single precision).
"""

from __future__ import annotations

from dataclasses import dataclass

from pystatistics.core.exceptions import ValidationError

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
    G = XtX + ridge * diag_mean[:, None, None] * eye  # ridged Gram, SPD (ridge>0)

    # Forward-Cholesky parameterization: factor G = L L' once and drive every
    # downstream solve through L. We never form the Gram inverse and never call
    # solve/inv/eigh — inv/eigh are slow or unimplemented on MPS, and computing
    # both a solve and an inverse (as the inverse path did) is wasteful on CUDA
    # too. cholesky_ex + triangular solves is faster on both devices.
    L = _safe_cholesky_spd(G)                        # (m, q+1, q+1) lower
    Lt = L.transpose(1, 2)

    # Draw both normal vectors up front, preserving the chi-then-beta RNG order.
    df = max(n_obs - n_params, 1)
    z_chi = torch.randn((m, df), generator=gen, dtype=dtype, device=device)
    z_beta = torch.randn((m, n_params), generator=gen, dtype=dtype, device=device)

    Xty = Xt @ y_obs.unsqueeze(-1)                   # (m, q+1, 1)
    fwd = torch.linalg.solve_triangular(L, Xty, upper=False)  # forward subst
    # The two upper solves share L', so stack them into one wide-RHS back-subst:
    #   beta_hat = L^{-T} fwd        (= G^{-1} X'y, the normal-equation solution)
    #   A z      = L^{-T} z_beta     (posterior factor A = L^{-T}, A A' = G^{-1})
    upper = torch.linalg.solve_triangular(
        Lt, torch.cat([fwd, z_beta.unsqueeze(-1)], dim=2), upper=True
    )                                                # (m, q+1, 2)
    beta_hat = upper[..., 0]                          # (m, q+1)
    Az = upper[..., 1]                               # (m, q+1)

    resid = y_obs - (Xa @ beta_hat.unsqueeze(-1)).squeeze(-1)  # (m, n_obs)
    rss = (resid * resid).sum(dim=1)                 # (m,)
    # chi2(df) = sum of df squared standard normals (df is an integer).
    chi = (z_chi * z_chi).sum(dim=1).clamp_min(torch.finfo(dtype).tiny)  # (m,)
    sigma_draw = torch.where(
        rss > 0, torch.sqrt(rss / chi), torch.zeros_like(rss)
    )                                                # (m,)

    beta_draw = beta_hat + sigma_draw[:, None] * Az
    return BatchedLinRegDraw(
        beta_hat=beta_hat, beta_draw=beta_draw, sigma_draw=sigma_draw
    )


def _safe_cholesky_spd(G):
    """Lower Cholesky of each SPD matrix in a batch, with escalating jitter.

    ``G`` is a ridged Gram (``X'X + ridge·mean·I``, ridge > 0): positive
    definite in exact arithmetic. FP32 rounding can still leave one marginally
    indefinite, so we symmetrize and add escalating jitter to the *whole batch*
    (cheap, keeps the op batched) until ``cholesky_ex`` succeeds for every
    matrix.

    Factoring ``G`` directly is well conditioned — unlike factoring the Gram
    *inverse*, which the previous path did and which needed an eigenvalue-clip
    last resort. That last resort is gone: if jitter cannot rescue a matrix here
    the predictors are genuinely degenerate, so we fail loud (Rule 1) rather
    than return a silently clipped factor. Avoiding ``eigh`` also keeps this
    path valid on MPS, where ``eigh`` is unimplemented.
    """
    import torch

    Gs = 0.5 * (G + G.transpose(1, 2))
    eye = torch.eye(Gs.shape[1], dtype=Gs.dtype, device=Gs.device)
    diag = torch.diagonal(Gs, dim1=1, dim2=2)
    scale = diag.mean().clamp_min(1.0).item()

    jitter = 0.0
    for _ in range(8):
        M = Gs if jitter == 0.0 else Gs + jitter * eye
        L, info = torch.linalg.cholesky_ex(M)
        if not torch.any(info):
            return L
        jitter = scale * (1e-8 if jitter == 0.0 else 10.0 * jitter)

    raise ValidationError(
        "Batched Cholesky of the predictor Gram failed after ridge + escalating "
        "jitter: the predictors for an imputation target are near-collinear "
        "(degenerate design). Inspect the data for redundant/constant columns."
    )
