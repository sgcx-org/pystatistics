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

Applying ``L^{-1}`` is device-split via the shared dispatch in
``core.compute.linalg``: on MPS (whose triangular-solve kernels are ~100-300x
slower than matmul) ``L`` is inverted with the matmul-only blocked inverse and
applied by matmul; on CUDA/CPU the two back-substitutions (which share ``L'``)
are stacked into one fast triangular solve. The
chi-square is built from standard normals (``df`` is a non-negative integer) so
the only randomness source is the seeded ``torch.Generator`` — no reliance on
``torch.distributions``, which ignores explicit generators.

All randomness flows through the passed generator (CLAUDE.md Rule 6). Results
match the CPU reference distributionally, at the GPU/FP32 tolerance tier — not
bit-for-bit (different RNG, single precision).
"""

from __future__ import annotations

from dataclasses import dataclass

from pystatistics.core.compute.linalg import (
    batched_tri_inv_series,
    use_blocked_inverse,
)

# Matches the CPU ridge (methods/_linreg.py) — a tiny relative penalty keeping
# the batched Gram matrices invertible under FP32.
_DEFAULT_RIDGE = 1e-5

# MPS dispatch threshold for the matmul-series inverse vs solve_triangular.
# torch's MPS solve_triangular is a single but ~250x-slower-than-CUDA kernel;
# the series inverse is ~log2(p) fast matmuls but more launches. In a per-step
# sweep the small-n regime is dispatch-bound, so for small n_obs the single
# solve_triangular kernel still edges out the multi-op series; above ~n_obs 3000
# the series wins (and keeps winning at scale). Empirically tuned (m=100,
# q+1~20); a speed heuristic, not a correctness boundary, so a fixed threshold
# is fine and may want per-device tuning. CUDA/CPU always use solve_triangular
# (fast there) — see use_blocked_inverse, which is MPS-only.
_SERIES_INV_MIN_NOBS = 3000


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

    # Factor the ridged Gram once: G = L L'. We need beta_hat = G^-1 X'y and the
    # posterior factor A = L^-T (A A' = G^-1) — never the dense inverse, never
    # solve/inv/eigh. The Cholesky is sync-free; degenerate (collinear) input is
    # caught by the backend's end-of-sweep non-finite guard, not a per-step check.
    L = _cholesky_ridged(G)                          # (m, q+1, q+1) lower
    Lt = L.transpose(1, 2)

    # Draw both normal vectors up front, preserving the chi-then-beta RNG order.
    df = max(n_obs - n_params, 1)
    z_chi = torch.randn((m, df), generator=gen, dtype=dtype, device=device)
    z_beta = torch.randn((m, n_params), generator=gen, dtype=dtype, device=device)

    # HOW we apply L^-1 is device- and size-split: MPS's triangular-solve kernel
    # is ~250x slower than its matmul, so above a size threshold we invert L with
    # the matmul-series inverse (Neumann doubling + 1 Newton step) and apply by
    # matmul. Below it (small n_obs) the sweep is dispatch-bound and the single
    # solve_triangular kernel still edges out the multi-op series; on CUDA/CPU
    # triangular solves are fast — so both keep solve_triangular there.
    Xty = Xt @ y_obs.unsqueeze(-1)                   # (m, q+1, 1)
    if use_blocked_inverse(L) and n_obs >= _SERIES_INV_MIN_NOBS:
        Linv = batched_tri_inv_series(L)             # L^-1, matmul-series (MPS)
        Linvt = Linv.transpose(1, 2)
        beta_hat = (Linvt @ (Linv @ Xty)).squeeze(-1)    # G^-1 X'y via matmul
        Az = (Linvt @ z_beta.unsqueeze(-1)).squeeze(-1)  # A z, A = L^-T
    else:
        # The two upper solves share L', so stack them into one wide-RHS solve:
        #   beta_hat = L^-T (L^-1 X'y),  A z = L^-T z_beta.
        fwd = torch.linalg.solve_triangular(L, Xty, upper=False)
        upper = torch.linalg.solve_triangular(
            Lt, torch.cat([fwd, z_beta.unsqueeze(-1)], dim=2), upper=True
        )                                            # (m, q+1, 2)
        beta_hat = upper[..., 0]                      # (m, q+1)
        Az = upper[..., 1]                           # (m, q+1)

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


def _cholesky_ridged(G):
    """Lower Cholesky of each ridged Gram in the batch — sync-free.

    ``G = X'X + ridge·mean·I`` (ridge > 0) is positive definite in exact
    arithmetic. We symmetrize and add a *tiny* unconditional jitter
    (``1e-8·scale``, three orders below the ridge, so statistically negligible)
    to absorb FP32 rounding, then factor with ``cholesky_ex``.

    Crucially there is **no per-step host sync**: the previous version called
    ``.item()`` and ``torch.any(info)`` on every sweep step (≈2 GPU↔CPU
    round-trips × ``maxit·p`` steps), which dominated the small-n sweep. Here we
    trust the ridge and defer fail-loud to the backend's single end-of-sweep
    non-finite guard: genuinely degenerate (collinear) predictors yield a
    non-finite factor that propagates to the imputations and is caught there —
    one sync per sweep instead of hundreds. (Avoiding ``eigh``/``solve`` also
    keeps this valid and fast on MPS.)
    """
    import torch

    Gs = 0.5 * (G + G.transpose(1, 2))
    eye = torch.eye(Gs.shape[1], dtype=Gs.dtype, device=Gs.device)
    diag = torch.diagonal(Gs, dim1=1, dim2=2)
    jitter = diag.mean(dim=1).clamp_min(1.0) * 1e-8          # (m,), tensor (no .item())
    L, _ = torch.linalg.cholesky_ex(Gs + jitter[:, None, None] * eye)
    return L
