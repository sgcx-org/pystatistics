"""Torch/GPU batched building blocks for the EM E-step.

Extracted from ``_em_batched.py`` on 2026-04-20 to keep each file under
the 500-SLOC hard limit (Coding Bible rule 4). See module ``_em_batched``
for a compatibility shim that re-exports these symbols.

All functions take ``torch_mod`` (the imported ``torch`` module) as an
explicit parameter so CPU-only installs never trigger a torch import.

The ``chi_square_mcar_batched_torch`` function that lived here through
2.3.x was removed in 3.0.0 along with ``mom_mcar_test`` (which was its
only caller); the MCAR chi-square machinery now lives in Lacuna.
"""
from __future__ import annotations

from pystatistics.mvnmle.backends._em_batched_patterns import _BatchedPatternIndex


def _e_step_full_torch(
    mu, sigma, index, data_padded, obs_pattern_id,
    n_per_pattern, obs_idx_t, obs_mask_t, mis_idx_t, mis_mask_t,
    eye_oo, torch_mod, device, dtype,
):
    """Fully observation-level batched E-step on torch/GPU.

    All inputs are pre-resident torch tensors; the per-iteration work
    is:
      - 1 batched Cholesky of (P, v_obs_max, v_obs_max)
      - 1 batched cholesky_solve for the regression betas
      - 1 scatter to build (P, v, v) fill matrix F and (P, v) bias b
      - 1 gather to (N, v, v) + (N, v) by obs_pattern_id
      - 1 bmm of (N, v, v) @ (N, v, 1) → (N, v) filled data
      - 2 gemms: T1 = sum, T2 = x_filled.T @ x_filled
      - 1 batched scatter + sum for cond_cov correction to T2

    Total: ~10 kernel launches per iteration regardless of P or N.
    """
    torch = torch_mod
    P = index.n_patterns
    v_obs_max = index.v_obs_max
    v_mis_max = index.v_mis_max
    v = sigma.shape[0]

    # Fast path: complete data, no missing values anywhere.
    if v_mis_max == 0:
        T1 = data_padded.sum(dim=0)
        T2 = data_padded.T @ data_padded
        return T1, T2

    # --- Batched per-pattern conditional regression parameters -------
    # sigma_oo via 2-D fancy indexing; identity-padded in unused slots.
    row_idx = obs_idx_t.unsqueeze(-1)
    col_idx = obs_idx_t.unsqueeze(-2)
    sigma_oo = sigma[row_idx, col_idx]
    mask_oo = obs_mask_t.unsqueeze(-1) & obs_mask_t.unsqueeze(-2)
    sigma_oo = torch.where(mask_oo, sigma_oo, eye_oo)

    # Cholesky with ridge fallback. See numpy path comment above.
    try:
        L_oo = torch.linalg.cholesky(sigma_oo)
    except torch._C._LinAlgError:
        import warnings
        ridge = 1e-10
        warnings.warn(
            f"_e_step_full_torch: per-pattern sigma_oo indefinite on "
            f"GPU path; retrying Cholesky with diagonal ridge {ridge:.0e}.",
            UserWarning, stacklevel=3,
        )
        eye_full = eye_oo if eye_oo.shape == sigma_oo.shape else torch.eye(
            v_obs_max, device=device, dtype=dtype
        ).expand_as(sigma_oo)
        L_oo = torch.linalg.cholesky(sigma_oo + ridge * eye_full)

    # sigma_mo: missing rows × observed cols.
    mrow_idx = mis_idx_t.unsqueeze(-1)
    sigma_mo = sigma[mrow_idx, col_idx]
    valid_mo = mis_mask_t.unsqueeze(-1) & obs_mask_t.unsqueeze(-2)
    sigma_mo = torch.where(valid_mo, sigma_mo,
                           torch.zeros((), device=device, dtype=dtype))

    # sigma_mm.
    mcol_idx = mis_idx_t.unsqueeze(-2)
    sigma_mm = sigma[mrow_idx, mcol_idx]
    valid_mm = mis_mask_t.unsqueeze(-1) & mis_mask_t.unsqueeze(-2)
    sigma_mm = torch.where(valid_mm, sigma_mm,
                           torch.zeros((), device=device, dtype=dtype))

    # Solve for beta: (P, v_obs_max, v_mis_max)^T in the sense that
    # beta^T = cholesky_solve(sigma_om, L_oo). Note sigma_om is transpose.
    sigma_om = sigma_mo.transpose(-1, -2)
    beta_T = torch.cholesky_solve(sigma_om, L_oo)     # (P, v_obs_max, v_mis_max)
    beta = beta_T.transpose(-1, -2)                   # (P, v_mis_max, v_obs_max)
    cond_cov = sigma_mm - torch.matmul(beta, sigma_om)  # (P, v_mis_max, v_mis_max)

    # --- Build full-size (P, v, v) fill matrix F and (P, v) bias b ----
    # For pattern k: F_k[m, o] = beta_k[m_slot, o_slot] for valid (m, o);
    #                b_k[m]   = mu[m] - beta_k[m_slot, :] @ mu_obs_k.
    #
    # Caution: padded slots in (mis_idx_t, obs_idx_t) point at index 0
    # by default. A naïve scatter assignment would repeatedly write
    # zeros to F[:, 0, 0] / b[:, 0] and wipe out any real beta that
    # legitimately targets (m=0, o=0). Mask-select to valid entries
    # only before scatter.
    F = torch.zeros((P, v, v), device=device, dtype=dtype)
    b = torch.zeros((P, v), device=device, dtype=dtype)

    k_ax = torch.arange(P, device=device)
    valid_mo_flat = mis_mask_t.unsqueeze(-1) & obs_mask_t.unsqueeze(-2)  # (P, v_mis_max, v_obs_max)
    k_bcast = k_ax[:, None, None].expand(-1, v_mis_max, v_obs_max)
    mis_bcast = mis_idx_t[:, :, None].expand(-1, -1, v_obs_max)
    obs_bcast = obs_idx_t[:, None, :].expand(-1, v_mis_max, -1)

    F[k_bcast[valid_mo_flat],
      mis_bcast[valid_mo_flat],
      obs_bcast[valid_mo_flat]] = beta[valid_mo_flat]

    # Bias: mu[mis] - beta @ mu_obs_k. Compute mu_obs_per_pattern via
    # gather, zero the padded slots, multiply.
    mu_obs_per_pattern = mu[obs_idx_t] * obs_mask_t.to(dtype)  # (P, v_obs_max)
    beta_mu_obs = torch.matmul(
        beta, mu_obs_per_pattern.unsqueeze(-1)
    ).squeeze(-1)  # (P, v_mis_max)
    b_values = mu[mis_idx_t] - beta_mu_obs  # (P, v_mis_max)

    k_b_bcast = k_ax[:, None].expand(-1, v_mis_max)
    b[k_b_bcast[mis_mask_t], mis_idx_t[mis_mask_t]] = b_values[mis_mask_t]

    # --- Per-observation scatter + batched matmul --------------------
    F_per_obs = F[obs_pattern_id]                   # (N, v, v)
    b_per_obs = b[obs_pattern_id]                   # (N, v)
    fill = torch.matmul(
        F_per_obs, data_padded.unsqueeze(-1)
    ).squeeze(-1)                                    # (N, v)
    x_filled = data_padded + fill + b_per_obs       # (N, v)

    # --- Accumulate T1, T2 via dense gemms ---------------------------
    T1 = x_filled.sum(dim=0)                        # (v,)
    T2 = x_filled.T @ x_filled                      # (v, v)

    # Conditional-covariance correction: Σ_k n_k * pad(cond_cov_k) on
    # the missing-missing block. Same masked-scatter discipline as F/b
    # above — padded slots point at mis_idx=0 and must not be written.
    cond_cov_full = torch.zeros((P, v, v), device=device, dtype=dtype)
    valid_mm_flat = mis_mask_t.unsqueeze(-1) & mis_mask_t.unsqueeze(-2)  # (P, v_mis_max, v_mis_max)
    k_cc_bcast = k_ax[:, None, None].expand(-1, v_mis_max, v_mis_max)
    mis_row_bcast = mis_idx_t[:, :, None].expand(-1, -1, v_mis_max)
    mis_col_bcast = mis_idx_t[:, None, :].expand(-1, v_mis_max, -1)
    cond_cov_full[k_cc_bcast[valid_mm_flat],
                  mis_row_bcast[valid_mm_flat],
                  mis_col_bcast[valid_mm_flat]] = cond_cov[valid_mm_flat]

    T2 = T2 + torch.einsum('p,pij->ij', n_per_pattern, cond_cov_full)

    return T1, T2


def _loglik_full_torch(
    mu, sigma, index, data_padded, obs_pattern_id, n_per_pattern,
    obs_idx_t, obs_mask_t, eye_oo, torch_mod, device, dtype,
):
    """Fully batched observed-data log-likelihood on torch/GPU.

    Mirrors ``compute_loglik_batched_np``: one batched Cholesky over
    patterns for log-determinants; one batched solve across all N
    observations for the quadratic form.
    """
    torch = torch_mod

    row_idx = obs_idx_t.unsqueeze(-1)
    col_idx = obs_idx_t.unsqueeze(-2)
    sigma_oo = sigma[row_idx, col_idx]
    mask_oo = obs_mask_t.unsqueeze(-1) & obs_mask_t.unsqueeze(-2)
    sigma_oo = torch.where(mask_oo, sigma_oo, eye_oo)

    # Cholesky with ridge fallback for indefinite per-pattern sub-blocks.
    # See numpy path comment.
    try:
        L_oo = torch.linalg.cholesky(sigma_oo)
    except torch._C._LinAlgError:
        import warnings
        ridge = 1e-10
        warnings.warn(
            f"_loglik_full_batched_torch: per-pattern sigma_oo "
            f"indefinite; retrying Cholesky with ridge {ridge:.0e}.",
            UserWarning, stacklevel=3,
        )
        v_obs_max_local = sigma_oo.shape[-1]
        eye_full = torch.eye(
            v_obs_max_local, device=sigma_oo.device, dtype=sigma_oo.dtype
        ).expand_as(sigma_oo)
        L_oo = torch.linalg.cholesky(sigma_oo + ridge * eye_full)
    log_diag = torch.log(torch.diagonal(L_oo, dim1=-2, dim2=-1))
    logdet_per_pattern = 2.0 * torch.sum(log_diag * obs_mask_t.to(dtype), dim=-1)

    # Observation-level gather.
    per_obs_obs_idx = obs_idx_t[obs_pattern_id]           # (N, v_obs_max)
    per_obs_obs_mask = obs_mask_t[obs_pattern_id].to(dtype)

    N = data_padded.shape[0]
    N_arange = torch.arange(N, device=device).unsqueeze(-1)
    y_gathered = data_padded[N_arange, per_obs_obs_idx]   # (N, v_obs_max)
    mu_gathered = mu[per_obs_obs_idx] * per_obs_obs_mask  # (N, v_obs_max)
    centered = (y_gathered - mu_gathered) * per_obs_obs_mask

    L_per_obs = L_oo[obs_pattern_id]                       # (N, v_obs_max, v_obs_max)
    z = torch.linalg.solve_triangular(
        L_per_obs, centered.unsqueeze(-1), upper=False,
    ).squeeze(-1)                                          # (N, v_obs_max)
    quad_total = (z * z).sum()

    logdet_sum = (n_per_pattern * logdet_per_pattern).sum()

    return -0.5 * logdet_sum - 0.5 * quad_total


def compute_conditional_parameters_torch(
    mu, sigma, index: _BatchedPatternIndex, torch_mod, device, dtype,
):
    """Same as :func:`compute_conditional_parameters_np` but on torch.

    Accepts torch tensors for ``mu``, ``sigma`` and returns torch
    tensors. ``device`` and ``dtype`` are propagated through all
    intermediate allocations. ``torch_mod`` is the imported ``torch``
    module (injected so CPU-only installs don't pay the import cost
    unnecessarily).
    """
    P = index.n_patterns
    v_obs_max = index.v_obs_max
    v_mis_max = index.v_mis_max

    obs_idx = torch_mod.as_tensor(index.obs_idx, device=device, dtype=torch_mod.long)
    mis_idx = torch_mod.as_tensor(index.mis_idx, device=device, dtype=torch_mod.long)
    obs_mask = torch_mod.as_tensor(index.obs_mask, device=device, dtype=torch_mod.bool)
    mis_mask = torch_mod.as_tensor(index.mis_mask, device=device, dtype=torch_mod.bool)

    row_idx = obs_idx.unsqueeze(-1)              # (P, v_obs_max, 1)
    col_idx = obs_idx.unsqueeze(-2)              # (P, 1, v_obs_max)
    sigma_oo = sigma[row_idx, col_idx]           # (P, v_obs_max, v_obs_max)

    mask_oo = obs_mask.unsqueeze(-1) & obs_mask.unsqueeze(-2)
    eye_oo = torch_mod.eye(v_obs_max, device=device, dtype=dtype).expand(P, -1, -1)
    sigma_oo = torch_mod.where(mask_oo, sigma_oo, eye_oo)

    mrow_idx = mis_idx.unsqueeze(-1)             # (P, v_mis_max, 1)
    sigma_mo = sigma[mrow_idx, col_idx]          # (P, v_mis_max, v_obs_max)
    valid_mo = mis_mask.unsqueeze(-1) & obs_mask.unsqueeze(-2)
    sigma_mo = torch_mod.where(valid_mo, sigma_mo,
                               torch_mod.zeros((), device=device, dtype=dtype))

    mcol_idx = mis_idx.unsqueeze(-2)             # (P, 1, v_mis_max)
    sigma_mm = sigma[mrow_idx, mcol_idx]         # (P, v_mis_max, v_mis_max)
    valid_mm = mis_mask.unsqueeze(-1) & mis_mask.unsqueeze(-2)
    sigma_mm = torch_mod.where(valid_mm, sigma_mm,
                               torch_mod.zeros((), device=device, dtype=dtype))

    # Cholesky with ridge fallback for indefinite per-pattern sub-blocks.
    # See e_step_full_batched_np / _e_step_full_torch for rationale.
    try:
        L_oo = torch_mod.linalg.cholesky(sigma_oo)
    except torch_mod._C._LinAlgError:
        import warnings
        ridge = 1e-10
        warnings.warn(
            f"e_step_batched_torch: per-pattern sigma_oo indefinite; "
            f"retrying Cholesky with ridge {ridge:.0e}.",
            UserWarning, stacklevel=3,
        )
        L_oo = torch_mod.linalg.cholesky(sigma_oo + ridge * eye_oo)
    sigma_om = sigma_mo.transpose(-1, -2)
    beta_T = torch_mod.cholesky_solve(sigma_om, L_oo)  # (P, v_obs_max, v_mis_max)
    beta = beta_T.transpose(-1, -2)

    cond_cov = sigma_mm - torch_mod.matmul(beta, sigma_om)

    return beta, cond_cov, obs_mask, mis_mask
