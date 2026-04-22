"""NumPy batched building blocks for the EM E-step.

Extracted from ``_em_batched.py`` on 2026-04-20 to keep each file under
the 500-SLOC hard limit (Coding Bible rule 4). See module ``_em_batched``
for a compatibility shim that re-exports these symbols.

All functions are FP64 and bit-faithful to the scalar reference
implementation on CPU.

The ``chi_square_mcar_batched_np`` function that lived here through
2.3.x was removed in 3.0.0 along with ``mom_mcar_test`` (which was its
only caller); the MCAR chi-square machinery now lives in Lacuna.
"""
from __future__ import annotations

import numpy as np

from pystatistics.mvnmle.backends._em_batched_patterns import _BatchedPatternIndex


def compute_conditional_parameters_np(
    mu: np.ndarray,
    sigma: np.ndarray,
    index: _BatchedPatternIndex,
) -> tuple:
    """Batched computation of per-pattern conditional regression matrices.

    For each pattern k with observed indices O_k and missing indices M_k,
    compute in a single batched Cholesky + batched solve:

        beta_k = Sigma[M_k, O_k] @ Sigma[O_k, O_k]^{-1}
        cond_cov_k = Sigma[M_k, M_k] - beta_k @ Sigma[O_k, M_k]

    The per-pattern results are padded to ``(P, v_mis_max, v_obs_max)``
    and ``(P, v_mis_max, v_mis_max)`` respectively; the caller must
    apply ``index.mis_mask`` / ``index.obs_mask`` when consuming slices.

    Uses NumPy's batched linalg (cholesky / solve_triangular support
    leading batch dimensions since NumPy 1.8 / 1.13). Runs on CPU.

    Parameters
    ----------
    mu : (v,) float64
    sigma : (v, v) float64
    index : _BatchedPatternIndex

    Returns
    -------
    beta_batched : (P, v_mis_max, v_obs_max)
    cond_cov_batched : (P, v_mis_max, v_mis_max)
    """
    P = index.n_patterns
    v_obs_max = index.v_obs_max
    v_mis_max = index.v_mis_max

    # Gather sigma_oo via 2D advanced indexing.
    # row_idx[k, i, j] = obs_idx[k, i]; col_idx[k, i, j] = obs_idx[k, j]
    row_idx = index.obs_idx[:, :, None]          # (P, v_obs_max, 1)
    col_idx = index.obs_idx[:, None, :]          # (P, 1, v_obs_max)
    sigma_oo = sigma[row_idx, col_idx]           # (P, v_obs_max, v_obs_max)

    # Replace padded rows/cols with identity so cholesky stays well-defined.
    mask_oo = index.obs_mask[:, :, None] & index.obs_mask[:, None, :]
    eye_oo = np.broadcast_to(
        np.eye(v_obs_max, dtype=sigma.dtype), sigma_oo.shape,
    )
    sigma_oo = np.where(mask_oo, sigma_oo, eye_oo)

    # Gather sigma_mo: rows are missing, cols are observed.
    mrow_idx = index.mis_idx[:, :, None]         # (P, v_mis_max, 1)
    sigma_mo = sigma[mrow_idx, col_idx]          # (P, v_mis_max, v_obs_max)
    # Zero out invalid rows/cols (missing side: mis_mask; observed side: obs_mask)
    valid_mo = index.mis_mask[:, :, None] & index.obs_mask[:, None, :]
    sigma_mo = np.where(valid_mo, sigma_mo, 0.0)

    # Gather sigma_mm.
    mcol_idx = index.mis_idx[:, None, :]         # (P, 1, v_mis_max)
    sigma_mm = sigma[mrow_idx, mcol_idx]         # (P, v_mis_max, v_mis_max)
    valid_mm = index.mis_mask[:, :, None] & index.mis_mask[:, None, :]
    sigma_mm = np.where(valid_mm, sigma_mm, 0.0)

    # (Note: an earlier revision computed a batched Cholesky here whose
    # factor was never used downstream. Removed — np.linalg.solve below
    # handles the per-pattern solve directly. Kept the pinv fallback
    # because np.linalg.solve still fails on strictly-singular sigma_oo,
    # which real tabular data with integer-encoded categoricals can
    # produce at the per-pattern sub-block level.)

    # Batched solve: beta^T = Sigma_oo^{-1} @ sigma_om = solve(sigma_oo, sigma_om)
    # sigma_om = sigma_mo^T  →  (P, v_obs_max, v_mis_max)
    sigma_om = np.swapaxes(sigma_mo, -1, -2)
    try:
        beta_T = np.linalg.solve(sigma_oo, sigma_om)
    except np.linalg.LinAlgError:
        # Per-pattern sigma_oo sub-block is singular. Fall back to pinv
        # for this batch. Issue a warning so the event is visible.
        import warnings
        warnings.warn(
            "e_step_batched_np: at least one per-pattern sigma_oo "
            "sub-block is numerically singular; falling back to "
            "Moore-Penrose pseudo-inverse for the batch.",
            UserWarning, stacklevel=3,
        )
        beta_T = np.matmul(np.linalg.pinv(sigma_oo), sigma_om)
    beta = np.swapaxes(beta_T, -1, -2)            # (P, v_mis_max, v_obs_max)

    # cond_cov = sigma_mm - beta @ sigma_om = sigma_mm - sigma_mo @ beta^T
    cond_cov = sigma_mm - np.matmul(beta, sigma_om)

    return beta, cond_cov


def e_step_full_batched_np(
    mu: np.ndarray,
    sigma: np.ndarray,
    index: _BatchedPatternIndex,
    n_vars: int,
) -> tuple:
    """Fully batched E-step in N-parallel form (no per-pattern Python loop).

    Uses observation-level padded representation from
    ``_BatchedPatternIndex`` to apply each pattern's conditional
    regression to its observations in a single batched matmul, then
    accumulates sufficient statistics with two dense gemms.

    Handles three edge cases inline:
      * Complete pattern (no missing) → beta/cond_cov have no missing
        rows, so fill and correction are zero. Contributes via the
        raw observed data through data_padded.
      * All-missing pattern (no observed) → conditional fill = mu,
        conditional covariance = sigma. Handled explicitly because
        the batched Cholesky degenerates on a 0×0 observed block.
      * Mix of patterns → batched.

    Returns (T1, T2) as numpy arrays of shape (v,) and (v, v).
    """
    v = n_vars
    N = index.data_padded.shape[0]
    P = index.n_patterns

    # Batched per-pattern regression parameters for patterns with at
    # least one observed variable. Patterns with v_obs_k == 0 are
    # handled separately below (their sigma_oo is 0-rank, which would
    # break the batched Cholesky).
    beta_batched, cond_cov_batched = compute_conditional_parameters_np(
        mu, sigma, index,
    )

    # Build per-pattern (v, v) "fill matrix" F_k and (v,) bias b_k such
    # that for observation i in pattern k,
    #     x_filled_i = data_padded_i + F_k @ data_padded_i + b_k
    # where data_padded_i is zero outside its pattern's observed
    # columns (so F_k @ data_padded_i only picks up the beta @ x_obs
    # term for missing rows).
    #
    # F_k[m, o] = beta_k[m_slot, o_slot] for m ∈ mis_k, o ∈ obs_k; 0 elsewhere.
    # b_k[m]    = mu[m] - beta_k[m_slot, :] @ mu_obs_k        for m ∈ mis_k; 0 elsewhere.
    F = np.zeros((P, v, v), dtype=mu.dtype)
    b = np.zeros((P, v), dtype=mu.dtype)

    for k in range(P):
        v_obs_k = int(index.obs_mask[k].sum())
        v_mis_k = int(index.mis_mask[k].sum())
        if v_mis_k == 0 or v_obs_k == 0:
            continue  # complete or all-missing; handled later
        obs_k = index.obs_idx[k, :v_obs_k]
        mis_k = index.mis_idx[k, :v_mis_k]
        beta_k = beta_batched[k, :v_mis_k, :v_obs_k]
        mu_obs_k = mu[obs_k]
        F[k][np.ix_(mis_k, obs_k)] = beta_k
        b[k][mis_k] = mu[mis_k] - beta_k @ mu_obs_k

    # Scatter per-observation using obs_pattern_id.
    F_per_obs = F[index.obs_pattern_id]       # (N, v, v)
    b_per_obs = b[index.obs_pattern_id]       # (N, v)

    # One batched matmul: fill_i = F_{k(i)} @ data_padded_i.
    # einsum here is ~2× faster than bmm + squeeze on CPU numpy and
    # equivalent on GPU torch.
    fill = np.einsum('nij,nj->ni', F_per_obs, index.data_padded)
    x_filled = index.data_padded + fill + b_per_obs    # (N, v)

    # All-missing patterns don't contribute via data_padded (it's zero
    # there); inject their means directly.
    for k in range(P):
        if index.obs_mask[k].sum() == 0:
            n_k = int(index.n_per_pattern[k])
            # Find those observations in x_filled and overwrite.
            mask = (index.obs_pattern_id == k)
            x_filled[mask] = mu  # broadcast

    T1 = x_filled.sum(axis=0)
    T2 = x_filled.T @ x_filled

    # Conditional-covariance correction — per-pattern scalar accumulator.
    # Build a (P, v, v) padded cond_cov_full with cond_cov in the
    # (mis_k, mis_k) sub-block, then weight by n_k and sum.
    cond_cov_full = np.zeros((P, v, v), dtype=mu.dtype)
    for k in range(P):
        v_mis_k = int(index.mis_mask[k].sum())
        if v_mis_k == 0:
            continue
        v_obs_k = int(index.obs_mask[k].sum())
        if v_obs_k == 0:
            # All-missing pattern: conditional covariance is Sigma itself.
            cond_cov_full[k] = sigma
            continue
        mis_k = index.mis_idx[k, :v_mis_k]
        cond_cov_full[k][np.ix_(mis_k, mis_k)] = cond_cov_batched[k, :v_mis_k, :v_mis_k]

    T2 += np.einsum('k,kij->ij', index.n_per_pattern.astype(mu.dtype),
                    cond_cov_full)

    return T1, T2


def compute_loglik_batched_np(
    mu: np.ndarray,
    sigma: np.ndarray,
    patterns,
    index: _BatchedPatternIndex,
) -> float:
    """Fully batched observed-data log-likelihood.

    Three batched operations replace the P-long Python loop of scalar
    cholesky/solve/slogdet that previously dominated SQUAREM wall-
    clock on many-pattern data:

      1. Batched Cholesky of (P, v_obs_max, v_obs_max) sigma_oo blocks
         (identity-padded in unused slots) → per-pattern log|Sigma_oo|.
      2. Observation-level gather: for each of N observations, pull
         the v_obs_max-vector of centred-data values at its pattern's
         observed column positions (with 0 at padded slots).
      3. One batched solve of (N, v_obs_max, v_obs_max) Sigma_oo
         blocks against the (N, v_obs_max) centred vectors, scattered
         via ``obs_pattern_id``; sum of squared residuals gives the
         quadratic-form contribution to log-likelihood.

    No per-pattern Python loop. Memory cost is (N, v_obs_max, v_obs_max)
    for the per-obs sigma-block gather, which for breast_cancer at
    N=569, v_obs_max=30 is ~4 MB — fine.
    """
    v_obs_max = index.v_obs_max

    # --- Log-determinant term (batched Cholesky over patterns) -----
    row_idx = index.obs_idx[:, :, None]
    col_idx = index.obs_idx[:, None, :]
    sigma_oo = sigma[row_idx, col_idx]
    mask_oo = index.obs_mask[:, :, None] & index.obs_mask[:, None, :]
    eye_oo = np.broadcast_to(np.eye(v_obs_max, dtype=sigma.dtype), sigma_oo.shape)
    sigma_oo = np.where(mask_oo, sigma_oo, eye_oo)

    # Batched Cholesky for log-det and the quadratic-form solve below.
    # Per-pattern sigma_oo sub-blocks can be numerically indefinite from
    # FP64 roundoff even when the global sigma is PD (confirmed on
    # credit_card_default via Project Lacuna). Apply a tiny diagonal
    # ridge before Cholesky rather than raising — the ridge (1e-12 on a
    # matrix normalised to trace ~v) is below any statistical precision.
    try:
        L_oo = np.linalg.cholesky(sigma_oo)
    except np.linalg.LinAlgError:
        import warnings
        ridge = 1e-10
        warnings.warn(
            f"e_step_full_batched_np: per-pattern sigma_oo indefinite; "
            f"retrying Cholesky with diagonal ridge {ridge:.0e}.",
            UserWarning, stacklevel=3,
        )
        eye_full = np.broadcast_to(
            np.eye(v_obs_max, dtype=sigma.dtype), sigma_oo.shape,
        )
        L_oo = np.linalg.cholesky(sigma_oo + ridge * eye_full)
    log_diag = np.log(np.diagonal(L_oo, axis1=-2, axis2=-1))
    logdet_per_pattern = 2.0 * np.sum(log_diag * index.obs_mask, axis=-1)

    # --- Quadratic-form term (fully batched over observations) -----
    # Per-observation gather of the pattern's observed indices. Padded
    # slots reuse obs_idx[0] harmlessly because the mask zeroes them.
    obs_pattern_id = index.obs_pattern_id
    per_obs_obs_idx = index.obs_idx[obs_pattern_id]            # (N, v_obs_max)
    per_obs_obs_mask = index.obs_mask[obs_pattern_id]          # (N, v_obs_max)

    # Gather each observation's data at its pattern's observed columns.
    # data_padded holds the real observed values at the right positions;
    # we pull them out in the pattern's canonical order.
    N_arange = np.arange(index.data_padded.shape[0])[:, None]   # (N, 1)
    y_gathered = index.data_padded[N_arange, per_obs_obs_idx]  # (N, v_obs_max)
    mu_gathered = mu[per_obs_obs_idx] * per_obs_obs_mask       # (N, v_obs_max)
    centered = (y_gathered - mu_gathered) * per_obs_obs_mask   # (N, v_obs_max)

    # Gather each observation's Sigma_oo block and Cholesky factor.
    L_per_obs = L_oo[obs_pattern_id]                            # (N, v_obs_max, v_obs_max)

    # Solve L_i z_i = centered_i for every observation in one call.
    z = np.linalg.solve(L_per_obs, centered[:, :, None]).squeeze(-1)  # (N, v_obs_max)

    quad_total = float(np.sum(z * z))

    # Weighted sum of log-determinants via n_per_pattern.
    logdet_sum = float(np.sum(
        index.n_per_pattern.astype(mu.dtype) * logdet_per_pattern
    ))

    return float(-0.5 * logdet_sum - 0.5 * quad_total)
