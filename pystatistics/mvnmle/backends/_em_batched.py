"""Batched building blocks for the EM E-step.

The per-pattern Python loop in the CPU/GPU EM backend issues O(P)
Cholesky + triangular-solve calls per E-step iteration. On GPU the
fixed kernel-launch cost per call (~50 µs) dominates the actual
compute on per-pattern submatrices (often <30x30 FP64), so the GPU
path sees no speedup over CPU.

This module stacks all per-pattern ``sigma_oo`` submatrices into a
single ``(P, v_max, v_max)`` tensor (padded with identity in the
unused slots so Cholesky is well-defined), then performs one
batched Cholesky and one batched triangular solve. The n_k-varying
data application still runs per-pattern in the caller — that part
is cheap and hard to stack without wasting memory on n_max padding.

Everything here is FP64 and bit-faithful to the scalar reference
implementation when the backing device is CPU. GPU FP32 results
match at the ``GPU_FP32`` tolerance tier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from pystatistics.mvnmle._objectives.base import PatternData


@dataclass(frozen=True)
class _BatchedPatternIndex:
    """Precomputed per-pattern + per-observation indices stacked for
    batched gathers.

    Built once per EM solve (patterns don't change across iterations).

    Fields use numpy; backend-specific conversion to torch.Tensor
    happens inside the batched helpers.

    In addition to per-pattern slots, we keep observation-level
    structures that let the E-step apply per-pattern regression
    matrices to all N observations in a single batched matmul —
    eliminating the per-pattern Python loop that previously ate the
    GPU wins on data where P approaches N (e.g. breast_cancer at
    high missingness has one pattern per observation).
    """
    obs_idx: np.ndarray        # (P, v_obs_max) int64; padded with 0
    obs_mask: np.ndarray       # (P, v_obs_max) bool
    mis_idx: np.ndarray        # (P, v_mis_max) int64; padded with 0
    mis_mask: np.ndarray       # (P, v_mis_max) bool
    v_obs_max: int
    v_mis_max: int
    n_patterns: int
    # Observation-level views, aligned with MLEObjectiveBase's pattern
    # ordering. data_padded has NaN → 0 at missing positions so that
    # (fill_matrix_k @ data_padded) zeros them out automatically.
    data_padded: np.ndarray    # (N, v) float64
    obs_pattern_id: np.ndarray # (N,) int64 — which pattern each obs belongs to
    obs_full_mask: np.ndarray  # (N, v) bool — True at observed positions
    n_per_pattern: np.ndarray  # (P,) int64 — n_k per pattern


def _pattern_n(pat):
    """Read the per-pattern observation count regardless of whether
    the caller passed an EM-side ``PatternData`` (``.n_obs``) or an
    mcar-side ``PatternInfo`` (``.n_cases``). Both types appear in
    this module's users."""
    return getattr(pat, 'n_obs', None) or getattr(pat, 'n_cases')


def build_pattern_index(patterns, n_vars: int) -> _BatchedPatternIndex:
    """Stack per-pattern indices and assemble the observation-level
    padded-data view used by the fully batched E-step and MCAR
    chi-square assembly.

    Accepts either ``PatternData`` (EM) or ``PatternInfo`` (MCAR) —
    reads the count via ``_pattern_n`` so both callsites work.
    """
    P = len(patterns)
    v_obs_max = max(len(pat.observed_indices) for pat in patterns)
    v_mis_max = max(
        (len(pat.missing_indices) for pat in patterns), default=0
    )

    obs_idx = np.zeros((P, max(v_obs_max, 1)), dtype=np.int64)
    obs_mask = np.zeros((P, max(v_obs_max, 1)), dtype=bool)
    mis_idx = np.zeros((P, max(v_mis_max, 1)), dtype=np.int64)
    mis_mask = np.zeros((P, max(v_mis_max, 1)), dtype=bool)
    n_per_pattern = np.zeros(P, dtype=np.int64)

    total_n = sum(_pattern_n(pat) for pat in patterns)
    data_padded = np.zeros((total_n, n_vars), dtype=np.float64)
    obs_full_mask = np.zeros((total_n, n_vars), dtype=bool)
    obs_pattern_id = np.zeros(total_n, dtype=np.int64)

    row = 0
    for k, pat in enumerate(patterns):
        v_obs_k = len(pat.observed_indices)
        v_mis_k = len(pat.missing_indices)
        n_k = _pattern_n(pat)
        n_per_pattern[k] = n_k
        obs_idx[k, :v_obs_k] = pat.observed_indices
        obs_mask[k, :v_obs_k] = True
        if v_mis_k > 0:
            mis_idx[k, :v_mis_k] = pat.missing_indices
            mis_mask[k, :v_mis_k] = True

        if n_k > 0 and v_obs_k > 0:
            data_padded[row:row + n_k, pat.observed_indices] = pat.data
            obs_full_mask[row:row + n_k, pat.observed_indices] = True
        obs_pattern_id[row:row + n_k] = k
        row += n_k

    return _BatchedPatternIndex(
        obs_idx=obs_idx,
        obs_mask=obs_mask,
        mis_idx=mis_idx,
        mis_mask=mis_mask,
        v_obs_max=v_obs_max,
        v_mis_max=v_mis_max,
        n_patterns=P,
        data_padded=data_padded,
        obs_pattern_id=obs_pattern_id,
        obs_full_mask=obs_full_mask,
        n_per_pattern=n_per_pattern,
    )


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


def chi_square_mcar_batched_np(
    mu: np.ndarray,
    sigma: np.ndarray,
    patterns,
    index: _BatchedPatternIndex,
    condition_threshold: float = 1e12,
    regularize: bool = True,
) -> tuple:
    """Batched MCAR chi-square assembly.

    Computes Σ_k n_k (ȳ_k - μ_o_k)^T Σ_{o_k}^{-1} (ȳ_k - μ_o_k)
    in a single batched Cholesky + batched solve + batched
    quadratic-form pass, with per-pattern ill-conditioning handled
    by falling back to Moore-Penrose pseudo-inverse where needed.

    Replaces a P-long Python loop that issued a separate
    ``np.linalg.cond`` (internally SVD) + ``np.linalg.inv`` per
    pattern — typically 30-50 % of MoM's wall-clock on
    many-pattern data.

    Returns
    -------
    test_statistic : float
    n_patterns_used : int
    n_regularized : int
        Number of patterns that fell through to pseudo-inverse.
    """
    v_obs_max = index.v_obs_max

    row_idx = index.obs_idx[:, :, None]
    col_idx = index.obs_idx[:, None, :]
    sigma_oo = sigma[row_idx, col_idx]
    mask_oo = index.obs_mask[:, :, None] & index.obs_mask[:, None, :]
    eye_oo = np.broadcast_to(np.eye(v_obs_max, dtype=sigma.dtype), sigma_oo.shape)
    sigma_oo = np.where(mask_oo, sigma_oo, eye_oo)

    # Batched condition-number check via batched SVD.
    cond = np.linalg.cond(sigma_oo)  # (P,)
    ill = cond > condition_threshold

    test_statistic = 0.0
    n_patterns_used = 0
    n_regularized = 0

    # Pre-compute pattern means (ȳ_k) for each pattern — one matmul per
    # pattern's data, but n_k is small so this loop is cheap relative
    # to the old per-pattern-SVD cost. Keeping scalar here avoids a
    # pad-by-n_max allocation.
    y_bars = np.zeros((index.n_patterns, v_obs_max), dtype=mu.dtype)
    n_per_pattern = index.n_per_pattern
    for k, pattern in enumerate(patterns):
        if pattern.n_observed == 0:
            continue
        v_obs_k = pattern.n_observed
        y_bars[k, :v_obs_k] = np.mean(pattern.data, axis=0)

    # Batched mu gather (P, v_obs_max) zeroed at padded slots.
    mu_obs_padded = mu[index.obs_idx] * index.obs_mask
    diff_padded = (y_bars - mu_obs_padded) * index.obs_mask  # (P, v_obs_max)

    # For well-conditioned patterns: batched solve of sigma_oo z = diff.
    # diff @ z gives the quadratic form.
    # For ill-conditioned patterns: fall back to pinv per-pattern.
    # Strategy: compute both paths and select by condition.
    if ill.any() and regularize:
        import warnings
        worst = float(cond[ill].max())
        warnings.warn(
            f"Covariance matrix for at least one missingness pattern is "
            f"ill-conditioned (worst cond={worst:.2e} > threshold="
            f"{condition_threshold:.0e}). Using Moore-Penrose "
            f"pseudo-inverse on {int(ill.sum())} of {index.n_patterns} "
            f"patterns; chi-square contribution for those patterns may "
            f"have reduced precision. Pass regularize=False to raise "
            f"instead.",
            UserWarning, stacklevel=4,
        )
    elif ill.any() and not regularize:
        from pystatistics.core.exceptions import NumericalError
        raise NumericalError(
            f"Covariance matrix for {int(ill.sum())} pattern(s) is "
            f"ill-conditioned (worst cond={float(cond[ill].max()):.2e} > "
            f"threshold={condition_threshold:.0e}). Pass "
            f"regularize=True to fall back to Moore-Penrose pseudo-"
            f"inverse, or increase condition_threshold."
        )

    diff_3d = diff_padded[:, :, None]  # (P, v_obs_max, 1)
    z = np.zeros_like(diff_padded)

    # Split patterns by conditioning and batch each group separately.
    # Well-conditioned patterns go through the cheap batched solve;
    # ill-conditioned patterns through batched pinv. Both branches
    # run as one BLAS call each — no Python loop over patterns.
    ok = ~ill
    if ok.any():
        z_ok = np.linalg.solve(sigma_oo[ok], diff_3d[ok]).squeeze(-1)
        z[ok] = z_ok
    if ill.any():
        sigma_pinv = np.linalg.pinv(sigma_oo[ill])
        z_ill = np.matmul(sigma_pinv, diff_3d[ill]).squeeze(-1)
        z[ill] = z_ill
        n_regularized = int(ill.sum())

    # Quadratic form per pattern: contribution_k = n_k * diff_k · z_k
    contribs = (diff_padded * z).sum(axis=-1)  # (P,) — padded slots contribute 0
    contribs_nk = contribs * n_per_pattern.astype(mu.dtype)

    # Count patterns used: those with at least one observed variable.
    used_mask = index.obs_mask.any(axis=-1)
    n_patterns_used = int(used_mask.sum())

    test_statistic = float(contribs_nk[used_mask].sum())
    return test_statistic, n_patterns_used, n_regularized


def chi_square_mcar_batched_torch(
    mu, sigma, index, n_per_pattern_t, obs_idx_t, obs_mask_t,
    y_bars_t, eye_oo, torch_mod, device, dtype,
    condition_threshold: float = 1e12,
    regularize: bool = True,
):
    """Torch/GPU version of ``chi_square_mcar_batched_np``.

    Same algorithmic structure; the ill-conditioned fallback uses
    batched SVD-based pseudo-inverse to avoid Python-level branching
    between per-pattern solve and pinv on device.
    """
    torch = torch_mod

    row_idx = obs_idx_t.unsqueeze(-1)
    col_idx = obs_idx_t.unsqueeze(-2)
    sigma_oo = sigma[row_idx, col_idx]
    mask_oo = obs_mask_t.unsqueeze(-1) & obs_mask_t.unsqueeze(-2)
    sigma_oo = torch.where(mask_oo, sigma_oo, eye_oo)

    # Batched cond via batched SVD singular values.
    svals = torch.linalg.svdvals(sigma_oo)  # (P, v_obs_max)
    # Ignore the zero singular values from padded rows (mask their
    # contribution by the obs_mask); avoid div-by-zero for all-padded
    # rows which shouldn't occur in practice.
    smax = svals[:, 0]
    # Smallest valid singular value per pattern: take the smallest
    # among positions that are genuine observed dims.
    n_obs_per_pattern = obs_mask_t.sum(dim=-1).clamp(min=1)
    # Gather the (n_obs_k-1)-th singular value per pattern as smin;
    # for padded slots svdvals are zeros; so smin = svals[torch.arange(P), n_obs_per_pattern - 1].
    P = svals.shape[0]
    smin = svals[torch.arange(P, device=device), n_obs_per_pattern - 1]
    cond = smax / smin.clamp(min=1e-300)
    ill = cond > condition_threshold

    if ill.any() and regularize:
        import warnings
        worst = float(cond[ill].max().item())
        warnings.warn(
            f"Covariance matrix for at least one missingness pattern is "
            f"ill-conditioned (worst cond={worst:.2e} > threshold="
            f"{condition_threshold:.0e}). Using Moore-Penrose "
            f"pseudo-inverse on {int(ill.sum().item())} patterns; "
            f"chi-square contribution may have reduced precision. Pass "
            f"regularize=False to raise instead.",
            UserWarning, stacklevel=4,
        )
    elif ill.any() and not regularize:
        from pystatistics.core.exceptions import NumericalError
        raise NumericalError(
            f"Covariance matrix for {int(ill.sum().item())} pattern(s) "
            f"is ill-conditioned. Pass regularize=True to fall back to "
            f"Moore-Penrose pseudo-inverse, or raise condition_threshold."
        )

    # Build a per-pattern inverse: cholesky_solve for well-conditioned,
    # pinv for ill-conditioned. All batched.
    mu_obs_padded = mu[obs_idx_t] * obs_mask_t.to(dtype)
    diff = (y_bars_t - mu_obs_padded) * obs_mask_t.to(dtype)  # (P, v_obs_max)

    # Always use pinv when regularize is True and there are any ill
    # patterns — it's bit-safe across patterns. For well-conditioned
    # sigma_oo, pinv == inv; we pay a modest perf overhead to avoid a
    # Python-level branch on device.
    sigma_inv = torch.linalg.pinv(sigma_oo) if ill.any() else None
    if sigma_inv is None:
        # Fast path: condition-number check says all are well-conditioned,
        # so attempt cholesky_solve. Cholesky requires positive-definiteness,
        # which is a STRICTER property than good conditioning — a matrix
        # can pass the cond-number threshold but still have tiny negative
        # eigenvalues due to FP32 roundoff (especially on GPU). When that
        # happens, fall back to pinv for those patterns. regularize=False
        # is respected: we only attempt the fallback when the user has
        # already opted into regularization.
        try:
            L = torch.linalg.cholesky(sigma_oo)
            z = torch.cholesky_solve(diff.unsqueeze(-1), L).squeeze(-1)
            n_cholesky_fallback = 0
        except torch._C._LinAlgError:
            if not regularize:
                raise
            import warnings
            warnings.warn(
                "Cholesky factorisation failed on the batched fast path "
                "despite condition-number check passing — likely FP32 "
                "roundoff producing a numerically-indefinite covariance. "
                "Falling back to Moore-Penrose pseudo-inverse for all "
                "patterns in this batch. Pass regularize=False to raise "
                "instead.",
                UserWarning, stacklevel=4,
            )
            sigma_inv = torch.linalg.pinv(sigma_oo)
            z = torch.matmul(sigma_inv, diff.unsqueeze(-1)).squeeze(-1)
            n_cholesky_fallback = sigma_oo.shape[0]
    else:
        z = torch.matmul(sigma_inv, diff.unsqueeze(-1)).squeeze(-1)
        n_cholesky_fallback = 0

    contribs = (diff * z).sum(dim=-1)  # (P,)
    contribs_nk = contribs * n_per_pattern_t

    used_mask = obs_mask_t.any(dim=-1)
    n_patterns_used = int(used_mask.sum().item())

    test_statistic = float(contribs_nk[used_mask].sum().item())
    n_regularized = int(ill.sum().item()) + n_cholesky_fallback
    return test_statistic, n_patterns_used, n_regularized


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
