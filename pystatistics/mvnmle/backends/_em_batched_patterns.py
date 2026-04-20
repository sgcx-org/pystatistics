"""Pattern-index infrastructure shared by numpy and torch batched backends.

Extracted from ``_em_batched.py`` on 2026-04-20 to keep each file under
the 500-SLOC hard limit (Coding Bible rule 4). See module ``_em_batched``
for a compatibility shim that re-exports these symbols.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pystatistics.mvnmle._objectives.base import PatternData  # noqa: F401 (public re-export via shim)


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
