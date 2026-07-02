"""Approximate significance tests for GAM smooth terms.

One job: the per-smooth test statistic and p-value shown in the summary
table. Follows the shape of mgcv's Wood (2013) test: the statistic is
``beta_j' pinv_r(V_jj) beta_j`` with the pseudo-inverse truncated at rank
``r ~ Ref.df``, referred to a chi-squared (known scale) or F (estimated
scale) distribution with fractional Ref.df degrees of freedom.

This is a SIMPLIFIED form of mgcv's ``testStat`` (which re-parameterises
via an extra pivoted QR and interpolates the rank truncation smoothly);
statistics agree closely but not bit-for-bit — the difference is
documented, not hidden. p-values remain approximate in both engines
(neither accounts fully for smoothing-parameter selection).

Reference: Wood, S.N. (2013). On p-values for smooth components of an
extended generalized additive model. Biometrika 100(1), 221-228.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


def smooth_term_test(
    beta_block: NDArray[np.floating[Any]],
    V_block: NDArray[np.floating[Any]],
    edf: float,
    ref_df: float,
    scale_known: bool,
    resid_df: float,
) -> tuple[float, float]:
    """Test H0: f_j = 0 for one smooth term.

    Args:
        beta_block: The term's (constrained) coefficients ``(k_c,)``.
        V_block: Matching block of the Bayesian posterior covariance.
        edf: The term's effective degrees of freedom.
        ref_df: The term's reference df (``tr(2H - HH)`` block).
        scale_known: True for fixed-dispersion families (chi-squared
            reference); False -> F reference with ``resid_df``.
        resid_df: Residual degrees of freedom ``n - total_edf``.

    Returns:
        ``(statistic, p_value)``. For a term penalised to (near) zero
        (edf ~ 0, all-zero covariance block) the statistic is 0 and the
        p-value 1 — a deliberately conservative degenerate answer rather
        than a division by zero.
    """
    r_df = max(float(ref_df), float(edf), 1e-8)
    # Rank-truncated pseudo-inverse of the covariance block at r ~ Ref.df.
    ev, U = np.linalg.eigh(0.5 * (V_block + V_block.T))
    rank = int(min(np.ceil(r_df), ev.shape[0]))
    order = np.argsort(ev)[::-1][:rank]
    ev_r = ev[order]
    keep = ev_r > (ev_r.max() * 1e-12 if ev_r.size and ev_r.max() > 0 else 0)
    if not np.any(keep):
        return 0.0, 1.0
    U_r = U[:, order][:, keep]
    proj = U_r.T @ beta_block
    stat = float(proj @ (proj / ev_r[keep]))

    if scale_known:
        p = float(sp_stats.chi2.sf(stat, df=r_df))
    else:
        f_stat = stat / r_df
        p = float(sp_stats.f.sf(f_stat, r_df, max(resid_df, 1.0)))
        stat = f_stat
    return stat, p
