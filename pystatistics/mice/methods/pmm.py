"""
``pmm`` — predictive mean matching (R mice's default for numeric columns).

Semi-parametric imputation: fit a Bayesian linear regression, compute fitted
values for both observed and missing rows, then impute each missing value by
copying a *real observed value* whose fitted value is among the ``k`` closest to
the missing row's fitted value (one donor chosen at random).

Because every imputed value is an actually-observed value, PMM respects the
empirical distribution (bounds, skew, discreteness) of the column — which is why
the issue specifically asks for it as the R-faithful numeric default.

Match type 1 (R's default): observed fitted values use the point estimate
``beta_hat``; missing fitted values use the posterior draw ``beta*``. This pairs
a stable target metric with between-imputation variability, the combination that
gives PMM its correct multiple-imputation behaviour.
"""

from __future__ import annotations

import numpy as np

from pystatistics.mice.methods._linreg import bayes_linreg_draw
from pystatistics.mice.methods.registry import register

# R mice default donor count.
_DEFAULT_DONORS = 5


class PMMMethod:
    """Predictive mean matching (conforms to ImputationMethod)."""

    name = "pmm"
    target_kind = "numeric"

    def __init__(self, donors: int = _DEFAULT_DONORS):
        if donors < 1:
            from pystatistics.core.exceptions import ValidationError
            raise ValidationError(f"donors must be >= 1, got {donors}")
        self.donors = donors

    def impute(
        self,
        y_obs: np.ndarray,
        X_obs: np.ndarray,
        X_mis: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        draw = bayes_linreg_draw(y_obs, X_obs, rng)
        yhat_obs = draw.predict_hat(X_obs)   # beta_hat (matchtype=1)
        yhat_mis = draw.predict_draw(X_mis)  # beta*    (matchtype=1)

        donor_idx = _match_donors(yhat_mis, yhat_obs, self.donors, rng)
        return y_obs[donor_idx]


def _match_donors(
    yhat_mis: np.ndarray,
    yhat_obs: np.ndarray,
    donors: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """For each missing fitted value, pick one of its ``k`` nearest observed
    fitted values at random; return the chosen observed-row indices.

    Scales like R's ``mice`` matcher: sort the observed fitted values once, then
    for each missing value search a small window around its insertion point
    instead of forming the full ``(n_mis, n_obs)`` distance matrix. Cost is
    ``O(n_obs log n_obs + n_mis·k)`` time and ``O(n_mis·k)`` memory — the dense
    version was ``O(n_mis·n_obs)`` in both, which blows up on large data.

    The window is provably exact. In a sorted array the ``k`` nearest neighbours
    of a query lie in the contiguous block ``[pos-k, pos+k-1]`` around the
    insertion point ``pos``; a width-``2k`` window covers that block (shifted to
    stay in range near the ends), so the windowed k-NN equals the global k-NN.
    """
    n_obs = yhat_obs.shape[0]
    n_mis = yhat_mis.shape[0]
    k = min(donors, n_obs)

    # Sort donor predictions once; keep the map back to original obs rows.
    order = np.argsort(yhat_obs, kind="stable")
    sorted_obs = yhat_obs[order]

    # Insertion point of each missing prediction into the sorted donors.
    pos = np.searchsorted(sorted_obs, yhat_mis)

    # Window of distinct candidate donors guaranteed to contain the k nearest.
    w = min(2 * k, n_obs)
    start = np.clip(pos - k, 0, n_obs - w)                  # (n_mis,)
    win = start[:, None] + np.arange(w)[None, :]            # (n_mis, w) sorted-idx
    dist = np.abs(sorted_obs[win] - yhat_mis[:, None])      # (n_mis, w)

    # k smallest distances within the window (unordered within the k).
    if k < w:
        knn = np.argpartition(dist, k - 1, axis=1)[:, :k]   # (n_mis, k) window cols
    else:
        knn = np.broadcast_to(np.arange(w), (n_mis, w))

    rows = np.arange(n_mis)
    pick = rng.integers(0, k, size=n_mis)                   # one donor per row
    chosen_win_col = knn[rows, pick]                        # (n_mis,)
    chosen_sorted = win[rows, chosen_win_col]               # idx into sorted_obs
    return order[chosen_sorted]                             # back to original rows


register(PMMMethod())
