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

    Vectorized across all missing rows — this is the donor-search kernel a
    Stage-2 GPU backend would replace with a batched nearest-neighbour search.
    """
    n_obs = yhat_obs.shape[0]
    k = min(donors, n_obs)

    # |yhat_mis_i - yhat_obs_j| for every (i, j): (n_mis, n_obs).
    dist = np.abs(yhat_mis[:, None] - yhat_obs[None, :])

    # Indices of the k smallest distances per row (unordered within the k).
    if k < n_obs:
        cand = np.argpartition(dist, k - 1, axis=1)[:, :k]
    else:
        cand = np.broadcast_to(np.arange(n_obs), (yhat_mis.shape[0], n_obs))

    # One random donor per row among its k candidates.
    pick = rng.integers(0, k, size=yhat_mis.shape[0])
    return cand[np.arange(yhat_mis.shape[0]), pick]


register(PMMMethod())
