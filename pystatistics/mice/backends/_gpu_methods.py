"""
Batched GPU imputation methods: ``norm`` and ``pmm``.

These mirror ``methods/norm.py`` and ``methods/pmm.py`` but operate on tensors
batched over the ``m`` chains. Each takes the observed/missing predictor blocks
for one target column across all chains and returns the imputed values for every
chain at once, so a whole sweep step is a handful of batched kernels.

The GPU backend dispatches to these by method name; a name with no GPU
implementation is refused by the backend (fail loud, Rule 1) rather than
silently downgraded.
"""

from __future__ import annotations

from pystatistics.mice.backends._gpu_linreg import (
    add_intercept,
    batched_bayes_linreg_draw,
)


def gpu_norm_impute(y_obs, X_obs, X_mis, gen, *, donors=None):
    """Batched Bayesian linear-regression imputation (R ``norm``).

    Parameters mirror the CPU method but are batched: ``y_obs`` (m, n_obs),
    ``X_obs`` (m, n_obs, q), ``X_mis`` (m, n_mis, q). Returns (m, n_mis).
    ``donors`` is accepted and ignored so the backend can call every method
    with one uniform signature.
    """
    import torch

    draw = batched_bayes_linreg_draw(y_obs, X_obs, gen)
    Xa_mis = add_intercept(X_mis)
    yhat_mis = (Xa_mis @ draw.beta_draw.unsqueeze(-1)).squeeze(-1)  # (m, n_mis)
    noise = torch.randn(
        yhat_mis.shape, generator=gen, dtype=yhat_mis.dtype, device=yhat_mis.device
    )
    return yhat_mis + draw.sigma_draw[:, None] * noise


def gpu_pmm_impute(y_obs, X_obs, X_mis, gen, *, donors=5):
    """Batched predictive mean matching (R ``pmm``, matchtype=1).

    Observed fitted values use ``beta_hat``; missing fitted values use the
    posterior draw ``beta*``. For each missing row in each chain, pick one of
    the ``k`` nearest observed fitted values at random and copy that observed
    value. Returns (m, n_mis).
    """
    import torch

    draw = batched_bayes_linreg_draw(y_obs, X_obs, gen)
    Xa_obs = add_intercept(X_obs)
    Xa_mis = add_intercept(X_mis)
    yhat_obs = (Xa_obs @ draw.beta_hat.unsqueeze(-1)).squeeze(-1)   # (m, n_obs)
    yhat_mis = (Xa_mis @ draw.beta_draw.unsqueeze(-1)).squeeze(-1)  # (m, n_mis)

    m, n_obs = yhat_obs.shape
    n_mis = yhat_mis.shape[1]
    k = min(donors, n_obs)

    # |yhat_mis - yhat_obs| for every (chain, missing row, observed row).
    dist = (yhat_mis[:, :, None] - yhat_obs[:, None, :]).abs()  # (m, n_mis, n_obs)
    # k nearest observed fitted values (indices into n_obs).
    _, cand = torch.topk(dist, k, dim=2, largest=False)         # (m, n_mis, k)

    # One random donor per (chain, missing row) among its k candidates.
    pick = torch.randint(
        0, k, (m, n_mis), generator=gen, device=yhat_obs.device
    )                                                          # (m, n_mis)
    donor_idx = torch.gather(cand, 2, pick.unsqueeze(-1)).squeeze(-1)  # (m, n_mis)

    # Copy the observed donor values (gather along the observed-row axis).
    return torch.gather(y_obs, 1, donor_idx)                   # (m, n_mis)


# Method-name -> batched implementation. The backend translates the design's
# per-column method name through this table; unknown names are refused there.
GPU_METHODS = {
    "norm": gpu_norm_impute,
    "pmm": gpu_pmm_impute,
}
