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

    donor_idx = _match_donors_windowed(yhat_mis, yhat_obs, donors, gen)
    # Copy the observed donor values (gather along the observed-row axis).
    return torch.gather(y_obs, 1, donor_idx)                   # (m, n_mis)


def _match_donors_windowed(yhat_mis, yhat_obs, donors, gen):
    """Batched sorted-window k-NN donor search — the GPU port of the CPU matcher.

    Sorts each chain's observed predictions once, then for every missing value
    searches a width-``2k`` window around its insertion point instead of forming
    the dense ``(m, n_mis, n_obs)`` distance tensor. Memory drops from
    ``O(m·n_mis·n_obs)`` to ``O(m·n_mis·k)``; the window is provably exact (the
    global k nearest lie in ``[pos-k, pos+k-1]`` of the sorted array). Returns
    the chosen *original* observed-row index per (chain, missing row).
    """
    import torch

    m, n_obs = yhat_obs.shape
    n_mis = yhat_mis.shape[1]
    device = yhat_obs.device
    k = min(donors, n_obs)
    w = min(2 * k, n_obs)

    # Sort donor predictions per chain; ``order`` maps back to original rows.
    sorted_obs, order = torch.sort(yhat_obs, dim=1)               # (m, n_obs)
    pos = torch.searchsorted(sorted_obs, yhat_mis)               # (m, n_mis)
    start = torch.clamp(pos - k, min=0, max=n_obs - w)           # (m, n_mis)
    win = start.unsqueeze(-1) + torch.arange(w, device=device)   # (m, n_mis, w)

    # Gather only the window's values (no dense materialisation).
    b = torch.arange(m, device=device)[:, None, None]
    cand_vals = sorted_obs[b, win]                               # (m, n_mis, w)
    dist = (cand_vals - yhat_mis.unsqueeze(-1)).abs()           # (m, n_mis, w)
    knn = torch.topk(dist, k, dim=2, largest=False).indices      # (m, n_mis, k)

    # One random donor per (chain, missing row) among its k candidates.
    pick = torch.randint(0, k, (m, n_mis), generator=gen, device=device)
    chosen_col = torch.gather(knn, 2, pick.unsqueeze(-1)).squeeze(-1)        # (m, n_mis)
    chosen_sorted = torch.gather(win, 2, chosen_col.unsqueeze(-1)).squeeze(-1)
    return torch.gather(order, 1, chosen_sorted)                 # original obs idx


# Method-name -> batched implementation. The backend translates the design's
# per-column method name through this table; unknown names are refused there.
GPU_METHODS = {
    "norm": gpu_norm_impute,
    "pmm": gpu_pmm_impute,
}
