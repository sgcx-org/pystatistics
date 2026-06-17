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

    Within the window the candidate values are a contiguous slice of the sorted
    array, so the distances to the query are unimodal (V-shaped) and the k
    nearest are a *contiguous block*. We pick the block whose worse endpoint is
    smallest — elementwise + a tiny argmin — instead of ``topk``, which is slow
    on both MPS and CUDA. Shared by both devices.
    """
    import torch

    m, n_obs = yhat_obs.shape
    n_mis = yhat_mis.shape[1]
    device = yhat_obs.device
    k = min(donors, n_obs)
    w = min(2 * k, n_obs)

    # Sort donor predictions per chain; ``order`` maps back to original rows.
    sorted_obs, order = torch.sort(yhat_obs, dim=1)               # (m, n_obs)
    pos = _insertion_rank(sorted_obs, yhat_mis)                  # (m, n_mis)
    start = torch.clamp(pos - k, min=0, max=n_obs - w)           # (m, n_mis)
    win = start.unsqueeze(-1) + torch.arange(w, device=device)   # (m, n_mis, w)

    # Gather only the window's (sorted) values — no dense materialisation.
    b = torch.arange(m, device=device)[:, None, None]
    cand_vals = sorted_obs[b, win]                               # (m, n_mis, w)
    dist = (cand_vals - yhat_mis.unsqueeze(-1)).abs()           # V-shaped in w

    # Contiguous k-block: minimise max(dist[s], dist[s+k-1]) over the starts s.
    n_starts = w - k + 1
    cost = torch.maximum(dist[..., :n_starts], dist[..., k - 1:k - 1 + n_starts])
    s = torch.argmin(cost, dim=-1)                              # (m, n_mis)

    # One random donor per (chain, missing row) among its k candidates. Because
    # the block is contiguous (cols s..s+k-1) the chosen sorted position is just
    # start + s + pick — one fused index instead of a chain of gathers.
    pick = torch.randint(0, k, (m, n_mis), generator=gen, device=device)
    chosen_sorted = start + s + pick                            # (m, n_mis)
    return torch.gather(order, 1, chosen_sorted)                 # original obs idx


def _insertion_rank(sorted_obs, yhat_mis):
    """Insertion rank of each missing prediction among the sorted observed
    predictions: ``pos[i] = #{observed < yhat_mis[i]}``.

    Device bridge — the one op that genuinely splits by device. ``searchsorted``
    is fast on CUDA but pathologically slow on MPS at scale, so on MPS we
    reconstruct the same ranks from one combined sort (``sort`` is fast on MPS):
    concatenate observed+missing, sort once, count observed elements preceding
    each slot with a ``cumsum``, and scatter the count back to each missing
    element. The two agree to within a ±1 tie convention, which is harmless
    here: ``pos`` only seeds the width-``2k`` window that the contiguous-block
    search then refines exactly.
    """
    import torch

    if sorted_obs.device.type != "mps":
        return torch.searchsorted(sorted_obs, yhat_mis)

    m, n_obs = sorted_obs.shape
    n_mis = yhat_mis.shape[1]
    device = sorted_obs.device
    allv = torch.cat([sorted_obs, yhat_mis], dim=1)             # (m, n_obs+n_mis)
    is_mis = torch.cat(
        [
            torch.zeros((m, n_obs), device=device, dtype=torch.long),
            torch.ones((m, n_mis), device=device, dtype=torch.long),
        ],
        dim=1,
    )
    sort_order = torch.argsort(allv, dim=1, stable=True)
    sorted_is_mis = torch.gather(is_mis, 1, sort_order)
    obs_before = torch.cumsum(1 - sorted_is_mis, dim=1) - (1 - sorted_is_mis)
    orig_mis = (sort_order - n_obs).clamp_min(0)
    # Route non-missing slots to a throwaway column so only missing slots write.
    tgt = torch.where(
        sorted_is_mis.bool(), orig_mis, torch.full_like(orig_mis, n_mis)
    )
    pos_ext = torch.zeros((m, n_mis + 1), device=device, dtype=obs_before.dtype)
    pos_ext.scatter_(1, tgt, obs_before)
    return pos_ext[:, :n_mis]


# Method-name -> batched implementation. The backend translates the design's
# per-column method name through this table; unknown names are refused there.
GPU_METHODS = {
    "norm": gpu_norm_impute,
    "pmm": gpu_pmm_impute,
}
