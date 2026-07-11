"""
Robust (sandwich) and cluster-robust variance for Cox PH.

The Lin-Wei sandwich estimator replaces the model-based variance
``V = I^{-1}`` with ``D' D``, where ``D`` is the matrix of per-observation
``dfbeta`` residuals (``D = L V``, ``L`` the score residuals). Clustering sums
``D`` within each cluster before the cross-product, so correlated rows of one
subject/cluster contribute once — matching ``coxph(..., robust=TRUE)`` and
``coxph(..., cluster=id)`` / ``residuals(fit, type='dfbeta')``.

Score residual for row ``i`` (Therneau-Grambsch), summed over the strata it
belongs to:

    L_i = [x_i - xbar(t_i)]·dN_i  -  w_i · Σ_{t_k in (entry_i, time_i]}
                                        (x_i - xbar(t_k)) dLambda0(t_k)

with ``w_i = exp(x_i·beta)``. The at-risk sum telescopes through cumulative
baseline-hazard sums, so the whole matrix costs O(n·p). Efron ties give each of
the ``d_k`` tied deaths a partial share of its own event-time increment; that
correction is applied only within genuine multi-death tie groups.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatistics.survival._cox import _reverse_cumsum, _score_and_information
from pystatistics.survival._cox_strata import _build_strata


def _pooled_information(
    strata_list, beta: NDArray, ties: str, p: int
) -> NDArray:
    """Observed information summed over strata at ``beta`` (the fit's I).

    Strata here carry a trailing original-index column in ``X``; the first ``p``
    columns are the covariates.
    """
    info = np.zeros((p, p))
    for s in strata_list:
        if len(s.unique_event_times) == 0:
            continue
        _ll, _sc, im = _score_and_information(
            beta, s.time, s.event, s.X[:, :p], s.unique_event_times, ties,
            entry=s.entry,
        )
        info += im
    return info


def _stratum_score_residuals(
    s, beta: NDArray, ties: str
) -> NDArray:
    """Score residuals L (n_k, p) for one stratum's rows (sorted order)."""
    n, p = s.X.shape
    uet = s.unique_event_times
    m = len(uet)
    L = np.zeros((n, p))
    if m == 0:
        return L

    eta = s.X @ beta
    eta_c = eta - np.max(eta)
    w = np.exp(eta_c)
    wx = w[:, None] * s.X

    cs0 = _reverse_cumsum(w)
    cs1 = _reverse_cumsum(wx)
    first = np.searchsorted(s.time, uet, side="left")
    S0 = cs0[first]                       # (m,)
    S1 = cs1[first]                       # (m, p)
    if s.entry is not None:
        from pystatistics.survival._cox import _entry_adjustment
        S0 = S0 - _entry_adjustment(w, s.entry, uet)
        S1 = S1 - _entry_adjustment(wx, s.entry, uet)
    xbar = S1 / S0[:, None]               # (m, p) Breslow risk mean per event time

    death = s.event == 1
    dtimes = s.time[death]
    dgrp = np.searchsorted(uet, dtimes)          # event-time index of each death
    d_k = np.bincount(dgrp, minlength=m).astype(float)

    # Breslow baseline-hazard increment dLambda0(t_k) = d_k / S0.
    dL0 = d_k / S0                                # (m,)

    # Cumulative sums so each row's at-risk integral is O(1).
    cumA = np.concatenate([[0.0], np.cumsum(dL0)])            # (m+1,)
    cumB = np.concatenate([np.zeros((1, p)),
                           np.cumsum(xbar * dL0[:, None], axis=0)])  # (m+1, p)

    # Row exit index hi = # event times <= time_i; entry index lo similarly.
    hi = np.searchsorted(uet, s.time, side="right")
    if s.entry is not None:
        lo = np.searchsorted(uet, s.entry, side="right")
    else:
        lo = np.zeros(n, dtype=np.intp)

    # Hazard (at-risk) term: w_i [ x_i (A_hi - A_lo) - (B_hi - B_lo) ].
    dA = cumA[hi] - cumA[lo]                      # (n,)
    dB = cumB[hi] - cumB[lo]                      # (n, p)
    L -= w[:, None] * (s.X * dA[:, None] - dB)

    # Event term: (x_i - xbar(t_i)) for each death (Breslow).
    L[death] += s.X[death] - xbar[dgrp]

    if ties == "efron":
        _apply_efron_correction(L, s, beta, w, S0, S1, xbar, uet, d_k, dgrp,
                                death, cumA, cumB, hi, lo)
    return L


def _apply_efron_correction(
    L, s, beta, w, S0, S1, xbar, uet, d_k, dgrp, death, cumA, cumB, hi, lo
):
    """Adjust the score residuals within genuine multi-death tie groups so they
    match Efron's partial-likelihood tie handling (single-death times are
    already exact under the Breslow form above).

    At tie time t_k with d deaths, Efron resolves the tie in d fractional steps
    a = 0..d-1: step a uses denominator ``S0 - (a/d) dS0`` and mean ``mean_a``,
    and each tied death is in that step's risk set with weight ``1 - a/d``. So
    relative to the Breslow terms already applied:
      - event term of a tied death: replace ``xbar_B`` by the average step-mean;
      - hazard term of any at-risk row: replace the Breslow increment by the
        Efron one (``Σ_a 1/denom_a``, ``Σ_a mean_a/denom_a``);
      - a tied death carries only fractional risk in its own group, so its
        hazard term drops the ``a/d`` share (added back here).
    """
    p = s.X.shape[1]
    tie_groups = np.nonzero(d_k > 1)[0]
    if len(tie_groups) == 0:
        return
    death_idx = np.nonzero(death)[0]
    for k in tie_groups:
        dk = int(d_k[k])
        members = death_idx[dgrp == k]           # rows dying at uet[k]
        dS0 = w[members].sum()
        dS1 = (w[members, None] * s.X[members]).sum(0)

        efron_dL0 = 0.0          # Σ_a 1/denom_a          (Efron hazard incr.)
        efron_meanL0 = np.zeros(p)   # Σ_a mean_a/denom_a
        frac_dL0 = 0.0           # Σ_a (a/d)/denom_a      (fractional-risk share)
        frac_meanL0 = np.zeros(p)
        event_mean = np.zeros(p)     # (1/d) Σ_a mean_a
        for a in range(dk):
            frac = a / dk
            denom = S0[k] - frac * dS0
            mean_a = (S1[k] - frac * dS1) / denom
            efron_dL0 += 1.0 / denom
            efron_meanL0 += mean_a / denom
            frac_dL0 += frac / denom
            frac_meanL0 += frac * mean_a / denom
            event_mean += mean_a
        event_mean /= dk
        dL0_B = dk / S0[k]
        xbar_B = S1[k] / S0[k]

        # (1) Event term of the tied deaths: Breslow added (x - xbar_B); Efron
        #     wants (x - event_mean).
        L[members] += xbar_B - event_mean

        # (2) Hazard term for every row at risk at t_k: swap Breslow -> Efron.
        rows = np.nonzero((lo <= k) & (hi > k))[0]
        breslow_haz = w[rows, None] * (s.X[rows] - xbar_B) * dL0_B
        efron_haz = w[rows, None] * (s.X[rows] * efron_dL0 - efron_meanL0)
        L[rows] += breslow_haz - efron_haz       # undo Breslow, apply Efron

        # (3) Tied deaths are at only fractional risk in their own group; add
        #     back the (a/d) share of the Efron hazard term subtracted in (2).
        L[members] += w[members, None] * (s.X[members] * frac_dL0 - frac_meanL0)


def cox_robust_variance(
    time: NDArray,
    event: NDArray,
    X: NDArray,
    beta: NDArray,
    ties: str,
    strata: NDArray | None,
    entry: NDArray | None,
    cluster: NDArray | None,
) -> dict:
    """Sandwich variance + dfbeta residuals for a fitted Cox model.

    Parameters
    ----------
    cluster : (n,) or None
        Grouping label per row; rows sharing a label are one independent unit.
        None treats every row as its own unit (ordinary Lin-Wei robust SE).

    Returns
    -------
    dict with ``robust_cov`` (p,p), ``naive_cov`` (p,p), ``dfbeta`` (n,p, in
    original row order), ``robust_se`` (p,), ``naive_se`` (p,).
    """
    n, p = X.shape
    # Mean-center (invariant to the fit; stabilizes the information — see
    # _cox.cox_fit). Score residuals and dfbeta are shift-invariant too.
    X = X - X.mean(axis=0)
    if strata is None:
        strata = np.zeros(n, dtype=np.intp)
    # Build per-stratum sorted views WITH original-row indices so residuals can
    # be scattered back to input order.
    idx_all = np.arange(n)
    strata_list = _build_strata(time, event,
                                np.column_stack([X, idx_all]), strata,
                                entry=entry)

    naive_cov = np.linalg.inv(_pooled_information(strata_list, beta, ties, p))

    L = np.zeros((n, p))
    for s in strata_list:
        orig = s.X[:, -1].astype(np.intp)        # recovered original indices
        s_core = s.__class__(time=s.time, event=s.event, X=s.X[:, :-1],
                             unique_event_times=s.unique_event_times,
                             entry=s.entry)
        L[orig] = _stratum_score_residuals(s_core, beta, ties)

    dfbeta = L @ naive_cov                        # (n, p)

    if cluster is not None:
        cl = np.asarray(cluster).ravel()
        _labels, inv = np.unique(cl, return_inverse=True)
        D = np.zeros((int(inv.max()) + 1, p))
        np.add.at(D, inv, dfbeta)
    else:
        D = dfbeta

    robust_cov = D.T @ D
    robust_se = np.sqrt(np.maximum(np.diag(robust_cov), 0.0))
    naive_se = np.sqrt(np.maximum(np.diag(naive_cov), 0.0))
    return {"robust_cov": robust_cov, "naive_cov": naive_cov,
            "dfbeta": dfbeta, "robust_se": robust_se, "naive_se": naive_se}
