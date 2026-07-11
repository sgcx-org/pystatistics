"""
Cox Proportional Hazards model via Newton-Raphson.

Implements Efron's and Breslow's methods for tied event times,
matching R's survival::coxph().

Algorithm:
    Initialize β = 0
    For iteration 1..max_iter:
        Compute: partial log-likelihood L(β), score U(β), information I(β)
        β_new = β + I(β)^{-1} @ U(β)
        Check convergence: max|β_new - β| < tol

Efron's partial likelihood (R default):
    L(β) = Σ_{j: event times} [ Σ_{i ∈ D_j} x_i @ β
            - Σ_{s=0}^{d_j-1} log(Σ_{l ∈ R_j} exp(x_l @ β)
                - (s/d_j) * Σ_{i ∈ D_j} exp(x_i @ β)) ]

    where D_j = set of events at time t_j, d_j = |D_j|,
          R_j = risk set at time t_j (alive just before t_j).

References:
    Cox, D. R. (1972). Regression models and life-tables. JRSS-B, 34(2), 187-220.
    Efron, B. (1977). The efficiency of Cox's likelihood function for
        censored data. JASA, 72(359), 557-565.
    R Core Team. survival::coxph, agreg.fit
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from pystatistics.survival._common import CoxParams
from pystatistics.survival._cox_newton import flag_infinite_coefs, solve_cox_newton
from pystatistics.survival._concordance_kernel import (
    concordance_counts_simple, concordance_counts_truncated,
)


def cox_fit(
    time: NDArray,
    event: NDArray,
    X: NDArray,
    ties: str = "efron",
    tol: float = 1e-9,
    max_iter: int = 20,
    conf_level: float = 0.95,
    entry: NDArray | None = None,
) -> CoxParams:
    """Fit Cox proportional hazards model.

    Parameters
    ----------
    time : NDArray
        (n,) time to event or censoring (the risk-interval exit).
    event : NDArray
        (n,) event indicator (1=event, 0=censored).
    X : NDArray
        (n, p) covariate matrix (NO intercept).
    ties : str
        Method for handling tied event times: "efron" (default) or "breslow".
    tol : float
        Convergence tolerance (max absolute change in β).
    max_iter : int
        Maximum Newton-Raphson iterations.
    entry : NDArray or None
        (n,) counting-process start times: each row is at risk on
        ``(entry, time]``. Enables time-varying covariates (a subject split
        into several rows) and left truncation. None = at risk from 0.

    Returns
    -------
    CoxParams
    """
    n, p = X.shape

    # Mean-center the covariates. The Cox partial likelihood, score, and
    # information are all exactly invariant to a constant shift of any covariate
    # (it cancels in every risk-set ratio), but computing the information on raw,
    # large-magnitude covariates loses precision to cancellation (E[XX'] -
    # E[X]E[X]') and can yield a numerically indefinite matrix — the failure R
    # avoids by centering. Concordance depends only on the ordering of X @ beta,
    # which a constant shift preserves. Coefficients are reported unchanged.
    X = X - X.mean(axis=0)

    # Sort by descending time (events before censoring at ties)
    # This puts the largest times first, matching the way we accumulate risk sets
    order = np.lexsort((event, time))  # ascending time, censored before events
    t_sorted = time[order]
    e_sorted = event[order]
    X_sorted = X[order]
    entry_sorted = entry[order] if entry is not None else None

    # Find distinct event times and group info
    event_mask = e_sorted == 1
    event_times_all = t_sorted[event_mask]
    unique_event_times = np.unique(event_times_all)

    n_events_total = int(np.sum(event))

    if n_events_total == 0:
        # No events — the partial likelihood is flat and the model is
        # unidentified. Report converged=False (like the discrete-time twin) and
        # NaN concordance rather than a fabricated 0.5 that asserts a fit that
        # never ran; R likewise returns NA here, not a definite number.
        return CoxParams(
            coefficients=np.zeros(p, dtype=np.float64),
            hazard_ratios=np.ones(p, dtype=np.float64),
            standard_errors=np.full(p, np.inf),
            z_values=np.full(p, np.nan),
            p_values=np.full(p, np.nan),
            loglik=(0.0, 0.0),
            concordance=np.nan,
            n_events=0,
            n_observations=n,
            n_iter=0,
            converged=False,
            ties=ties,
            conf_level=conf_level,
        )

    # --- Newton-Raphson (shared driver) ---
    def eval_full(b: NDArray) -> tuple[float, NDArray, NDArray]:
        return _score_and_information(
            b, t_sorted, e_sorted, X_sorted, unique_event_times, ties,
            entry=entry_sorted,
        )

    def eval_loglik(b: NDArray) -> float:
        return _partial_loglik(
            b, t_sorted, e_sorted, X_sorted, unique_event_times, ties,
            entry=entry_sorted,
        )

    fit = solve_cox_newton(p, eval_full, eval_loglik, tol, max_iter)
    beta = fit.beta

    # Standard errors from observed information matrix
    infinite_coefs: tuple[int, ...] = ()
    try:
        var_matrix = np.linalg.inv(fit.information)
        se = np.sqrt(np.maximum(np.diag(var_matrix), 0.0))
        if fit.converged:
            infinite_coefs = flag_infinite_coefs(beta, fit.score, var_matrix, tol)
    except np.linalg.LinAlgError:
        se = np.full(p, np.inf)

    # Wald z-statistics and p-values
    z = np.where(se > 0, beta / se, 0.0)
    p_values = 2.0 * stats.norm.sf(np.abs(z))

    # Harrell's concordance
    concordant, discordant, tied_risk = _concordance_counts(
        beta, time, event, X, entry=entry)
    concordance = _concordance_ratio(concordant, discordant, tied_risk)

    return CoxParams(
        coefficients=beta,
        hazard_ratios=np.exp(beta),
        standard_errors=se,
        z_values=z,
        p_values=p_values,
        loglik=(fit.null_loglik, fit.model_loglik),
        concordance=concordance,
        n_events=n_events_total,
        n_observations=n,
        n_iter=fit.n_iter,
        converged=fit.converged,
        ties=ties,
        conf_level=conf_level,
        infinite_coefs=infinite_coefs,
    )


def _reverse_cumsum(a: NDArray) -> NDArray:
    """Reverse cumulative sum along axis 0: out[k] = sum of a[k:].

    With observations sorted by ascending time, index ``k`` onward is exactly the
    risk set {i : time_i >= time_k}. So a single reverse cumulative sum yields,
    at every index, the risk-set sum needed by the Cox partial likelihood —
    turning the per-event-time O(n) re-scan into one O(n) sweep.
    """
    return np.cumsum(a[::-1], axis=0)[::-1]


def _death_group_sums(
    time: NDArray, event: NDArray, X: NDArray, eta_c: NDArray, w: NDArray,
    unique_event_times: NDArray,
) -> dict:
    """Per-event-time sums over the *deaths* at that time (vectorized via bincount).

    Returns d_j (deaths), the death-set weighted sums dS0/dS1/dS2, the sum of
    covariates over deaths (eX_sum), and the sum of centered linear predictors
    over deaths (eta_c_sum) — one entry per unique event time.
    """
    p = X.shape[1]
    m = len(unique_event_times)
    death = event == 1
    grp = np.searchsorted(unique_event_times, time[death])  # event-time index
    wd = w[death]
    Xd = X[death]
    d = np.bincount(grp, minlength=m).astype(np.float64)
    dS0 = np.bincount(grp, weights=wd, minlength=m)
    eta_c_sum = np.bincount(grp, weights=eta_c[death], minlength=m)
    dS1 = np.empty((m, p)); eX_sum = np.empty((m, p)); dS2 = np.empty((m, p, p))
    for j in range(p):
        dS1[:, j] = np.bincount(grp, weights=wd * Xd[:, j], minlength=m)
        eX_sum[:, j] = np.bincount(grp, weights=Xd[:, j], minlength=m)
        for l in range(p):
            dS2[:, j, l] = np.bincount(grp, weights=wd * Xd[:, j] * Xd[:, l],
                                       minlength=m)
    return {"d": d, "dS0": dS0, "dS1": dS1, "dS2": dS2,
            "eX_sum": eX_sum, "eta_c_sum": eta_c_sum}


def _entry_adjustment(
    values: NDArray,
    entry: NDArray,
    unique_event_times: NDArray,
) -> NDArray:
    """Risk-set correction for delayed entry: sum of ``values`` over rows with
    ``entry >= t_j``, per event time.

    With counting-process data the risk set at ``t`` is ``{entry < t <= time}``,
    so each stop-side reverse-cumsum ``A(t) = sum over time >= t`` must be
    reduced by ``B(t) = sum over entry >= t``. Computed the same way: one
    reverse cumulative sum over entry-sorted rows, indexed per event time.
    """
    eorder = np.argsort(entry, kind="mergesort")
    esorted = entry[eorder]
    cs = _reverse_cumsum(values[eorder])
    pad = np.zeros((1,) + values.shape[1:], dtype=np.float64)
    cs = np.concatenate([cs, pad], axis=0)           # index n -> empty sum
    idx = np.searchsorted(esorted, unique_event_times, side="left")
    return cs[idx]


def _efron_steps(groups: NDArray, dkg: NDArray) -> tuple[NDArray, NDArray]:
    """Flatten multi-death tie groups into per-step arrays, fully vectorized.

    Returns ``(gk, fracs)`` where ``gk[j]`` is the event-time index of step j and
    ``fracs[j] = step / d_k`` runs 0, 1/d_k, …, (d_k-1)/d_k within each group.
    Avoids a Python loop over tie groups (which dominated on data with many
    small multi-death groups, e.g. integer-day survival times).
    """
    dkg = dkg.astype(np.intp)
    total = int(dkg.sum())
    gk = np.repeat(groups, dkg)                       # group index per step
    starts = np.repeat(np.cumsum(dkg) - dkg, dkg)     # step 0 offset per group
    steps = np.arange(total) - starts                 # 0..d_k-1 within a group
    fracs = steps / np.repeat(dkg, dkg)
    return gk, fracs


def _partial_loglik(
    beta: NDArray,
    time: NDArray,
    event: NDArray,
    X: NDArray,
    unique_event_times: NDArray,
    ties: str,
    entry: NDArray | None = None,
) -> float:
    """Compute the Cox partial log-likelihood.

    Vectorized: the risk-set denominators S0(t_j) come from one reverse
    cumulative sum over time-sorted data, indexed at each event time, rather
    than re-scanning the risk set per event time. Efron's tie correction adds a
    bounded inner loop only over genuine multi-death ties (single-death times
    reduce to Breslow and are handled in the vectorized term). With ``entry``
    (counting-process rows), each risk-set sum is reduced by the not-yet-entered
    rows (see ``_entry_adjustment``).

    Parameters
    ----------
    beta : (p,)
    time, event, X : (n,) / (n,) / (n, p), sorted ascending by time
    unique_event_times : distinct event times (ascending)
    ties : "efron" or "breslow"
    entry : (n,) or None — row entry times, aligned with the sorted rows
    """
    n = len(time)
    eta = X @ beta
    eta_max = np.max(eta) if n > 0 else 0.0          # center (cancels)
    eta_c = eta - eta_max
    w = np.exp(eta_c)

    cs0 = _reverse_cumsum(w)
    first = np.searchsorted(time, unique_event_times, side="left")
    S0 = cs0[first]                                  # (m,) risk-set sums
    if entry is not None:
        S0 = S0 - _entry_adjustment(w, entry, unique_event_times)

    dg = _death_group_sums(time, event, X, eta_c, w, unique_event_times)
    d, dS0, eta_c_sum = dg["d"], dg["dS0"], dg["eta_c_sum"]

    # Breslow term for every event time (exact for single-death times).
    loglik = float(eta_c_sum.sum() - (d * np.log(S0)).sum())

    if ties == "efron":
        groups = np.nonzero(d > 1)[0]
        if len(groups) > 0:
            loglik += float((d[groups] * np.log(S0[groups])).sum())  # undo Breslow
            gk, fracs = _efron_steps(groups, d[groups])
            loglik -= float(np.log(S0[gk] - fracs * dS0[gk]).sum())
    return loglik


def _score_and_information(
    beta: NDArray,
    time: NDArray,
    event: NDArray,
    X: NDArray,
    unique_event_times: NDArray,
    ties: str,
    entry: NDArray | None = None,
) -> tuple[float, NDArray, NDArray]:
    """Compute log-likelihood, score vector, and observed information matrix.

    Returns
    -------
    (loglik, score, info_matrix)
        loglik : float
        score : (p,) — gradient of log-likelihood
        info_matrix : (p, p) — negative Hessian (observed information)
    """
    n, p = X.shape
    eta = X @ beta
    eta_max = np.max(eta) if n > 0 else 0.0          # center (cancels)
    eta_c = eta - eta_max
    w = np.exp(eta_c)
    wx = w[:, None] * X

    # Risk-set sums S0/S1/S2 at every event time via one reverse cumsum each,
    # indexed by the first occurrence of each event time (risk set = time >= t_j).
    cs0 = _reverse_cumsum(w)                          # (n,)
    cs1 = _reverse_cumsum(wx)                         # (n, p)
    cs2 = _reverse_cumsum(wx[:, :, None] * X[:, None, :])  # (n, p, p) — O(n p^2)
    first = np.searchsorted(time, unique_event_times, side="left")
    S0 = cs0[first]; S1 = cs1[first]; S2 = cs2[first]
    if entry is not None:
        # Counting-process rows: remove the not-yet-entered from each risk set.
        S0 = S0 - _entry_adjustment(w, entry, unique_event_times)
        S1 = S1 - _entry_adjustment(wx, entry, unique_event_times)
        S2 = S2 - _entry_adjustment(wx[:, :, None] * X[:, None, :],
                                    entry, unique_event_times)

    dg = _death_group_sums(time, event, X, eta_c, w, unique_event_times)
    d = dg["d"]

    # --- Breslow term, vectorized over ALL event times (exact for d_j == 1) ---
    mean0 = S1 / S0[:, None]                          # (m, p)
    loglik = float(dg["eta_c_sum"].sum() - (d * np.log(S0)).sum())
    score = dg["eX_sum"].sum(0) - (d[:, None] * mean0).sum(0)
    info_matrix = (d[:, None, None]
                   * (S2 / S0[:, None, None]
                      - mean0[:, :, None] * mean0[:, None, :])).sum(0)

    # --- Efron correction: only for genuine multi-death ties ---
    # Vectorized over ALL tie groups and their fractional steps at once (the
    # per-group / per-step Python loop was the fit's hot spot on tied data).
    if ties == "efron":
        groups = np.nonzero(d > 1)[0]
        if len(groups) > 0:
            dS0, dS1, dS2 = dg["dS0"], dg["dS1"], dg["dS2"]
            dkg = d[groups]                              # (G,)
            s0g, s1g, s2g = S0[groups], S1[groups], S2[groups]
            mean0g = s1g / s0g[:, None]                  # (G, p)

            # Undo the Breslow term (added above) for every tie group at once.
            loglik += float((dkg * np.log(s0g)).sum())
            score += (dkg[:, None] * mean0g).sum(0)
            info_matrix -= (dkg[:, None, None]
                            * (s2g / s0g[:, None, None]
                               - mean0g[:, :, None] * mean0g[:, None, :])).sum(0)

            # Add the proper Efron sum, flattening (group, step) into one axis.
            gk, fracs = _efron_steps(groups, dkg)
            denom = S0[gk] - fracs * dS0[gk]             # (T,)
            num1 = S1[gk] - fracs[:, None] * dS1[gk]     # (T, p)
            num2 = S2[gk] - fracs[:, None, None] * dS2[gk]  # (T, p, p)
            mean = num1 / denom[:, None]                 # (T, p)
            loglik -= float(np.log(denom).sum())
            score -= mean.sum(0)
            info_matrix += (num2 / denom[:, None, None]
                            - mean[:, :, None] * mean[:, None, :]).sum(0)

    return loglik, score, info_matrix


def _concordance_counts(
    beta: NDArray,
    time: NDArray,
    event: NDArray,
    X: NDArray,
    entry: NDArray | None = None,
) -> tuple[float, float, float]:
    """Concordant / discordant / risk-tied pair counts, O(n log n).

    C = P(risk_i > risk_j | T_i < T_j, event_i = 1). Over comparable ordered
    pairs — an event at t versus every row at risk beyond t — this counts how
    many have a larger / smaller / equal risk score. A Fenwick tree keyed by
    risk-score rank holds the currently-at-risk rows while an ascending sweep
    over event times activates rows as their ``entry`` passes and deactivates
    them as their exit passes, so each event's contribution costs O(log n).

    Conventions (matching R's ``concordance``): a row censored AT an event
    time outlives the event (comparable); two events at the same time are a
    time tie (not comparable); with counting-process rows, a row is comparable
    with an event at t only while at risk, ``entry < t <= exit`` — so
    non-overlapping spells of one subject never form a pair.

    Returned as raw counts (not the ratio) so a stratified fit can sum
    within-stratum counts before forming a single C — matching R's
    ``concordance.coxph``, which only compares pairs sharing a stratum.
    """
    eta = X @ beta
    n = len(time)

    # Rank risk scores into 1..K (ties share a rank).
    uniq, inv = np.unique(eta, return_inverse=True)
    size = len(uniq)
    rank = (inv + 1).astype(np.int64)

    e = np.asarray(event, dtype=np.float64)
    uet = np.unique(time[e == 1])

    if entry is None:
        # Add-only descending sweep (every subject at risk from the start).
        order = np.argsort(time, kind="mergesort")
        return concordance_counts_simple(
            rank[order], np.asarray(time, dtype=np.float64)[order], e[order],
            uet, size)

    # Counting-process rows: activation + deactivation sweep.
    t = np.asarray(time, dtype=np.float64)
    entry_vals = np.asarray(entry, dtype=np.float64)
    stop_order = np.argsort(t, kind="mergesort")
    event_rows = stop_order[e[stop_order] == 1].astype(np.int64)
    return concordance_counts_truncated(rank, t, entry_vals, event_rows, uet,
                                        size)


def _concordance_ratio(
    concordant: float, discordant: float, tied_risk: float
) -> float:
    """Harrell's C from concordant / discordant / risk-tied pair counts.

    With no comparable pairs the C-statistic is undefined; return NaN (as R
    does) rather than a fabricated 0.5 that reads as 'no discrimination'.
    """
    total = concordant + discordant + tied_risk
    if total == 0:
        return np.nan
    return (concordant + 0.5 * tied_risk) / total
