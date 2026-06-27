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


def cox_fit(
    time: NDArray,
    event: NDArray,
    X: NDArray,
    ties: str = "efron",
    tol: float = 1e-9,
    max_iter: int = 20,
    conf_level: float = 0.95,
) -> CoxParams:
    """Fit Cox proportional hazards model.

    Parameters
    ----------
    time : NDArray
        (n,) time to event or censoring.
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

    Returns
    -------
    CoxParams
    """
    n, p = X.shape

    # Sort by descending time (events before censoring at ties)
    # This puts the largest times first, matching the way we accumulate risk sets
    order = np.lexsort((event, time))  # ascending time, censored before events
    t_sorted = time[order]
    e_sorted = event[order]
    X_sorted = X[order]

    # Find distinct event times and group info
    event_mask = e_sorted == 1
    event_times_all = t_sorted[event_mask]
    unique_event_times = np.unique(event_times_all)

    n_events_total = int(np.sum(event))

    if n_events_total == 0:
        # No events — cannot fit Cox model
        return CoxParams(
            coefficients=np.zeros(p, dtype=np.float64),
            hazard_ratios=np.ones(p, dtype=np.float64),
            standard_errors=np.full(p, np.inf),
            z_values=np.zeros(p, dtype=np.float64),
            p_values=np.ones(p, dtype=np.float64),
            loglik=(0.0, 0.0),
            concordance=0.5,
            n_events=0,
            n_observations=n,
            n_iter=0,
            converged=True,
            ties=ties,
            conf_level=conf_level,
        )

    # --- Newton-Raphson ---
    beta = np.zeros(p, dtype=np.float64)

    # Null log-likelihood (β=0)
    null_loglik = _partial_loglik(
        beta, t_sorted, e_sorted, X_sorted, unique_event_times, ties
    )

    converged = False
    n_iter = 0
    loglik_old = null_loglik

    for iteration in range(1, max_iter + 1):
        loglik, score, info_matrix = _score_and_information(
            beta, t_sorted, e_sorted, X_sorted, unique_event_times, ties
        )

        # Newton step: β_new = β + I^{-1} @ U
        try:
            step = np.linalg.solve(info_matrix, score)
        except np.linalg.LinAlgError:
            # Singular information matrix
            break

        # Step-halving to prevent overflow (R's coxph does this too)
        # Limit step size so exp(X @ beta) doesn't overflow
        max_step = np.max(np.abs(step))
        if max_step > 5.0:
            step = step * (5.0 / max_step)

        beta_new = beta + step

        # Check convergence on relative change in log-likelihood
        loglik_new = _partial_loglik(
            beta_new, t_sorted, e_sorted, X_sorted, unique_event_times, ties
        )

        # R-style convergence: max|β_new - β| < tol
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            n_iter = iteration
            break

        # Also accept convergence on relative loglik change
        if iteration > 1 and abs(loglik_new - loglik_old) / (abs(loglik_old) + 0.1) < tol:
            beta = beta_new
            converged = True
            n_iter = iteration
            break

        beta = beta_new
        loglik_old = loglik_new
        n_iter = iteration

    # Final log-likelihood
    model_loglik = _partial_loglik(
        beta, t_sorted, e_sorted, X_sorted, unique_event_times, ties
    )

    # Standard errors from observed information matrix
    _, _, info_final = _score_and_information(
        beta, t_sorted, e_sorted, X_sorted, unique_event_times, ties
    )

    try:
        var_matrix = np.linalg.inv(info_final)
        se = np.sqrt(np.maximum(np.diag(var_matrix), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(p, np.inf)

    # Wald z-statistics and p-values
    z = np.where(se > 0, beta / se, 0.0)
    p_values = 2.0 * stats.norm.sf(np.abs(z))

    # Harrell's concordance
    concordance = _concordance(beta, time, event, X)

    return CoxParams(
        coefficients=beta,
        hazard_ratios=np.exp(beta),
        standard_errors=se,
        z_values=z,
        p_values=p_values,
        loglik=(null_loglik, model_loglik),
        concordance=concordance,
        n_events=n_events_total,
        n_observations=n,
        n_iter=n_iter,
        converged=converged,
        ties=ties,
        conf_level=conf_level,
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


def _partial_loglik(
    beta: NDArray,
    time: NDArray,
    event: NDArray,
    X: NDArray,
    unique_event_times: NDArray,
    ties: str,
) -> float:
    """Compute the Cox partial log-likelihood.

    Vectorized: the risk-set denominators S0(t_j) come from one reverse
    cumulative sum over time-sorted data, indexed at each event time, rather
    than re-scanning the risk set per event time. Efron's tie correction adds a
    bounded inner loop only over genuine multi-death ties (single-death times
    reduce to Breslow and are handled in the vectorized term).

    Parameters
    ----------
    beta : (p,)
    time, event, X : (n,) / (n,) / (n, p), sorted ascending by time
    unique_event_times : distinct event times (ascending)
    ties : "efron" or "breslow"
    """
    n = len(time)
    eta = X @ beta
    eta_max = np.max(eta) if n > 0 else 0.0          # center (cancels)
    eta_c = eta - eta_max
    w = np.exp(eta_c)

    cs0 = _reverse_cumsum(w)
    first = np.searchsorted(time, unique_event_times, side="left")
    S0 = cs0[first]                                  # (m,) risk-set sums

    dg = _death_group_sums(time, event, X, eta_c, w, unique_event_times)
    d, dS0, eta_c_sum = dg["d"], dg["dS0"], dg["eta_c_sum"]

    # Breslow term for every event time (exact for single-death times).
    loglik = float(eta_c_sum.sum() - (d * np.log(S0)).sum())

    if ties == "efron":
        for k in np.nonzero(d > 1)[0]:
            dk = int(d[k])
            loglik += dk * np.log(S0[k])             # undo the Breslow term
            for s in range(dk):
                loglik -= np.log(S0[k] - (s / dk) * dS0[k])
    return loglik


def _score_and_information(
    beta: NDArray,
    time: NDArray,
    event: NDArray,
    X: NDArray,
    unique_event_times: NDArray,
    ties: str,
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

    dg = _death_group_sums(time, event, X, eta_c, w, unique_event_times)
    d = dg["d"]

    # --- Breslow term, vectorized over ALL event times (exact for d_j == 1) ---
    mean0 = S1 / S0[:, None]                          # (m, p)
    loglik = float(dg["eta_c_sum"].sum() - (d * np.log(S0)).sum())
    score = dg["eX_sum"].sum(0) - (d[:, None] * mean0).sum(0)
    info_matrix = (d[:, None, None]
                   * (S2 / S0[:, None, None]
                      - mean0[:, :, None] * mean0[:, None, :])).sum(0)

    # --- Efron correction: only for genuine multi-death ties (few, small) ---
    if ties == "efron":
        dS0, dS1, dS2 = dg["dS0"], dg["dS1"], dg["dS2"]
        for k in np.nonzero(d > 1)[0]:
            dk = int(d[k])
            s0k, s1k, s2k = S0[k], S1[k], S2[k]
            # Undo the Breslow term added above for this tie group, ...
            loglik += dk * np.log(s0k)
            score += dk * (s1k / s0k)
            info_matrix -= dk * (s2k / s0k - np.outer(s1k / s0k, s1k / s0k))
            # ... then add the proper Efron sum over the d_k tied deaths.
            for s in range(dk):
                frac = s / dk
                denom = s0k - frac * dS0[k]
                num1 = s1k - frac * dS1[k]
                num2 = s2k - frac * dS2[k]
                mean = num1 / denom
                loglik -= np.log(denom)
                score -= mean
                info_matrix += num2 / denom - np.outer(mean, mean)

    return loglik, score, info_matrix


def _fenwick_add(tree: NDArray, r: int, size: int) -> None:
    """Add 1 at 1-based index ``r`` of a Fenwick (binary-indexed) tree."""
    while r <= size:
        tree[r] += 1.0
        r += r & (-r)


def _fenwick_prefix(tree: NDArray, r: int) -> float:
    """Sum of tree[1..r] (count of inserted items with rank <= r)."""
    s = 0.0
    while r > 0:
        s += tree[r]
        r -= r & (-r)
    return s


def _concordance(
    beta: NDArray,
    time: NDArray,
    event: NDArray,
    X: NDArray,
) -> float:
    """Harrell's concordance statistic (C-statistic), O(n log n).

    C = P(risk_i > risk_j | T_i < T_j, event_i = 1). Over comparable ordered
    pairs (i an event, j with strictly larger time), this counts how many have a
    larger / smaller / equal risk score. The naive form is an O(n^2) all-pairs
    scan; here a Fenwick tree keyed by risk-score rank counts each event's
    contribution in O(log n). Subjects are processed in descending time so the
    tree holds exactly the strictly-later-time subjects when an event is queried;
    a whole tied-time block is inserted only after its events are queried, which
    enforces the strict T_i < T_j comparability. Bit-for-bit equivalent to the
    all-pairs definition.
    """
    eta = X @ beta
    n = len(time)

    # Rank risk scores into 1..K (ties share a rank).
    uniq, inv = np.unique(eta, return_inverse=True)
    size = len(uniq)
    rank = (inv + 1).astype(np.intp)

    order = np.argsort(time, kind="mergesort")
    t = np.asarray(time)[order]
    e = np.asarray(event)[order]
    rk = rank[order]

    tree = np.zeros(size + 1, dtype=np.float64)
    concordant = discordant = tied_risk = 0.0
    seen = 0

    i = n - 1
    while i >= 0:
        # Identify the block of equal times [lo .. i].
        tj = t[i]
        lo = i
        while lo >= 0 and t[lo] == tj:
            lo -= 1
        lo += 1
        # Query events in this block against strictly-later-time subjects.
        for k in range(lo, i + 1):
            if e[k] == 1:
                r = int(rk[k])
                less = _fenwick_prefix(tree, r - 1)     # eta_j < eta_i
                leq = _fenwick_prefix(tree, r)
                concordant += less
                tied_risk += leq - less
                discordant += seen - leq
        # Insert the whole block (now they become "later time" for earlier events).
        for k in range(lo, i + 1):
            _fenwick_add(tree, int(rk[k]), size)
            seen += 1
        i = lo - 1

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5

    return (concordant + 0.5 * tied_risk) / total
