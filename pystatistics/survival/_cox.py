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
            z_statistics=np.zeros(p, dtype=np.float64),
            p_values=np.ones(p, dtype=np.float64),
            loglik=(0.0, 0.0),
            concordance=0.5,
            n_events=0,
            n_observations=n,
            n_iter=0,
            converged=True,
            ties=ties,
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
        z_statistics=z,
        p_values=p_values,
        loglik=(null_loglik, model_loglik),
        concordance=concordance,
        n_events=n_events_total,
        n_observations=n,
        n_iter=n_iter,
        converged=converged,
        ties=ties,
    )


def _partial_loglik(
    beta: NDArray,
    time: NDArray,
    event: NDArray,
    X: NDArray,
    unique_event_times: NDArray,
    ties: str,
) -> float:
    """Compute partial log-likelihood.

    Parameters
    ----------
    beta : (p,)
    time : (n,) sorted ascending
    event : (n,) sorted
    X : (n, p) sorted
    unique_event_times : distinct event times
    ties : "efron" or "breslow"
    """
    n = len(time)
    eta = X @ beta  # (n,) linear predictor

    # Center eta for numerical stability (cancels in partial likelihood)
    eta_max = np.max(eta) if len(eta) > 0 else 0.0
    eta_c = eta - eta_max
    exp_eta = np.exp(eta_c)

    loglik = 0.0

    for t_j in unique_event_times:
        # Risk set: all subjects with time >= t_j
        risk_mask = time >= t_j
        risk_exp_sum = np.sum(exp_eta[risk_mask])

        # Events at this time
        event_at_tj = (time == t_j) & (event == 1)
        d_j = int(np.sum(event_at_tj))

        if d_j == 0:
            continue

        # Sum of centered linear predictors for events
        event_eta_sum = np.sum(eta_c[event_at_tj])

        if ties == "breslow" or d_j == 1:
            # Breslow: simple
            if risk_exp_sum > 0:
                loglik += event_eta_sum - d_j * np.log(risk_exp_sum)

        elif ties == "efron":
            # Efron approximation
            death_exp_sum = np.sum(exp_eta[event_at_tj])

            for s in range(d_j):
                denom = risk_exp_sum - (s / d_j) * death_exp_sum
                if denom > 0:
                    loglik -= np.log(denom)

            loglik += event_eta_sum

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

    # Center eta for numerical stability
    eta_max = np.max(eta) if n > 0 else 0.0
    eta_c = eta - eta_max
    exp_eta = np.exp(eta_c)

    loglik = 0.0
    score = np.zeros(p, dtype=np.float64)
    info_matrix = np.zeros((p, p), dtype=np.float64)

    for t_j in unique_event_times:
        # Risk set
        risk_mask = time >= t_j
        risk_exp = exp_eta[risk_mask]
        risk_X = X[risk_mask]

        # Weighted sums over risk set
        S0 = np.sum(risk_exp)                          # scalar
        S1 = risk_X.T @ risk_exp                       # (p,)
        S2 = (risk_X * risk_exp[:, np.newaxis]).T @ risk_X  # (p, p)

        # Events at this time
        event_at_tj = (time == t_j) & (event == 1)
        d_j = int(np.sum(event_at_tj))

        if d_j == 0:
            continue

        event_X = X[event_at_tj]
        event_eta_c = eta_c[event_at_tj]
        event_exp = exp_eta[event_at_tj]

        event_X_sum = np.sum(event_X, axis=0)  # (p,)
        event_eta_c_sum = np.sum(event_eta_c)

        if ties == "breslow" or d_j == 1:
            # Breslow (or single event — both give same result)
            if S0 > 0:
                loglik += event_eta_c_sum - d_j * np.log(S0)
                score += event_X_sum - d_j * S1 / S0
                info_matrix += d_j * (S2 / S0 - np.outer(S1, S1) / S0**2)

        elif ties == "efron":
            # Efron approximation
            death_S0 = np.sum(event_exp)
            death_S1 = event_X.T @ event_exp
            death_S2 = (event_X * event_exp[:, np.newaxis]).T @ event_X

            loglik += event_eta_c_sum

            for s in range(d_j):
                frac = s / d_j
                denom = S0 - frac * death_S0
                if denom <= 0:
                    continue

                s1_adj = S1 - frac * death_S1
                s2_adj = S2 - frac * death_S2

                mean = s1_adj / denom

                loglik -= np.log(denom)
                score -= mean
                info_matrix += s2_adj / denom - np.outer(mean, mean)

            score += event_X_sum

    return loglik, score, info_matrix


def _concordance(
    beta: NDArray,
    time: NDArray,
    event: NDArray,
    X: NDArray,
) -> float:
    """Harrell's concordance statistic (C-statistic).

    C = P(risk_i > risk_j | T_i < T_j, event_i = 1)
    """
    eta = X @ beta
    n = len(time)

    concordant = 0
    discordant = 0
    tied_risk = 0

    for i in range(n):
        if event[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if time[j] <= time[i]:
                continue
            # Subject i had event before subject j's time

            if eta[i] > eta[j]:
                concordant += 1
            elif eta[i] < eta[j]:
                discordant += 1
            else:
                tied_risk += 1

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5

    return (concordant + 0.5 * tied_risk) / total
