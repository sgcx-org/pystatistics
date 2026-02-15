"""
Kaplan-Meier product-limit estimator.

Matches R's survival::survfit(Surv(time, event) ~ 1):
- Product-limit survival estimate: S(t) = ∏(1 - d_j / n_j)
- Greenwood variance: Var(S(t)) = S(t)^2 * Σ(d_j / (n_j * (n_j - d_j)))
- Confidence intervals via log, plain, or log-log transformation

References:
    Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from
        incomplete observations. JASA, 53(282), 457-481.
    R Core Team. survival::survfit.formula
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from pystatistics.survival._common import KMParams


def kaplan_meier_fit(
    time: NDArray,
    event: NDArray,
    conf_level: float,
    conf_type: str,
) -> KMParams:
    """Compute Kaplan-Meier survival curve.

    Parameters
    ----------
    time : NDArray
        (n,) time to event or censoring.
    event : NDArray
        (n,) event indicator (1=event, 0=censored).
    conf_level : float
        Confidence level for CI (e.g. 0.95).
    conf_type : str
        CI type: "log" (default, matches R), "plain", "log-log".

    Returns
    -------
    KMParams
    """
    n_total = len(time)
    n_events_total = int(np.sum(event))

    # Sort by time, with events before censoring at tied times
    # (R sorts events first within ties)
    order = np.lexsort((-event, time))
    t_sorted = time[order]
    e_sorted = event[order]

    # Find unique event times (only times where events occur)
    event_mask = e_sorted == 1
    unique_event_times = np.unique(t_sorted[event_mask])

    if len(unique_event_times) == 0:
        # No events — survival is 1 everywhere
        return KMParams(
            time=np.array([], dtype=np.float64),
            survival=np.array([], dtype=np.float64),
            n_risk=np.array([], dtype=np.float64),
            n_events=np.array([], dtype=np.float64),
            n_censored=np.array([], dtype=np.float64),
            se=np.array([], dtype=np.float64),
            ci_lower=np.array([], dtype=np.float64),
            ci_upper=np.array([], dtype=np.float64),
            conf_level=conf_level,
            conf_type=conf_type,
            n_observations=n_total,
            n_events_total=0,
        )

    m = len(unique_event_times)
    out_time = unique_event_times
    out_n_risk = np.zeros(m, dtype=np.float64)
    out_n_events = np.zeros(m, dtype=np.float64)
    out_n_censored = np.zeros(m, dtype=np.float64)

    # At each unique event time, compute:
    #   n_risk: number alive (not yet failed or censored) just before time t
    #   n_events: number of events at time t
    #   n_censored: number censored in [t_prev, t) (between previous and current event time)
    n_at_risk = n_total

    prev_time_idx = 0  # index into sorted arrays

    for j, t_j in enumerate(unique_event_times):
        # Count censored before this event time (strictly less than t_j)
        # These are observations with t < t_j and event=0
        cens_count = 0
        while prev_time_idx < n_total and t_sorted[prev_time_idx] < t_j:
            if e_sorted[prev_time_idx] == 0:
                cens_count += 1
            n_at_risk -= 1
            prev_time_idx += 1

        out_n_censored[j] = cens_count
        out_n_risk[j] = n_at_risk

        # Count events and censoring at exactly this time
        d_j = 0
        c_j = 0
        while prev_time_idx < n_total and t_sorted[prev_time_idx] == t_j:
            if e_sorted[prev_time_idx] == 1:
                d_j += 1
            else:
                c_j += 1
            prev_time_idx += 1

        out_n_events[j] = d_j
        # Censoring at t_j is counted in the *next* interval by R convention,
        # but we track it here for the risk set update
        n_at_risk -= (d_j + c_j)

        # Add censoring AT this time to the next interval's count
        # (R groups them with the interval after the event time)
        if j < m - 1:
            # Will be added to next interval's censored count
            out_n_censored[j] += 0  # censoring at t_j counted in risk set
        # For the last interval, count remaining censored
    # Count any remaining censored after last event time
    remaining_cens = 0
    while prev_time_idx < n_total:
        if e_sorted[prev_time_idx] == 0:
            remaining_cens += 1
        prev_time_idx += 1

    # Product-limit estimate: S(t) = ∏_{j: t_j <= t} (1 - d_j / n_j)
    hazard_component = out_n_events / out_n_risk
    survival = np.cumprod(1.0 - hazard_component)

    # Greenwood variance: Var(S(t)) = S(t)^2 * Σ(d_j / (n_j * (n_j - d_j)))
    # Avoid division by zero when n_j == d_j (all at risk die)
    denom = out_n_risk * (out_n_risk - out_n_events)
    denom = np.where(denom > 0, denom, np.inf)
    greenwood_sum = np.cumsum(out_n_events / denom)
    variance = survival ** 2 * greenwood_sum
    se = np.sqrt(variance)

    # Confidence intervals
    z = stats.norm.ppf((1.0 + conf_level) / 2.0)

    ci_lower, ci_upper = _compute_ci(
        survival, se, z, conf_type,
    )

    return KMParams(
        time=out_time,
        survival=survival,
        n_risk=out_n_risk,
        n_events=out_n_events,
        n_censored=out_n_censored,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        conf_level=conf_level,
        conf_type=conf_type,
        n_observations=n_total,
        n_events_total=n_events_total,
    )


def _compute_ci(
    survival: NDArray,
    se: NDArray,
    z: float,
    conf_type: str,
) -> tuple[NDArray, NDArray]:
    """Compute CI for survival function.

    Parameters
    ----------
    survival : S(t) values
    se : Greenwood standard errors
    z : normal quantile (e.g. 1.96 for 95%)
    conf_type : "log", "plain", or "log-log"

    Returns
    -------
    (ci_lower, ci_upper) clipped to [0, 1]
    """
    if conf_type == "plain":
        # Plain: S(t) ± z * se
        ci_lower = survival - z * se
        ci_upper = survival + z * se

    elif conf_type == "log":
        # Log transformation (R default): exp(log(S) ± z * se / S)
        # This is the default in R's survfit()
        with np.errstate(divide='ignore', invalid='ignore'):
            log_s = np.log(survival)
            # se of log(S) = se(S) / S
            se_log = se / survival
            ci_lower = np.exp(log_s - z * se_log)
            ci_upper = np.exp(log_s + z * se_log)

    elif conf_type == "log-log":
        # Log-log transformation: exp(-exp(log(-log(S)) ± z * se / (S * log(S))))
        with np.errstate(divide='ignore', invalid='ignore'):
            log_s = np.log(survival)
            log_neg_log_s = np.log(-log_s)
            # se of log(-log(S)) = se(S) / (S * |log(S)|)
            se_loglog = se / (survival * np.abs(log_s))
            ci_lower = np.exp(-np.exp(log_neg_log_s + z * se_loglog))
            ci_upper = np.exp(-np.exp(log_neg_log_s - z * se_loglog))
    else:
        raise ValueError(
            f"Unknown conf_type '{conf_type}'. "
            f"Choose from 'log', 'plain', 'log-log'."
        )

    # Clip to [0, 1]
    ci_lower = np.clip(ci_lower, 0.0, 1.0)
    ci_upper = np.clip(ci_upper, 0.0, 1.0)

    # Handle NaN (from S=0 or S=1 edge cases)
    ci_lower = np.where(np.isnan(ci_lower), 0.0, ci_lower)
    ci_upper = np.where(np.isnan(ci_upper), 1.0, ci_upper)

    return ci_lower, ci_upper
