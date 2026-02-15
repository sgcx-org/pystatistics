"""
Log-rank test (G-rho family) for comparing survival curves across groups.

Matches R's survival::survdiff(Surv(time, event) ~ group, rho=0):
- Standard log-rank test (rho=0): Mantel-Haenszel / Cochran-Mantel
- G-rho family (rho>0): Fleming-Harrington weighted variant
  When rho=1, gives the Peto & Peto modification of the Gehan-Wilcoxon test.

Algorithm:
    1. Sort all observations by time
    2. At each distinct event time t_j:
       - n_ki = number at risk in group k at t_j
       - d_ki = observed events in group k at t_j
       - N_j = total at risk, D_j = total events
       - Expected events in group k: E_ki = n_ki * D_j / N_j
       - Weight w_j = S_hat(t_j-)^rho (KM estimate just before t_j)
    3. Test statistic: chi-squared based on (O - E) with variance

References:
    Harrington, D. P. & Fleming, T. R. (1982). A class of rank test
        procedures for censored survival data. Biometrika, 69(3), 553-566.
    R Core Team. survival::survdiff
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from pystatistics.survival._common import LogRankParams


def logrank_test(
    time: NDArray,
    event: NDArray,
    group: NDArray,
    rho: float = 0.0,
) -> LogRankParams:
    """Compute log-rank test (G-rho family).

    Parameters
    ----------
    time : NDArray
        (n,) time to event or censoring.
    event : NDArray
        (n,) event indicator (1=event, 0=censored).
    group : NDArray
        (n,) group labels.
    rho : float
        G-rho weight parameter: rho=0 is standard log-rank,
        rho=1 is Peto & Peto / Gehan-Wilcoxon.

    Returns
    -------
    LogRankParams
    """
    n = len(time)

    # Identify groups
    unique_groups = np.unique(group)
    n_groups = len(unique_groups)

    if n_groups < 2:
        raise ValueError(
            f"Need at least 2 groups for log-rank test, got {n_groups}"
        )

    # Map group labels to indices 0..n_groups-1
    group_idx = np.zeros(n, dtype=np.intp)
    for k, g in enumerate(unique_groups):
        group_idx[group == g] = k

    # Sort by time (events before censoring at ties, matching R)
    order = np.lexsort((-event, time))
    t_sorted = time[order]
    e_sorted = event[order]
    g_sorted = group_idx[order]

    # Find distinct event times
    event_mask = e_sorted == 1
    unique_event_times = np.unique(t_sorted[event_mask])

    if len(unique_event_times) == 0:
        # No events — test is meaningless
        return LogRankParams(
            statistic=0.0,
            df=n_groups - 1,
            p_value=1.0,
            n_groups=n_groups,
            observed=np.zeros(n_groups, dtype=np.float64),
            expected=np.zeros(n_groups, dtype=np.float64),
            n_per_group=np.array(
                [np.sum(group == g) for g in unique_groups],
                dtype=np.float64,
            ),
            rho=rho,
            group_labels=unique_groups,
        )

    # For G-rho family, we need the pooled Kaplan-Meier estimate
    # We compute S_hat(t-) at each event time (survival just before)
    # For rho=0, the weight is 1 everywhere, so we can skip this.

    # --- Compute at each event time: n_risk per group, n_events per group ---
    # Track who is still at risk
    # We walk through sorted observations, tracking risk sets

    m = len(unique_event_times)

    # Arrays: per event time × per group
    d_kg = np.zeros((m, n_groups), dtype=np.float64)  # events per group
    n_kg = np.zeros((m, n_groups), dtype=np.float64)  # at risk per group

    # Total at risk per group (decremented as we go)
    at_risk = np.zeros(n_groups, dtype=np.float64)
    for k in range(n_groups):
        at_risk[k] = np.sum(g_sorted == k)

    ptr = 0  # pointer into sorted arrays

    for j, t_j in enumerate(unique_event_times):
        # Remove subjects with time < t_j from risk sets
        while ptr < n and t_sorted[ptr] < t_j:
            at_risk[g_sorted[ptr]] -= 1
            ptr += 1

        # Record current risk set
        n_kg[j] = at_risk.copy()

        # Count events and censored at exactly t_j
        while ptr < n and t_sorted[ptr] == t_j:
            if e_sorted[ptr] == 1:
                d_kg[j, g_sorted[ptr]] += 1
            at_risk[g_sorted[ptr]] -= 1
            ptr += 1

    # Totals per event time
    D_j = d_kg.sum(axis=1)     # (m,) total events at each time
    N_j = n_kg.sum(axis=1)     # (m,) total at risk at each time

    # --- Compute weights for G-rho family ---
    if rho == 0.0:
        weights = np.ones(m, dtype=np.float64)
    else:
        # S_hat(t-) using the pooled Kaplan-Meier estimate
        # S(t_j-) = product of (1 - D_i/N_i) for all i < j
        hazard = D_j / np.maximum(N_j, 1.0)
        surv_components = 1.0 - hazard
        # S(t_0-) = 1 (before first event)
        # S(t_j-) = cumulative product of survival up to j-1
        cum_surv = np.cumprod(surv_components)
        # S(t_j-) = shifted: S before event time j
        s_before = np.ones(m, dtype=np.float64)
        s_before[1:] = cum_surv[:-1]
        weights = s_before ** rho

    # --- Expected events per group: E_k = Σ_j w_j * n_kj * D_j / N_j ---
    # Only count times where N_j > 0
    valid = N_j > 0
    expected_per_group = np.zeros(n_groups, dtype=np.float64)
    observed_per_group = np.zeros(n_groups, dtype=np.float64)

    for k in range(n_groups):
        observed_per_group[k] = np.sum(weights[valid] * d_kg[valid, k])
        expected_per_group[k] = np.sum(
            weights[valid] * n_kg[valid, k] * D_j[valid] / N_j[valid]
        )

    # --- Variance-covariance matrix ---
    # V_kl = Σ_j w_j^2 * D_j * (N_j - D_j) / (N_j^2 * (N_j - 1))
    #          * (δ_kl * n_kj / N_j - n_kj * n_lj / N_j^2) * N_j
    # Simplified: standard Mantel-Haenszel variance
    # V_kl = Σ_j w_j^2 * n_kj * (δ_kl * N_j - n_lj) * D_j * (N_j - D_j)
    #                                            / (N_j^2 * (N_j - 1))

    V = np.zeros((n_groups, n_groups), dtype=np.float64)

    for j in range(m):
        if not valid[j] or N_j[j] <= 1:
            continue

        w2 = weights[j] ** 2
        factor = w2 * D_j[j] * (N_j[j] - D_j[j]) / (N_j[j] ** 2 * (N_j[j] - 1))

        for k in range(n_groups):
            for l in range(n_groups):
                if k == l:
                    V[k, l] += factor * n_kg[j, k] * (N_j[j] - n_kg[j, k])
                else:
                    V[k, l] -= factor * n_kg[j, k] * n_kg[j, l]

    # --- Chi-squared statistic ---
    # Use the first (n_groups-1) groups (drop last for non-singularity)
    # The last row/column is linearly dependent since Σ(O_k - E_k) = 0
    df = n_groups - 1

    if df == 1:
        # For 2 groups, simple formula: (O1 - E1)^2 / V11
        oe_diff = observed_per_group[0] - expected_per_group[0]
        if V[0, 0] > 0:
            statistic = oe_diff ** 2 / V[0, 0]
        else:
            statistic = 0.0
    else:
        # General case: (O - E)_{1:K-1}^T V^{-1}_{1:K-1, 1:K-1} (O - E)_{1:K-1}
        oe_diff = (observed_per_group - expected_per_group)[:df]
        V_sub = V[:df, :df]
        try:
            V_inv = np.linalg.inv(V_sub)
            statistic = float(oe_diff @ V_inv @ oe_diff)
        except np.linalg.LinAlgError:
            statistic = 0.0

    # p-value from chi-squared distribution
    p_value = float(1.0 - stats.chi2.cdf(statistic, df))

    n_per_group = np.array(
        [np.sum(group == g) for g in unique_groups],
        dtype=np.float64,
    )

    # R reports weighted O and E (same values used in the test statistic)
    obs_report = observed_per_group
    exp_report = expected_per_group

    return LogRankParams(
        statistic=statistic,
        df=df,
        p_value=p_value,
        n_groups=n_groups,
        observed=obs_report,
        expected=exp_report,
        n_per_group=n_per_group,
        rho=rho,
        group_labels=unique_groups,
    )
