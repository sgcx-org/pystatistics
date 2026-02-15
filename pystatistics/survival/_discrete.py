"""
Discrete-time survival model via person-period logistic regression.

Converts time-to-event data into a person-period (long-format) dataset
and fits a logistic regression model. This approach allows GPU acceleration
since it reduces survival analysis to a standard GLM problem.

Algorithm:
    1. Define time intervals: either from unique event times or user-specified.
    2. Person-period expansion: Each subject contributes one row per interval
       they are alive at the start of, with y=1 in the event interval (if any),
       y=0 otherwise.
    3. Construct design matrix: Interval indicators (for baseline hazard) +
       original covariates.
    4. Fit logistic regression: fit(X_pp, y_pp, family='binomial', backend=backend)
    5. Extract results: covariate hazard ratios, baseline discrete hazard.

References:
    Singer, J. D. & Willett, J. B. (1993). It's about time: Using
        discrete-time survival analysis to study duration and the
        timing of events. Journal of Educational Statistics, 18(2), 155-195.
    Allison, P. D. (1982). Discrete-time methods for the analysis
        of event histories. Sociological Methodology, 13, 61-98.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pystatistics.survival._common import DiscreteTimeParams


def discrete_time_fit(
    time: NDArray,
    event: NDArray,
    X: NDArray,
    intervals: NDArray | None = None,
    backend: Literal["auto", "cpu", "gpu"] = "auto",
) -> DiscreteTimeParams:
    """Fit discrete-time survival model.

    Parameters
    ----------
    time : NDArray
        (n,) time to event or censoring.
    event : NDArray
        (n,) event indicator (1=event, 0=censored).
    X : NDArray
        (n, p) covariate matrix (NO intercept).
    intervals : NDArray or None
        Time interval boundaries. If None, uses unique event times.
    backend : str
        Backend for logistic regression: "auto", "cpu", or "gpu".

    Returns
    -------
    DiscreteTimeParams
    """
    from pystatistics.regression.solvers import fit as regression_fit

    n, p = X.shape
    n_events_total = int(np.sum(event))

    # --- Define intervals ---
    if intervals is not None:
        interval_bounds = np.sort(np.asarray(intervals, dtype=np.float64))
    else:
        # Use unique event times as interval boundaries
        event_mask = event == 1
        interval_bounds = np.unique(time[event_mask])

    if len(interval_bounds) == 0:
        # No events — return degenerate result
        return DiscreteTimeParams(
            coefficients=np.zeros(p, dtype=np.float64),
            standard_errors=np.full(p, np.inf),
            z_statistics=np.zeros(p, dtype=np.float64),
            p_values=np.ones(p, dtype=np.float64),
            hazard_ratios=np.ones(p, dtype=np.float64),
            baseline_hazard=np.array([], dtype=np.float64),
            interval_labels=np.array([], dtype=np.float64),
            person_period_n=0,
            n_intervals=0,
            n_observations=n,
            n_events=0,
            glm_deviance=0.0,
            glm_aic=0.0,
        )

    n_intervals = len(interval_bounds)

    # --- Person-period expansion ---
    # For each subject, create one row per interval they are alive at the start of.
    # y=1 if the event occurs in that interval, y=0 otherwise.
    rows_y = []
    rows_X_cov = []
    rows_interval_idx = []

    for i in range(n):
        for j, t_bound in enumerate(interval_bounds):
            # Subject i is at risk in interval j if their time >= t_bound
            # (they are alive at the start of this interval)
            if j == 0:
                # First interval: everyone is at risk
                at_risk = True
            else:
                # At risk if their observed time >= this interval's start
                # (they survived past the previous interval)
                at_risk = time[i] >= t_bound

            if not at_risk:
                break

            # Response: 1 if event happens at this interval's time
            if event[i] == 1 and time[i] == t_bound:
                y_ij = 1.0
            elif event[i] == 1 and j < n_intervals - 1 and time[i] < interval_bounds[j + 1] and time[i] >= t_bound:
                y_ij = 1.0
            elif event[i] == 1 and j == n_intervals - 1 and time[i] >= t_bound:
                y_ij = 1.0
            else:
                y_ij = 0.0

            rows_y.append(y_ij)
            rows_X_cov.append(X[i])
            rows_interval_idx.append(j)

            # If event occurred, stop contributing rows after this interval
            if y_ij == 1.0:
                break

            # If censored and time falls within this interval, stop
            if event[i] == 0 and j < n_intervals - 1 and time[i] < interval_bounds[j + 1]:
                break
            if event[i] == 0 and j == n_intervals - 1:
                break

    y_pp = np.array(rows_y, dtype=np.float64)
    X_cov_pp = np.array(rows_X_cov, dtype=np.float64)
    interval_idx_pp = np.array(rows_interval_idx, dtype=np.intp)
    person_period_n = len(y_pp)

    # --- Construct design matrix with interval indicators ---
    # One-hot encoding for intervals (no intercept needed, interval dummies serve as intercepts)
    X_interval = np.zeros((person_period_n, n_intervals), dtype=np.float64)
    for k in range(person_period_n):
        X_interval[k, interval_idx_pp[k]] = 1.0

    # Full design matrix: [interval_indicators | covariates]
    X_pp = np.column_stack([X_interval, X_cov_pp])

    # --- Fit logistic regression ---
    glm_result = regression_fit(
        X_pp, y_pp,
        family='binomial',
        backend=backend,
    )

    # --- Extract results ---
    all_coefs = glm_result.coefficients

    # Interval coefficients (baseline log-odds of hazard)
    interval_coefs = all_coefs[:n_intervals]
    # Covariate coefficients
    cov_coefs = all_coefs[n_intervals:]

    # Standard errors
    all_se = glm_result.standard_errors
    cov_se = all_se[n_intervals:]

    # z-statistics and p-values for covariates
    cov_z = np.where(cov_se > 0, cov_coefs / cov_se, 0.0)
    from scipy import stats
    cov_p = 2.0 * stats.norm.sf(np.abs(cov_z))

    # Baseline discrete hazard: h_0(t) = expit(interval_coef)
    from scipy.special import expit
    baseline_hazard = expit(interval_coefs)

    # Hazard ratios for covariates: exp(coef)
    hazard_ratios = np.exp(cov_coefs)

    return DiscreteTimeParams(
        coefficients=cov_coefs,
        standard_errors=cov_se,
        z_statistics=cov_z,
        p_values=cov_p,
        hazard_ratios=hazard_ratios,
        baseline_hazard=baseline_hazard,
        interval_labels=interval_bounds,
        person_period_n=person_period_n,
        n_intervals=n_intervals,
        n_observations=n,
        n_events=n_events_total,
        glm_deviance=glm_result.deviance,
        glm_aic=glm_result.aic,
    )
