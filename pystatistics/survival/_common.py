"""
Parameter payloads for survival analysis results.

Each dataclass is a frozen payload carried inside a Result[P] envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class KMParams:
    """Kaplan-Meier survival curve parameters.

    Matches the output of R's survival::survfit().
    """

    time: NDArray                # (m,) — unique event times
    survival: NDArray            # (m,) — S(t) at each event time
    n_risk: NDArray              # (m,) — number at risk just before each time
    n_events: NDArray            # (m,) — events at each time
    n_censored: NDArray          # (m,) — censored between this and next time
    se: NDArray                  # (m,) — Greenwood standard error
    ci_lower: NDArray            # (m,) — lower CI for S(t)
    ci_upper: NDArray            # (m,) — upper CI for S(t)
    conf_level: float            # confidence level (e.g. 0.95)
    conf_type: str               # CI type: "log" (default), "plain", "log-log"
    n_observations: int          # total n
    n_events_total: int          # total events


@dataclass(frozen=True)
class LogRankParams:
    """Log-rank test parameters.

    Matches the output of R's survival::survdiff().
    """

    statistic: float             # chi-squared statistic
    df: int                      # degrees of freedom (n_groups - 1)
    p_value: float
    n_groups: int
    observed: NDArray            # (n_groups,) — observed events per group
    expected: NDArray            # (n_groups,) — expected events per group
    n_per_group: NDArray         # (n_groups,) — subjects per group
    rho: float                   # weight parameter (0=log-rank, 1=Peto-Peto)
    group_labels: NDArray        # unique group labels


@dataclass(frozen=True)
class CoxParams:
    """Cox proportional hazards model parameters.

    Matches the output of R's survival::coxph().
    """

    coefficients: NDArray        # (p,) — log hazard ratios
    hazard_ratios: NDArray       # (p,) — exp(coef)
    standard_errors: NDArray     # (p,) — from observed information matrix
    z_statistics: NDArray        # (p,) — coef / se
    p_values: NDArray            # (p,) — two-sided Wald test
    loglik: tuple[float, float]  # (null log-lik, model log-lik)
    concordance: float           # Harrell's C-statistic
    n_events: int
    n_observations: int
    n_iter: int                  # Newton-Raphson iterations
    converged: bool
    ties: str                    # "efron" or "breslow"


@dataclass(frozen=True)
class DiscreteTimeParams:
    """Discrete-time survival model parameters.

    Fitted via person-period logistic regression.
    """

    coefficients: NDArray        # (p,) — covariate log odds ratios
    standard_errors: NDArray
    z_statistics: NDArray
    p_values: NDArray
    hazard_ratios: NDArray       # exp(coef) — discrete-time hazard ratios
    baseline_hazard: NDArray     # (T,) — baseline discrete hazard per interval
    interval_labels: NDArray     # (T,) — interval endpoints
    person_period_n: int         # total rows in expanded dataset
    n_intervals: int
    n_observations: int
    n_events: int
    glm_deviance: float          # deviance from logistic fit
    glm_aic: float               # AIC from logistic fit
