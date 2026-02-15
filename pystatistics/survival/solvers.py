"""
Public API for survival analysis.

    kaplan_meier(time, event) → KMSolution
    survdiff(time, event, group) → LogRankSolution
    coxph(time, event, X) → CoxSolution          # CPU only
    discrete_time(time, event, X) → DiscreteTimeSolution  # GPU accelerated

Each function validates inputs, creates a SurvivalDesign, dispatches to
the appropriate backend, and wraps the Result in a Solution.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.survival.design import SurvivalDesign
from pystatistics.survival._common import CoxParams, DiscreteTimeParams, KMParams, LogRankParams
from pystatistics.survival._km import kaplan_meier_fit
from pystatistics.survival._logrank import logrank_test
from pystatistics.survival._cox import cox_fit
from pystatistics.survival._discrete import discrete_time_fit
from pystatistics.survival.solution import (
    CoxSolution, DiscreteTimeSolution, KMSolution, LogRankSolution,
)


def kaplan_meier(
    time,
    event,
    *,
    strata=None,
    conf_level: float = 0.95,
    conf_type: Literal["log", "plain", "log-log"] = "log",
) -> KMSolution:
    """Kaplan-Meier survival curve estimation.

    Matches R's survival::survfit(Surv(time, event) ~ 1).

    Parameters
    ----------
    time : array-like
        Time to event or censoring.
    event : array-like
        Event indicator (1=event, 0=censored).
    strata : array-like or None
        Strata labels for stratified KM (not yet implemented).
    conf_level : float
        Confidence level for CI (default 0.95).
    conf_type : str
        CI transformation: "log" (R default), "plain", "log-log".

    Returns
    -------
    KMSolution
    """
    design = SurvivalDesign.for_survival(time, event, strata=strata)

    if conf_level <= 0 or conf_level >= 1:
        raise ValueError(
            f"conf_level must be in (0, 1), got {conf_level}"
        )

    if conf_type not in ("log", "plain", "log-log"):
        raise ValueError(
            f"conf_type must be 'log', 'plain', or 'log-log', "
            f"got '{conf_type}'"
        )

    if design.strata is not None:
        raise NotImplementedError(
            "Stratified Kaplan-Meier is not yet implemented"
        )

    timer = Timer()
    timer.start()

    params = kaplan_meier_fit(
        design.time, design.event,
        conf_level=conf_level,
        conf_type=conf_type,
    )

    timer.stop()

    result = Result(
        params=params,
        info={"method": "Kaplan-Meier"},
        timing=timer.result(),
        backend_name="cpu_km",
        warnings=(),
    )

    return KMSolution(_result=result)


def survdiff(
    time,
    event,
    group,
    *,
    rho: float = 0.0,
) -> LogRankSolution:
    """Log-rank test (and G-rho family).

    Matches R's survival::survdiff().

    Parameters
    ----------
    time : array-like
        Time to event or censoring.
    event : array-like
        Event indicator (1=event, 0=censored).
    group : array-like
        Group labels (e.g. treatment vs control).
    rho : float
        G-rho weight parameter. rho=0 (default) gives the standard
        log-rank test. rho=1 gives Peto & Peto / Gehan-Wilcoxon.

    Returns
    -------
    LogRankSolution
    """
    design = SurvivalDesign.for_survival(time, event)
    group = np.asarray(group).ravel()

    if len(group) != design.n:
        raise ValueError(
            f"group must have {design.n} elements to match time, "
            f"got {len(group)}"
        )

    timer = Timer()
    timer.start()

    params = logrank_test(
        design.time, design.event, group,
        rho=rho,
    )

    timer.stop()

    result = Result(
        params=params,
        info={"method": "Log-rank test", "rho": rho},
        timing=timer.result(),
        backend_name="cpu_logrank",
        warnings=(),
    )

    return LogRankSolution(_result=result)


def coxph(
    time,
    event,
    X,
    *,
    strata=None,
    ties: Literal["efron", "breslow"] = "efron",
    tol: float = 1e-9,
    max_iter: int = 20,
) -> CoxSolution:
    """Cox proportional hazards model.

    CPU only — no backend parameter. Matches R's survival::coxph().

    Parameters
    ----------
    time : array-like
        Time to event or censoring.
    event : array-like
        Event indicator (1=event, 0=censored).
    X : array-like
        Covariate matrix (n, p). No intercept — Cox model has no intercept.
    strata : array-like or None
        Strata labels for stratified Cox (not yet implemented).
    ties : str
        Method for handling tied event times: "efron" (default) or "breslow".
    tol : float
        Convergence tolerance for Newton-Raphson.
    max_iter : int
        Maximum Newton-Raphson iterations.

    Returns
    -------
    CoxSolution
    """
    design = SurvivalDesign.for_survival(time, event, X)

    if design.X is None:
        raise ValueError("X (covariates) is required for coxph()")

    if ties not in ("efron", "breslow"):
        raise ValueError(
            f"ties must be 'efron' or 'breslow', got '{ties}'"
        )

    if strata is not None:
        raise NotImplementedError(
            "Stratified Cox PH is not yet implemented"
        )

    timer = Timer()
    timer.start()

    params = cox_fit(
        design.time, design.event, design.X,
        ties=ties,
        tol=tol,
        max_iter=max_iter,
    )

    timer.stop()

    warnings_list = []
    if not params.converged:
        warnings_list.append(
            f"Newton-Raphson did not converge in {max_iter} iterations"
        )

    result = Result(
        params=params,
        info={
            "method": "Cox PH",
            "ties": ties,
            "n_iter": params.n_iter,
        },
        timing=timer.result(),
        backend_name="cpu_cox",
        warnings=tuple(warnings_list),
    )

    return CoxSolution(_result=result)


def discrete_time(
    time,
    event,
    X,
    *,
    intervals=None,
    backend: Literal["auto", "cpu", "gpu"] = "auto",
) -> DiscreteTimeSolution:
    """Discrete-time survival via person-period logistic regression.

    GPU-accelerated — delegates to regression.fit(family='binomial').

    Parameters
    ----------
    time : array-like
        Time to event or censoring.
    event : array-like
        Event indicator (1=event, 0=censored).
    X : array-like
        Covariate matrix (n, p). No intercept — interval dummies
        serve as the intercept.
    intervals : array-like or None
        Time interval boundaries. If None, uses unique event times.
    backend : str
        Backend for logistic regression: "auto", "cpu", or "gpu".
        "auto" selects GPU if available, else CPU.

    Returns
    -------
    DiscreteTimeSolution
    """
    design = SurvivalDesign.for_survival(time, event, X)

    if design.X is None:
        raise ValueError("X (covariates) is required for discrete_time()")

    intervals_arr = None
    if intervals is not None:
        intervals_arr = np.asarray(intervals, dtype=np.float64)

    timer = Timer()
    timer.start()

    params = discrete_time_fit(
        design.time, design.event, design.X,
        intervals=intervals_arr,
        backend=backend,
    )

    timer.stop()

    result = Result(
        params=params,
        info={
            "method": "Discrete-time survival",
            "backend": backend,
            "person_period_n": params.person_period_n,
        },
        timing=timer.result(),
        backend_name=f"{'gpu' if backend == 'gpu' else 'cpu'}_discrete",
        warnings=(),
    )

    return DiscreteTimeSolution(_result=result)
