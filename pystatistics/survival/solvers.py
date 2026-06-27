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

from pystatistics.core.exceptions import (
    NotImplementedFeatureError, ValidationError,
)

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
        raise ValidationError(
            f"conf_level must be in (0, 1), got {conf_level}"
        )

    if conf_type not in ("log", "plain", "log-log"):
        raise ValidationError(
            f"conf_type must be 'log', 'plain', or 'log-log', "
            f"got '{conf_type}'"
        )

    if design.strata is not None:
        raise NotImplementedFeatureError(
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
        raise ValidationError(
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

    # Non-fatal transparency warnings: the log-rank chi-square is an
    # asymptotic approximation that degrades when any group's expected
    # event count is small, and a group contributing no events makes its
    # comparison degenerate.
    warnings_list = []
    for i in range(params.n_groups):
        raw_label = params.group_labels[i]
        # Render numpy scalars (np.int64(2)) as plain Python values (2).
        label = raw_label.item() if hasattr(raw_label, "item") else raw_label
        if params.expected[i] < 5:
            warnings_list.append(
                f"Group {label!r} has a small expected event count "
                f"({params.expected[i]:.2f} < 5); the chi-square "
                f"approximation may be unreliable"
            )
        if params.observed[i] == 0:
            warnings_list.append(
                f"Group {label!r} has zero observed events; its "
                f"contribution to the test is degenerate"
            )

    result = Result(
        params=params,
        info={"method": "Log-rank test", "rho": rho},
        timing=timer.result(),
        backend_name="cpu_logrank",
        warnings=tuple(warnings_list),
    )

    return LogRankSolution(_result=result)


def coxph(
    time,
    event,
    X,
    *,
    terms=None,
    names: list[str] | None = None,
    strata=None,
    ties: Literal["efron", "breslow"] = "efron",
    tol: float = 1e-9,
    max_iter: int = 20,
    conf_level: float = 0.95,
) -> CoxSolution:
    """Cox proportional hazards model.

    CPU only — no backend parameter. Matches R's survival::coxph().

    Parameters
    ----------
    time : array-like
        Time to event or censoring.
    event : array-like
        Event indicator (1=event, 0=censored).
    X : array-like or DataSource
        Covariate matrix (n, p). No intercept — Cox model has no intercept.
        When ``terms`` is given, ``X`` is instead the data source (a
        DataSource or column mapping) from which the term spec pulls columns.
    terms : sequence or None
        Structured term spec for categorical predictors and interactions
        (bare column names, C(name, ref=...), or tuples of those). When
        given, the covariate matrix is built from ``X`` (the source) with no
        intercept, and the expanded column labels are used automatically.
        Mutually exclusive with ``names``.
    strata : array-like or None
        Strata labels for stratified Cox (not yet implemented).
    ties : str
        Method for handling tied event times: "efron" (default) or "breslow".
    tol : float
        Convergence tolerance for Newton-Raphson.
    max_iter : int
        Maximum Newton-Raphson iterations.
    conf_level : float
        Confidence level for ``.conf_int`` (default 0.95). Wald intervals on the
        coefficient (log-hazard-ratio) scale; ``exp(.conf_int)`` gives hazard-
        ratio intervals.

    Returns
    -------
    CoxSolution
    """
    if terms is not None:
        if names is not None:
            raise ValidationError("Pass either terms or names, not both")
        from pystatistics.regression.terms import build_terms_design
        X, names = build_terms_design(X, terms, intercept=False)

    design = SurvivalDesign.for_survival(time, event, X)

    if design.X is None:
        raise ValidationError("X (covariates) is required for coxph()")

    if ties not in ("efron", "breslow"):
        raise ValidationError(
            f"ties must be 'efron' or 'breslow', got '{ties}'"
        )

    if conf_level <= 0 or conf_level >= 1:
        raise ValidationError(
            f"conf_level must be in (0, 1), got {conf_level}"
        )

    if strata is not None:
        raise NotImplementedFeatureError(
            "Stratified Cox PH is not yet implemented"
        )

    timer = Timer()
    timer.start()

    params = cox_fit(
        design.time, design.event, design.X,
        ties=ties,
        tol=tol,
        max_iter=max_iter,
        conf_level=conf_level,
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

    # Resolve names
    resolved_names = None
    if names is not None:
        p = design.X.shape[1]
        if len(names) != p:
            raise ValidationError(
                f"names must have {p} elements to match X with "
                f"{p} columns, got {len(names)}"
            )
        resolved_names = tuple(names)

    return CoxSolution(_result=result, _names=resolved_names)


def discrete_time(
    time,
    event,
    X,
    *,
    names: list[str] | None = None,
    intervals=None,
    backend: Literal["auto", "cpu", "gpu"] | None = None,
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
    backend : str or None
        Backend for the person-period logistic regression. Default
        None → 'cpu' (R-reference path). Explicit values: "cpu", "gpu",
        or "auto" to prefer GPU when available.

    Returns
    -------
    DiscreteTimeSolution
    """
    if backend is None:
        backend = "cpu"

    design = SurvivalDesign.for_survival(time, event, X)

    if design.X is None:
        raise ValidationError("X (covariates) is required for discrete_time()")

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

    # Resolve names
    resolved_names = None
    if names is not None:
        p = design.X.shape[1]
        if len(names) != p:
            raise ValidationError(
                f"names must have {p} elements to match X with "
                f"{p} columns, got {len(names)}"
            )
        resolved_names = tuple(names)

    return DiscreteTimeSolution(_result=result, _names=resolved_names)
