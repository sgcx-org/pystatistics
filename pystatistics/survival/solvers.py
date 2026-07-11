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

from pystatistics.core.exceptions import ValidationError

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.survival.design import SurvivalDesign
from pystatistics.survival._common import CoxParams, DiscreteTimeParams, KMParams, LogRankParams
from pystatistics.survival._km import kaplan_meier_fit
from pystatistics.survival._km_strata import (
    StratifiedKMSolution, stratified_km_curves,
)
from pystatistics.survival._logrank import logrank_test
from pystatistics.survival._cox import cox_fit
from pystatistics.survival._cox_strata import cox_fit_stratified
from pystatistics.survival._cox_robust import cox_robust_variance
from pystatistics.survival._cox_zph import CoxZphSolution, cox_zph_compute
from pystatistics.survival._discrete import discrete_time_fit
from pystatistics.survival.solution import (
    CoxSolution, DiscreteTimeSolution, KMSolution, LogRankSolution,
)


def kaplan_meier(
    time,
    event,
    *,
    strata=None,
    entry=None,
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
        Stratum label per observation. When given, returns a
        ``StratifiedKMSolution`` holding one product-limit curve per stratum
        (matching ``survfit(Surv(time, event) ~ g)``).
    entry : array-like or None
        Delayed-entry (left-truncation) time per subject: the subject is at
        risk on ``(entry, time]``, so early risk sets exclude subjects who
        have not yet entered. Matches ``survfit(Surv(entry, time, event))``.
        Each entry must be strictly less than the corresponding ``time``.
    conf_level : float
        Confidence level for CI (default 0.95).
    conf_type : str
        CI transformation: "log" (R default), "plain", "log-log".

    Returns
    -------
    KMSolution
    """
    design = SurvivalDesign.for_survival(time, event, strata=strata,
                                         entry=entry)

    if conf_level <= 0 or conf_level >= 1:
        raise ValidationError(
            f"conf_level must be in (0, 1), got {conf_level}"
        )

    if conf_type not in ("log", "plain", "log-log"):
        raise ValidationError(
            f"conf_type must be 'log', 'plain', or 'log-log', "
            f"got '{conf_type}'"
        )

    timer = Timer()
    timer.start()

    if design.strata is not None:
        curves = stratified_km_curves(
            design.time, design.event, design.strata,
            conf_level=conf_level, conf_type=conf_type,
            entry=design.entry,
        )
        timer.stop()
        wrapped = {}
        for label, params in curves:
            result = Result(
                params=params,
                info={"method": "Kaplan-Meier", "stratum": label},
                timing=None,
                backend_name="cpu_km",
                warnings=(),
            )
            wrapped[label] = KMSolution(_result=result)
        return StratifiedKMSolution(_curves=wrapped, _timing=timer.result())

    params = kaplan_meier_fit(
        design.time, design.event,
        conf_level=conf_level,
        conf_type=conf_type,
        entry=design.entry,
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


def _apply_robust_se(params, design, ties, cluster_arr):
    """Return a copy of ``params`` whose SE/z/p are the sandwich estimator."""
    import dataclasses

    rv = cox_robust_variance(
        design.time, design.event, design.X,
        beta=params.coefficients,
        ties=ties,
        strata=design.strata,
        entry=design.entry,
        cluster=cluster_arr,
    )
    robust_se = rv["robust_se"]
    z = np.where(robust_se > 0, params.coefficients / robust_se, 0.0)
    from scipy import stats as _stats
    p_values = 2.0 * _stats.norm.sf(np.abs(z))
    return dataclasses.replace(
        params,
        standard_errors=robust_se,
        z_values=z,
        p_values=p_values,
        robust=True,
        naive_standard_errors=params.standard_errors,
    )


def coxph(
    time,
    event,
    X,
    *,
    terms=None,
    names: list[str] | None = None,
    strata=None,
    start=None,
    robust: bool = False,
    cluster=None,
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
        Time to event or censoring. With ``start``, this is the risk-interval
        exit (R's ``stop``).
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
        Stratum label per observation. When given, fits a stratified Cox model:
        one shared coefficient vector, a separate baseline hazard (separate risk
        sets) per stratum. Matches ``coxph(Surv(t, e) ~ x + strata(g))``.
    start : array-like or None
        Counting-process start time per ROW: the row is at risk on
        ``(start, time]``. A subject may span several rows with different
        covariate values (time-varying covariates), or a single row with a
        delayed entry (left truncation). Matches
        ``coxph(Surv(start, stop, event) ~ x)`` with ``time`` as the stop.
        Each start must be strictly less than its ``time`` (R NA-drops such
        rows with a warning; PyStatistics refuses loudly instead).
    robust : bool
        If True, report the Lin-Wei sandwich (robust / Huber-White) standard
        errors instead of the model-based ones; ``.z_values`` / ``.p_values`` /
        ``.conf_int`` then use the robust SE, and ``.naive_standard_errors``
        keeps the model-based SE. Matches ``coxph(..., robust=TRUE)``. (Distinct
        from ``timeseries``'s LOESS-robustness ``robust`` — CONVENTIONS A9: a
        robust *variance estimator* here, not outlier downweighting.)
    cluster : array-like or None
        Cluster label per row for grouped-robust SE (correlated rows of one
        subject/cluster count as one independent unit). Implies ``robust=True``.
        Matches ``coxph(..., cluster=id)``.
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
        X, names, _assign, _term_names = build_terms_design(
            X, terms, intercept=False)

    design = SurvivalDesign.for_survival(time, event, X, strata=strata,
                                         entry=start)

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

    cluster_arr = None
    if cluster is not None:
        cluster_arr = np.asarray(cluster).ravel()
        if len(cluster_arr) != design.n:
            raise ValidationError(
                f"cluster must have {design.n} elements to match time, "
                f"got {len(cluster_arr)}"
            )
        robust = True  # a cluster grouping implies the sandwich estimator

    if robust and design.entry is not None and cluster_arr is None:
        # Counting-process rows of one subject are correlated; a sandwich that
        # treats each row as independent is silently wrong. R refuses too
        # ("one of cluster or id is needed").
        raise ValidationError(
            "robust=True with counting-process data (start=) requires "
            "cluster=: rows belonging to one subject must be grouped for "
            "the sandwich variance"
        )

    timer = Timer()
    timer.start()

    if design.strata is not None:
        params = cox_fit_stratified(
            design.time, design.event, design.X, design.strata,
            ties=ties,
            tol=tol,
            max_iter=max_iter,
            conf_level=conf_level,
            entry=design.entry,
        )
    else:
        params = cox_fit(
            design.time, design.event, design.X,
            ties=ties,
            tol=tol,
            max_iter=max_iter,
            conf_level=conf_level,
            entry=design.entry,
        )

    if robust and params.n_events > 0:
        params = _apply_robust_se(params, design, ties, cluster_arr)

    timer.stop()

    # Resolve names (needed for a descriptive infinite-coefficient warning)
    resolved_names = None
    if names is not None:
        p = design.X.shape[1]
        if len(names) != p:
            raise ValidationError(
                f"names must have {p} elements to match X with "
                f"{p} columns, got {len(names)}"
            )
        resolved_names = tuple(names)

    warnings_list = []
    if not params.converged:
        warnings_list.append(
            f"Newton-Raphson did not converge in {max_iter} iterations"
        )
    elif params.infinite_coefs:
        # Matches R coxph's "Loglik converged before variable ...; coefficient
        # may be infinite" — the fit plateaued while a coefficient runs to
        # +/- infinity (monotone likelihood / separation).
        labels = [
            resolved_names[i] if resolved_names is not None else f"x{i}"
            for i in params.infinite_coefs
        ]
        warnings_list.append(
            f"Loglik converged before variable(s) {', '.join(labels)}; "
            f"coefficient may be infinite (monotone likelihood / separation)"
        )

    result = Result(
        params=params,
        info={
            "method": "Stratified Cox PH" if params.n_strata > 1 else "Cox PH",
            "ties": ties,
            "n_iter": params.n_iter,
            "n_strata": params.n_strata,
        },
        timing=timer.result(),
        backend_name="cpu_cox",
        warnings=tuple(warnings_list),
    )

    return CoxSolution(_result=result, _names=resolved_names, _design=design)


def discrete_time(
    time,
    event,
    X,
    *,
    names: list[str] | None = None,
    intervals=None,
    backend: Literal["auto", "cpu", "gpu", "gpu_fp64"] | None = None,
    conf_level: float = 0.95,
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
        Backend for the person-period logistic regression, forwarded to
        ``regression.fit``. Default None → 'cpu' (the R-reference path).
        Explicit values: "cpu" (float64), "gpu" (float32, CUDA or Apple
        Silicon), "gpu_fp64" (float64, CUDA only — the exact double-precision
        GPU path), or "auto" (GPU-fp32 if CUDA present, else CPU).
    conf_level : float
        Confidence level for ``.conf_int`` (default 0.95). Wald intervals on the
        covariate coefficient scale; ``exp(.conf_int)`` gives discrete-time
        hazard-ratio intervals.

    Returns
    -------
    DiscreteTimeSolution
    """
    if backend is None:
        backend = "cpu"

    if conf_level <= 0 or conf_level >= 1:
        raise ValidationError(f"conf_level must be in (0, 1), got {conf_level}")

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
        backend_name=f"{'cpu' if backend == 'cpu' else 'gpu'}_discrete",
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

    return DiscreteTimeSolution(_result=result, _names=resolved_names,
                                _conf_level=conf_level)


def cox_zph(
    fit: CoxSolution,
    *,
    transform: Literal["km", "rank", "identity", "log"] = "km",
) -> CoxZphSolution:
    """Test the proportional-hazards assumption of a fitted Cox model.

    Score test on scaled Schoenfeld residuals — the survival >= 3.0
    formulation of ``survival::cox.zph`` (not the pre-3.0 correlation test).
    Reports one chi-square per covariate plus a GLOBAL test, and carries the
    scaled Schoenfeld residuals for plotting.

    Parameters
    ----------
    fit : CoxSolution
        A fitted Cox model from :func:`coxph` (stratified or not). The fit
        retains its data for exactly this purpose.
    transform : str
        Time transform g(t): "km" (default, left-continuous KM CDF of the
        pooled sample — R's default), "rank", "identity", or "log".

    Returns
    -------
    CoxZphSolution
    """
    if not isinstance(fit, CoxSolution):
        raise ValidationError(
            f"fit must be a CoxSolution from coxph(), got {type(fit).__name__}"
        )
    design = fit._design
    if design is None:
        raise ValidationError(
            "This CoxSolution does not carry its design data (it predates "
            "cox_zph support); refit with coxph() and call cox_zph on the "
            "fresh fit"
        )
    params = fit._result.params
    if params.n_events == 0:
        raise ValidationError("cox_zph requires at least one event")

    data = cox_zph_compute(
        design.time, design.event, design.X,
        beta=params.coefficients,
        strata=design.strata,
        ties=params.ties,
        transform=transform,
        entry=design.entry,
    )

    p = design.X.shape[1]
    names = fit._names or tuple(f"x{i}" for i in range(p))
    return CoxZphSolution(_data=data, _names=tuple(names))
