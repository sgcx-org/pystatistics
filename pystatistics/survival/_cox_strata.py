"""
Stratified Cox proportional hazards.

A stratified fit shares one coefficient vector across strata but gives each
stratum its own baseline hazard (its own risk sets). The stratified partial
log-likelihood, score, and information are simply the *sums* over strata of the
single-stratum quantities evaluated at the shared ``beta`` — so this module
reuses ``_cox``'s vectorized per-stratum primitives (``_partial_loglik`` /
``_score_and_information``) and the shared Newton-Raphson driver, adding only the
per-stratum bookkeeping and the cross-stratum aggregation.

Concordance follows R's ``concordance.coxph``: pairs are comparable only within a
stratum, so the concordant / discordant / risk-tied counts are summed across
strata before forming a single C.

Matches ``survival::coxph(Surv(time, event) ~ x + strata(g))``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from pystatistics.survival._common import CoxParams
from pystatistics.survival._cox import (
    _concordance_counts,
    _concordance_ratio,
    _partial_loglik,
    _score_and_information,
)
from pystatistics.survival._cox_newton import flag_infinite_coefs, solve_cox_newton


@dataclass(frozen=True)
class _Stratum:
    """One stratum's time-sorted data plus its distinct event times.

    Sorted ascending by time (censored before events at ties) exactly as
    ``_cox.cox_fit`` sorts a single group, so the reused risk-set primitives see
    the layout they expect.
    """

    time: NDArray          # (n_k,) sorted ascending
    event: NDArray         # (n_k,)
    X: NDArray             # (n_k, p)
    unique_event_times: NDArray  # (m_k,) distinct event times, ascending
    entry: NDArray | None = None  # (n_k,) counting-process starts, row-aligned


def _build_strata(
    time: NDArray, event: NDArray, X: NDArray, strata: NDArray,
    entry: NDArray | None = None,
) -> list[_Stratum]:
    """Partition the sample into per-stratum, time-sorted views."""
    strata_flat = np.asarray(strata).ravel()
    # np.unique gives a deterministic (sorted) stratum order, matching R's
    # order(strata, time); the reported coefficients are order-invariant anyway.
    labels = np.unique(strata_flat)
    out: list[_Stratum] = []
    for lab in labels:
        idx = np.nonzero(strata_flat == lab)[0]
        tk = time[idx]
        ek = event[idx]
        Xk = X[idx]
        entryk = entry[idx] if entry is not None else None
        order = np.lexsort((ek, tk))  # ascending time, censored before events
        tk = tk[order]
        ek = ek[order]
        Xk = Xk[order]
        if entryk is not None:
            entryk = entryk[order]
        uet = np.unique(tk[ek == 1])
        out.append(_Stratum(time=tk, event=ek, X=Xk, unique_event_times=uet,
                            entry=entryk))
    return out


def cox_fit_stratified(
    time: NDArray,
    event: NDArray,
    X: NDArray,
    strata: NDArray,
    ties: str = "efron",
    tol: float = 1e-9,
    max_iter: int = 20,
    conf_level: float = 0.95,
    entry: NDArray | None = None,
) -> CoxParams:
    """Fit a stratified Cox proportional hazards model.

    Parameters
    ----------
    time : (n,)
        Time to event or censoring (risk-interval exit).
    event : (n,)
        Event indicator (1=event, 0=censored).
    X : (n, p)
        Covariate matrix (NO intercept).
    strata : (n,)
        Stratum label per observation.
    ties : str
        "efron" (default) or "breslow".
    tol, max_iter : float, int
        Newton-Raphson controls.
    conf_level : float
        Confidence level carried on the result for ``.conf_int``.
    entry : (n,) or None
        Counting-process start times (row at risk on ``(entry, time]``).

    Returns
    -------
    CoxParams
    """
    n, p = X.shape
    # Global mean-centering for numerical stability (invariant to the fit; see
    # cox_fit). R centers globally across all strata, so we do too.
    X = X - X.mean(axis=0)
    strata_list = _build_strata(time, event, X, strata, entry=entry)
    n_strata = len(strata_list)
    n_events_total = int(np.sum(event))

    if n_events_total == 0:
        # No events in any stratum — unidentified (see cox_fit's no-events note).
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
            n_strata=n_strata,
        )

    def eval_full(beta: NDArray) -> tuple[float, NDArray, NDArray]:
        loglik = 0.0
        score = np.zeros(p, dtype=np.float64)
        info = np.zeros((p, p), dtype=np.float64)
        for s in strata_list:
            if len(s.unique_event_times) == 0:
                continue  # a stratum with no events contributes nothing
            ll, sc, im = _score_and_information(
                beta, s.time, s.event, s.X, s.unique_event_times, ties,
                entry=s.entry,
            )
            loglik += ll
            score += sc
            info += im
        return loglik, score, info

    def eval_loglik(beta: NDArray) -> float:
        loglik = 0.0
        for s in strata_list:
            if len(s.unique_event_times) == 0:
                continue
            loglik += _partial_loglik(
                beta, s.time, s.event, s.X, s.unique_event_times, ties,
                entry=s.entry,
            )
        return loglik

    fit = solve_cox_newton(p, eval_full, eval_loglik, tol, max_iter)
    beta = fit.beta

    infinite_coefs: tuple[int, ...] = ()
    try:
        var_matrix = np.linalg.inv(fit.information)
        se = np.sqrt(np.maximum(np.diag(var_matrix), 0.0))
        if fit.converged:
            infinite_coefs = flag_infinite_coefs(beta, fit.score, var_matrix, tol)
    except np.linalg.LinAlgError:
        se = np.full(p, np.inf)

    z = np.where(se > 0, beta / se, 0.0)
    p_values = 2.0 * stats.norm.sf(np.abs(z))

    # Concordance: sum comparable-pair counts WITHIN each stratum, then ratio.
    concordant = discordant = tied_risk = 0.0
    for s in strata_list:
        c, d, tr = _concordance_counts(beta, s.time, s.event, s.X,
                                       entry=s.entry)
        concordant += c
        discordant += d
        tied_risk += tr
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
        n_strata=n_strata,
        infinite_coefs=infinite_coefs,
    )
