"""
ETS automatic model selection ("Z" wildcards), matching R forecast::ets.

Public entry point for ETS fitting.  A model string may fix any component
to a concrete letter or leave it to be selected with ``"Z"`` — e.g.
``"ZZZ"`` (select everything, the default, as in ``forecast::ets``),
``"ZZN"`` (select error/trend, no season), ``"MZZ"`` (multiplicative
error, select the rest).  Candidates consistent with the fixed letters
are enumerated, each is fitted with the engine in ``_ets_fit.py``, and
the model minimising the requested information criterion (AICc by
default) is returned, with the full candidate table disclosed on
``solution.info["selection"]``.

The candidate set mirrors ``forecast::ets`` with its defaults
(``restrict=TRUE``, ``allow.multiplicative.trend=FALSE``), enumerated in
R's loop order (error, then trend type, then season, with the damped
variant innermost):

* error ``Z`` expands to {A, M}; trend ``Z`` to {N, A, Ad}; season ``Z``
  to {N, A, M};
* additive error with multiplicative season is excluded (numerically
  unstable), so ``"ZZZ"`` on strictly positive seasonal data enumerates
  15 models (6 A-error, 9 M-error);
* multiplicative-error candidates are dropped when the data are not
  strictly positive (an *explicitly requested* M error/season on such
  data raises instead — fail loud);
* a ``Z`` season collapses to ``N`` when ``period == 1``, when
  ``n <= period``, or when ``period > 24`` (as in R, which warns in the
  last case — the warning is preserved on the solution);
* an *explicit* seasonal letter with ``period == 1`` or ``period > 24``
  raises, as R stops ("Nonseasonal data" / "Frequency too high").  For
  ``n <= period`` R silently coerces an explicit seasonal letter to
  non-seasonal; PyStatistics instead lets those seasonal candidates fail
  their data requirements loudly (deliberate fail-loud divergence);
* candidates whose fit fails are skipped and recorded (R silently
  assigns them infinite IC).

Deliberate divergences from ``forecast::ets`` (documented, fail-loud):

* A fully-specified model (no ``"Z"``) with ``damped=None`` is fitted
  exactly as written; R would also try the damped variant and return the
  better one.  Selection here is always explicit — via ``"Z"`` or
  ``damped=None`` *with* a ``"Z"`` string.
* R falls back to unoptimised Holt-Winters fits for very short series
  (``n <= npars + 4``); PyStatistics raises instead of silently swapping
  estimators.  In particular the default ``ets(y)`` needs ``n >= 5`` for
  AICc-based selection (the smallest candidate has k = 3 parameters and
  AICc requires ``n > k + 1``); shorter series raise with a remedy.
* Reported log-likelihood/AIC use the full-Gaussian convention (see
  ``_ets_fit.py``); model *rankings* are identical to R's.

Selection parity with R: the *candidate set* matches ``forecast::ets``
exactly.  The *selected model* usually matches but can differ when
candidates are nearly tied under the criterion, because the fitting
engine optimises a slightly wider parameter space than R's defaults —
independent ``(0, 1)`` bounds on beta/gamma where R's "usual" region
enforces ``beta < alpha`` and ``gamma < 1 - alpha``; ``m`` free initial
seasonal states where R estimates ``m - 1`` and normalises the last;
``phi`` bounded at 0.999 vs R's 0.98 — and therefore can reach slightly
better likelihoods for trended/seasonal candidates.  Each returned
solution discloses its full candidate IC table in
``info["selection"]``, so any selection can be audited.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np
from numpy.typing import ArrayLike

from pystatistics.core.exceptions import (
    ConvergenceError,
    PyStatisticsError,
    ValidationError,
)
from pystatistics.core.validation import check_array, check_finite
from pystatistics.timeseries._ets_fit import ETSSolution, fit_ets_model

_WILDCARD_RE = re.compile(
    r"^(?:ETS\()?"
    r"([AMZ])"          # error
    r",?\s*"
    r"(N|Ad?|Z)"        # trend
    r",?\s*"
    r"(N|[AMZ])"        # season
    r"(?:\))?$",
    re.IGNORECASE,
)

_VALID_IC = ("aicc", "aic", "bic")
_MAX_SEASONAL_PERIOD = 24  # forecast::ets refuses seasonal periods > 24


def _parse_letters(model: str) -> tuple[str, str, str]:
    """Split a model string into (error, trend, season), allowing 'Z'."""
    if not isinstance(model, str) or not model.strip():
        raise ValidationError(f"model: expected non-empty string, got {model!r}")
    match = _WILDCARD_RE.match(model.strip())
    if match is None:
        raise ValidationError(
            f"model: cannot parse {model!r}. Expected format like 'AAN', "
            "'AAdN', 'A,Ad,N', 'ETS(A,Ad,N)', or 'Z' wildcards like 'ZZZ'"
        )
    error = match.group(1).upper()
    trend_raw = match.group(2)
    trend = "Ad" if trend_raw.upper() == "AD" else trend_raw.upper()
    season = match.group(3).upper()
    return error, trend, season


def _check_period(period: object) -> int:
    """Periods are non-bool integers >= 1 (fail loud on floats/strings)."""
    if isinstance(period, bool) or not isinstance(period, (int, np.integer)):
        raise ValidationError(
            f"period: expected integer, got {type(period).__name__}"
        )
    if period < 1:
        raise ValidationError(f"period: must be >= 1, got {period}")
    return int(period)


def _trend_options(
    trend: str, damped: bool | None
) -> list[tuple[str, list[bool]]]:
    """
    ``(trend_type, damped_flags)`` pairs for one requested trend component.

    The pairs — and the flags within each pair — are ordered exactly as
    forecast::ets's selection loops visit them (trend types outer, the
    damped variant first within each type), which also fixes tie-breaks.
    """
    if trend == "Z":
        if damped is None:
            return [("N", [False]), ("A", [True, False])]
        return [("A", [True])] if damped else [("N", [False]), ("A", [False])]
    if trend == "N":
        if damped:
            raise ValidationError(
                "damped=True requires a trend component (model has trend='N')"
            )
        return [("N", [False])]
    if trend == "A":
        if damped is None:
            return [("A", [True, False])]
        return [("A", [True])] if damped else [("A", [False])]
    # trend == "Ad": explicitly damped in the string
    if damped is False:
        raise ValidationError(
            "damped: model specifies a damped trend ('Ad') but damped=False"
        )
    return [("A", [True])]


def _season_wildcard(
    period: int, n_obs: int
) -> tuple[list[str], list[str]]:
    """
    Expand a ``Z`` season, honouring R's limits (see module docstring).

    Explicit seasonal letters are validated in :func:`ets` before the
    candidate enumeration; this handles only the wildcard collapse rules.
    """
    warnings: list[str] = []
    if period < 2 or n_obs <= period:
        return ["N"], warnings
    if period > _MAX_SEASONAL_PERIOD:
        warnings.append(
            f"seasonal candidates dropped: period {period} exceeds the "
            f"supported maximum of {_MAX_SEASONAL_PERIOD} (as in R "
            "forecast::ets); only non-seasonal models were considered"
        )
        return ["N"], warnings
    return ["N", "A", "M"], warnings


def _validate_series(y: ArrayLike) -> np.ndarray:
    """Boundary validation shared by both paths: 1-D, finite, n >= 3."""
    arr = check_array(y, "y")
    if sum(1 for size in arr.shape if size > 1) > 1:
        raise ValidationError(
            f"y: expected a 1-D series, got shape {arr.shape}"
        )
    y_arr = arr.ravel()
    check_finite(y_arr, "y")
    if len(y_arr) < 3:
        raise ValidationError(
            f"y: requires at least 3 observations, got {len(y_arr)}"
        )
    return y_arr


def ets(
    y: ArrayLike,
    *,
    model: str = "ZZZ",
    period: int = 1,
    damped: bool | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    phi: float | None = None,
    ic: str = "aicc",
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> ETSSolution:
    """
    Fit an ETS (ExponenTial Smoothing) state space model.

    Matches R's ``forecast::ets()``: each component of the model string
    may be a concrete letter or a ``"Z"`` wildcard, and wildcards are
    resolved by fitting every admissible candidate and selecting the one
    minimising *ic*.  The default ``model="ZZZ"`` performs full automatic
    selection, as in R.  See the module docstring for the exact candidate
    table and the documented divergences from R.

    Parameters
    ----------
    y : ArrayLike
        Time series (1-D). Multiplicative components require strictly
        positive values.
    model : str
        ETS model string — error, trend, season — e.g. ``'ZZZ'``
        (default: select everything), ``'ANN'``, ``'AAdN'``, ``'MZZ'``,
        ``'ZZN'``.
    period : int
        Seasonal period (e.g. 12 for monthly, 4 for quarterly).
        Seasonal models require ``2 <= period <= 24``.
    damped : bool or None
        Damped-trend control. With a concrete model string it forces or
        forbids damping; with a ``'Z'`` trend, ``None`` means both damped
        and undamped candidates are tried, ``True``/``False`` restricts
        the candidate set accordingly. ``damped=True`` with a trend fixed
        to ``'N'`` raises (R: "Forbidden model combination").
    alpha, beta, gamma, phi : float or None
        Fix specific smoothing parameters (applies to every candidate).
    ic : str
        Selection criterion for wildcard models: ``'aicc'`` (default,
        matching forecast::ets), ``'aic'``, or ``'bic'``. AICc-based
        selection needs ``n >= 5`` (see module docstring).
    tol : float
        Convergence tolerance for the optimiser.
    max_iter : int
        Maximum optimiser iterations.

    Returns
    -------
    ETSSolution
        The fitted (for wildcards: selected) model.  For wildcard
        requests, ``solution.info["selection"]`` records the requested
        string, the criterion, every candidate's IC value, and any
        skipped candidates with the reason.

    Raises
    ------
    ValidationError
        On invalid inputs; on an explicitly requested component that
        cannot be honoured (e.g. multiplicative error on non-positive
        data, a seasonal letter with ``period`` of 1 or > 24,
        ``damped=True`` with trend ``'N'``); when no candidate is
        admissible; or when no candidate has a finite *ic* (series too
        short for AICc).
    ConvergenceError
        If candidates were attempted but none could be fitted.
    """
    if ic not in _VALID_IC:
        raise ValidationError(f"ic: must be one of {_VALID_IC}, got {ic!r}")

    error, trend, season = _parse_letters(model)
    period = _check_period(period)

    # Explicit seasonal letters share R's hard limits on both paths.
    if season in ("A", "M"):
        if period < 2:
            raise ValidationError(
                f"model: seasonal component {season!r} requires "
                f"period >= 2, got period={period}"
            )
        if period > _MAX_SEASONAL_PERIOD:
            raise ValidationError(
                f"period: seasonal ETS supports period <= "
                f"{_MAX_SEASONAL_PERIOD}, got {period} (matching R "
                "forecast::ets's 'Frequency too high' limit)"
            )

    y_arr = _validate_series(y)

    if "Z" not in (error, trend[0], season):
        # Fully specified: fit exactly what was asked (documented
        # divergence: R with damped=NULL would also try the damped twin).
        return fit_ets_model(
            y_arr, model=model, period=period, damped=damped, alpha=alpha,
            beta=beta, gamma=gamma, phi=phi, tol=tol, max_iter=max_iter,
        )

    n_obs = len(y_arr)
    data_positive = bool(np.min(y_arr) > 0.0)

    if error == "M" and not data_positive:
        raise ValidationError(
            "model: multiplicative error requires strictly positive data"
        )
    if season == "M" and not data_positive:
        raise ValidationError(
            "model: multiplicative season requires strictly positive data"
        )

    error_letters = ["A", "M"] if error == "Z" else [error]
    trend_options = _trend_options(trend, damped)
    if season == "Z":
        season_letters, sel_warnings = _season_wildcard(period, n_obs)
    else:
        season_letters, sel_warnings = [season], []

    skipped: list[dict] = []
    candidates: list[dict] = []
    best: ETSSolution | None = None
    best_ic = np.inf
    attempted_failure = False
    for err in error_letters:
        if err == "M" and not data_positive:
            skipped.append({
                "model": f"M{trend}{season}",
                "reason": "multiplicative error requires strictly "
                          "positive data",
            })
            continue
        for trend_type, damped_flags in trend_options:
            for sea in season_letters:
                for is_damped in damped_flags:
                    trd = "Ad" if is_damped else trend_type
                    name = f"{err}{trd}{sea}"
                    if err == "A" and sea == "M":
                        skipped.append({
                            "model": name,
                            "reason": "additive error with multiplicative "
                                      "season is excluded (forecast::ets "
                                      "restrict=TRUE)",
                        })
                        continue
                    try:
                        fit = fit_ets_model(
                            y_arr, model=name, period=period, alpha=alpha,
                            beta=beta, gamma=gamma, phi=phi, tol=tol,
                            max_iter=max_iter,
                        )
                    except PyStatisticsError as exc:
                        skipped.append({"model": name, "reason": str(exc)})
                        if isinstance(exc, ConvergenceError):
                            attempted_failure = True
                        continue
                    ic_value = {"aic": fit.aic, "aicc": fit.aicc,
                                "bic": fit.bic}[ic]
                    candidates.append({"model": fit.spec.name, ic: ic_value})
                    if np.isfinite(ic_value) and ic_value < best_ic:
                        best = fit
                        best_ic = ic_value

    if best is None:
        if candidates:
            # Fits succeeded but every criterion value is non-finite
            # (AICc is infinite whenever n <= k + 1).
            raise ValidationError(
                f"ets: all {len(candidates)} admissible candidates fitted "
                f"but none has a finite {ic} (n={n_obs} is too small for "
                f"{ic}-based selection; the smallest candidate needs "
                "n >= 5). Fit a concrete model (e.g. model='ANN') or "
                "select with ic='aic' or ic='bic'."
            )
        detail = "; ".join(f"{s['model']}: {s['reason']}" for s in skipped)
        if attempted_failure:
            raise ConvergenceError(
                f"ets: no candidate model could be fitted for "
                f"model={model!r} (period={period}, n={n_obs}). "
                f"Skipped: {detail}",
                iterations=0,
                reason="no_fittable_candidate",
            )
        raise ValidationError(
            f"ets: no admissible candidate for model={model!r} "
            f"(period={period}, n={n_obs}). Skipped: {detail}"
        )

    selection = {
        "requested": model,
        "ic": ic,
        "selected": best.spec.name,
        "candidates": candidates,
        "skipped": skipped,
    }
    result = best._result
    augmented = dataclasses.replace(
        result,
        info={**result.info, "selection": selection},
        warnings=result.warnings + tuple(sel_warnings),
    )
    return ETSSolution(_result=augmented)
