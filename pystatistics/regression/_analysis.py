"""
Analysis-of-deviance tables for fitted linear / generalized-linear models:
``anova`` (sequential, or a nested-model comparison) and ``drop1``.

These mirror R's ``anova.lm`` / ``anova.glm`` and ``drop1``:

- ``anova(model)``          — sequential (Type I) analysis of deviance: add each
  term in turn and test the deviance it explains.
- ``anova(m1, m2, ...)``    — compare nested models: the deviance difference
  between consecutive fits.
- ``drop1(model)``          — drop each term singly from the full model and test
  the deviance it accounts for.

The test statistic follows R's convention: a chi-square (LRT) test for
fixed-dispersion families (binomial/poisson), an F test for models with an
estimated dispersion (gaussian/gamma/quasi/inverse-gaussian) — using the full
model's dispersion as the scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from pystatistics.core.exceptions import ValidationError


@dataclass(frozen=True)
class AnovaRow:
    """One row of an analysis-of-deviance table."""
    term: str
    df: int | None            # degrees of freedom for the term (None for a base row)
    deviance: float | None    # deviance explained (or SS for gaussian)
    resid_df: int             # residual df after this row
    resid_deviance: float     # residual deviance after this row
    statistic: float | None = None   # chi-square or F statistic
    p_value: float | None = None
    aic: float | None = None         # only populated by drop1


@dataclass(frozen=True)
class AnovaTable:
    """Result of :func:`anova` / :func:`drop1` — a list of rows plus metadata."""
    rows: tuple[AnovaRow, ...]
    test: str | None
    kind: str                 # 'sequential', 'comparison', or 'drop1'

    def summary(self) -> str:
        head = {
            'sequential': "Analysis of Deviance Table (Type I, sequential)",
            'comparison': "Analysis of Deviance Table (model comparison)",
            'drop1': "Single-term deletions (drop1)",
        }[self.kind]
        lines = [head, "=" * 74,
                 f"{'Term':<22}{'Df':>5}{'Deviance':>12}{'Resid.Df':>10}"
                 f"{'Resid.Dev':>12}"
                 + (f"{'Stat':>10}{'Pr(>'+ (self.test or '')+')':>10}"
                    if self.test else "")]
        for r in self.rows:
            df = "" if r.df is None else f"{r.df:>5d}"
            dev = "" if r.deviance is None else f"{r.deviance:>12.4f}"
            stat = "" if r.statistic is None else f"{r.statistic:>10.4f}"
            p = "" if r.p_value is None else f"{r.p_value:>10.4g}"
            lines.append(
                f"{r.term:<22}{df:>5}{dev:>12}{r.resid_df:>10d}"
                f"{r.resid_deviance:>12.4f}"
                + (f"{stat}{p}" if self.test else "")
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"AnovaTable(kind={self.kind!r}, rows={len(self.rows)}, test={self.test!r})"


# --------------------------------------------------------------------------
# Term structure + refitting helpers
# --------------------------------------------------------------------------

def _term_structure(model) -> tuple[NDArray, NDArray, list[int], list[str]]:
    """Return (X, y, assign, term_names) for a fitted model.

    Uses the design's term spec when present; otherwise treats each column as its
    own term (detecting an all-ones intercept column), matching R's handling of a
    numeric-only model matrix.
    """
    design = model._design
    X = np.asarray(design.X, dtype=np.float64)
    y = np.asarray(design.y, dtype=np.float64)
    if design.assign is not None:
        return X, y, list(design.assign), list(design.term_names)

    # Raw-array design: prefer the user-supplied column names on the solution,
    # falling back to the design's names, then to positional labels.
    names = getattr(model, '_names', None) or design.names
    p = X.shape[1]
    assign = list(range(p))
    term_names = []
    for j in range(p):
        if np.allclose(X[:, j], 1.0):
            term_names.append("(Intercept)")
        elif names is not None and j < len(names):
            term_names.append(names[j])
        else:
            term_names.append(f"x{j + 1}")
    return X, y, assign, term_names


def _family_of(model):
    """The fitted GLM family object, or None for an OLS/linear fit."""
    return model._result.info.get('family')


def _refit_deviance(model, cols: list[int]) -> tuple[float, int]:
    """Refit on the given column subset; return (deviance, residual_df).

    For a linear model the 'deviance' is the residual sum of squares (R's
    convention in ``anova.lm``).
    """
    from pystatistics.regression.solvers import fit
    X = np.asarray(model._design.X, dtype=np.float64)[:, cols]
    y = np.asarray(model._design.y, dtype=np.float64)
    family = _family_of(model)
    sub = fit(X, y, family=family)
    if family is None:
        return float(sub.rss), int(sub._result.params.df_residual)
    return float(sub.deviance), int(sub._result.params.df_residual)


def _default_test(model) -> str:
    """R's default test: F for estimated-dispersion families / LM, else Chisq."""
    family = _family_of(model)
    if family is None:
        return 'F'                      # linear model
    return 'Chisq' if family.dispersion_is_fixed else 'F'


def _cols_for_terms(assign: list[int], term_ids: list[int]) -> list[int]:
    keep = set(term_ids)
    return [j for j, t in enumerate(assign) if t in keep]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def anova(*models, test: str | None = None) -> AnovaTable:
    """Analysis-of-deviance table.

    ``anova(model)`` gives a sequential (Type I) table; ``anova(m1, m2, ...)``
    compares nested models. ``test`` is one of ``'Chisq'``/``'LRT'``, ``'F'``, or
    ``None`` (no p-values); it defaults to R's choice for the model's family.
    """
    if not models:
        raise ValidationError("anova requires at least one fitted model")
    if len(models) == 1:
        return _anova_sequential(models[0], test)
    return _anova_comparison(list(models), test)


def _term_test(deviance: float, df: int, dispersion: float,
               resid_df: int, test: str | None):
    """Return (statistic, p_value) for a term, per the requested test."""
    if test is None or df <= 0:
        return None, None
    if test in ('Chisq', 'LRT'):
        stat = deviance / dispersion
        return stat, float(stats.chi2.sf(stat, df))
    if test == 'F':
        f = (deviance / df) / dispersion
        return f, float(stats.f.sf(f, df, resid_df))
    raise ValidationError(f"Unknown test {test!r}. Use 'Chisq', 'LRT', 'F', or None.")


def _anova_sequential(model, test: str | None) -> AnovaTable:
    X, y, assign, term_names = _term_structure(model)
    if test is None:
        test = _default_test(model)
    family = _family_of(model)

    # Ordered non-intercept term ids in the order they appear.
    seen: list[int] = []
    for t in assign:
        if t not in seen:
            seen.append(t)
    intercept_ids = [t for t in seen if term_names[t] == "(Intercept)"]
    term_ids = [t for t in seen if t not in intercept_ids]

    # Full-model dispersion is the scale for the tests (R's convention).
    disp_full = 1.0
    if family is None or not family.dispersion_is_fixed:
        disp_full = model.dispersion if family is not None else \
            model.residual_std_error ** 2

    base_cols = _cols_for_terms(assign, intercept_ids)
    prev_dev, prev_rdf = _refit_deviance(model, base_cols) if base_cols else \
        (_refit_deviance(model, list(range(X.shape[1])))[0], X.shape[0])

    rows = [AnovaRow(term="NULL" if base_cols else term_names[term_ids[0]],
                     df=None, deviance=None,
                     resid_df=prev_rdf, resid_deviance=prev_dev)]
    cols = list(base_cols)
    for t in term_ids:
        tcols = _cols_for_terms(assign, [t])
        cols = cols + tcols
        dev, rdf = _refit_deviance(model, cols)
        d_dev = prev_dev - dev
        df = len(tcols)
        stat, p = _term_test(d_dev, df, disp_full, model._result.params.df_residual, test)
        rows.append(AnovaRow(term=term_names[t], df=df, deviance=d_dev,
                             resid_df=rdf, resid_deviance=dev,
                             statistic=stat, p_value=p))
        prev_dev = dev
    return AnovaTable(rows=tuple(rows), test=test, kind='sequential')


def _anova_comparison(models: list, test: str | None) -> AnovaTable:
    if test is None:
        test = _default_test(models[0])
    family = _family_of(models[0])
    disp = 1.0
    # Use the largest (last) model's dispersion as the scale.
    biggest = max(models, key=lambda m: m._result.params.rank)
    if family is None:
        disp = biggest.residual_std_error ** 2
    elif not family.dispersion_is_fixed:
        disp = biggest.dispersion

    def dev_of(m):
        return (float(m.rss) if _family_of(m) is None else float(m.deviance))

    rows = []
    prev = None
    for i, m in enumerate(models):
        rdf = m._result.params.df_residual
        rdev = dev_of(m)
        if prev is None:
            rows.append(AnovaRow(term=f"Model {i+1}", df=None, deviance=None,
                                 resid_df=rdf, resid_deviance=rdev))
        else:
            df = prev[0] - rdf
            d_dev = prev[1] - rdev
            stat, p = _term_test(abs(d_dev), abs(df), disp,
                                 min(rdf, prev[0]), test) if df != 0 else (None, None)
            rows.append(AnovaRow(term=f"Model {i+1}", df=df, deviance=d_dev,
                                 resid_df=rdf, resid_deviance=rdev,
                                 statistic=stat, p_value=p))
        prev = (rdf, rdev)
    return AnovaTable(rows=tuple(rows), test=test, kind='comparison')


def drop1(model, test: str | None = None) -> AnovaTable:
    """Single-term deletions: drop each term singly and test its contribution.

    Mirrors R's ``drop1`` — the ``<none>`` row is the full model, and each
    subsequent row drops one term, reporting the resulting residual deviance,
    AIC, and (optionally) a test of the deviance increase.
    """
    X, y, assign, term_names = _term_structure(model)
    if test is None:
        test = _default_test(model)
    family = _family_of(model)

    seen: list[int] = []
    for t in assign:
        if t not in seen:
            seen.append(t)
    term_ids = [t for t in seen if term_names[t] != "(Intercept)"]

    full_dev = float(model.rss) if family is None else float(model.deviance)
    full_rdf = model._result.params.df_residual
    full_aic = model.aic if family is not None else _lm_aic(model)
    disp_full = 1.0
    if family is None:
        disp_full = model.residual_std_error ** 2
    elif not family.dispersion_is_fixed:
        disp_full = model.dispersion

    rows = [AnovaRow(term="<none>", df=None, deviance=None,
                     resid_df=full_rdf, resid_deviance=full_dev, aic=full_aic)]
    all_ids = seen
    for t in term_ids:
        keep = [tid for tid in all_ids if tid != t]
        cols = _cols_for_terms(assign, keep)
        dev, rdf = _refit_deviance(model, cols)
        df = len(_cols_for_terms(assign, [t]))
        d_dev = dev - full_dev
        stat, p = _term_test(d_dev, df, disp_full, full_rdf, test)
        # AIC of the reduced model (fixed-dispersion families / LM only rigorously).
        sub_aic = _reduced_aic(model, cols)
        rows.append(AnovaRow(term=term_names[t], df=df, deviance=d_dev,
                             resid_df=rdf, resid_deviance=dev,
                             statistic=stat, p_value=p, aic=sub_aic))
    return AnovaTable(rows=tuple(rows), test=test, kind='drop1')


def _lm_aic(model) -> float:
    """AIC of a linear model, matching R's ``AIC.lm``.

    ``n log(2 pi RSS/n) + n + 2 (p + 1)`` — the +1 counts the error variance.
    """
    n = model._design.n
    p = model._result.params.rank
    rss = float(model.rss)
    return n * np.log(2 * np.pi * rss / n) + n + 2 * (p + 1)


def _reduced_aic(model, cols: list[int]) -> float:
    from pystatistics.regression.solvers import fit
    X = np.asarray(model._design.X, dtype=np.float64)[:, cols]
    y = np.asarray(model._design.y, dtype=np.float64)
    family = _family_of(model)
    sub = fit(X, y, family=family)
    return _lm_aic(sub) if family is None else float(sub.aic)
