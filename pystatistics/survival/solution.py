"""
Solution wrappers for survival analysis results.

Each Solution wraps a Result[Params] and exposes user-friendly properties
with R-style summary() methods.
"""

from __future__ import annotations

from pystatistics.core.result import Result, SolutionReprMixin
from pystatistics.survival._common import (
    CoxParams,
    DiscreteTimeParams,
    KMParams,
    LogRankParams,
)


class KMSolution(SolutionReprMixin):
    """Kaplan-Meier survival curve solution.

    Properties mirror R's survfit() output.
    """

    __slots__ = ('_result',)

    def __init__(self, _result: Result[KMParams]) -> None:
        self._result = _result

    # -- Properties delegating to KMParams --

    @property
    def time(self):
        """Unique event times."""
        return self._result.params.time

    @property
    def survival(self):
        """S(t) at each event time."""
        return self._result.params.survival

    @property
    def n_risk(self):
        """Number at risk just before each event time."""
        return self._result.params.n_risk

    @property
    def n_events(self):
        """Number of events at each event time."""
        return self._result.params.n_events

    @property
    def n_censored(self):
        """Number censored in each interval."""
        return self._result.params.n_censored

    @property
    def se(self):
        """Greenwood standard error of S(t)."""
        return self._result.params.se

    @property
    def standard_errors(self):
        """Greenwood standard error of S(t) (uniform-accessor alias of ``se``)."""
        return self._result.params.se

    @property
    def ci_lower(self):
        """Lower confidence bound for S(t)."""
        return self._result.params.ci_lower

    @property
    def ci_upper(self):
        """Upper confidence bound for S(t)."""
        return self._result.params.ci_upper

    @property
    def conf_int(self):
        """Confidence band for S(t) at each event time, shape (m, 2).

        Columns are ``[lower, upper]`` — the uniform-accessor form of
        ``ci_lower`` / ``ci_upper`` (which remain available).
        """
        import numpy as np
        return np.column_stack([self._result.params.ci_lower,
                                self._result.params.ci_upper])

    @property
    def conf_level(self) -> float:
        return self._result.params.conf_level

    @property
    def conf_type(self) -> str:
        return self._result.params.conf_type

    @property
    def n_observations(self) -> int:
        return self._result.params.n_observations

    @property
    def n_events_total(self) -> int:
        return self._result.params.n_events_total

    @property
    def median_survival(self) -> float | None:
        """Median survival time, matching R survfit's ``minmin`` convention.

        The median is the smallest ``t`` with ``S(t) <= 0.5``; but when the
        curve touches exactly 0.5 (a flat step at 0.5), R averages that time
        with the next time the curve drops strictly below 0.5. Returns None if
        the curve never reaches 0.5.
        """
        import numpy as np

        surv = np.asarray(self.survival)
        time = np.asarray(self.time)
        if len(surv) == 0:
            return None
        tol = np.finfo(np.float64).eps ** 0.5
        keep = surv < 0.5 + tol           # includes S == 0.5 within tolerance
        if not keep.any():
            return None
        x = time[keep]
        y = surv[keep]
        if abs(y[0] - 0.5) < tol and np.any(y < y[0]):
            j = int(np.argmax(y < y[0]))  # first point strictly below S(t_first)
            return float((x[0] + x[j]) / 2.0)
        return float(x[0])

    @property
    def backend_name(self) -> str:
        return self._result.backend_name

    @property
    def timing(self):
        return self._result.timing

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._result.warnings

    def summary(self) -> str:
        """R-style summary of Kaplan-Meier fit."""
        lines = []
        lines.append("Call: kaplan_meier()")
        lines.append("")
        lines.append(
            f"  n={self.n_observations}, "
            f"events={self.n_events_total}"
        )
        lines.append("")

        median = self.median_survival
        median_str = f"{median:.4g}" if median is not None else "NA"
        lines.append(f"  median survival = {median_str}")
        lines.append("")

        # Table header
        ci_pct = int(self.conf_level * 100)
        lines.append(
            f"  {'time':>8s}  {'n.risk':>8s}  {'n.event':>8s}  "
            f"{'survival':>10s}  {'se':>10s}  "
            f"{'lower ' + str(ci_pct) + '%':>14s}  {'upper ' + str(ci_pct) + '%':>14s}"
        )

        # Show up to 20 rows
        m = len(self.time)
        show = min(m, 20)
        for i in range(show):
            lines.append(
                f"  {self.time[i]:8.4g}  {self.n_risk[i]:8.0f}  "
                f"{self.n_events[i]:8.0f}  "
                f"{self.survival[i]:10.6f}  {self.se[i]:10.6f}  "
                f"{self.ci_lower[i]:10.6f}  {self.ci_upper[i]:10.6f}"
            )
        if m > 20:
            lines.append(f"  ... ({m - 20} more rows)")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"KMSolution(n={self.n_observations}, "
            f"events={self.n_events_total}, "
            f"median={self.median_survival})"
        )


class LogRankSolution(SolutionReprMixin):
    """Log-rank test solution.

    Properties mirror R's survdiff() output.
    """

    __slots__ = ('_result',)

    def __init__(self, _result: Result[LogRankParams]) -> None:
        self._result = _result

    @property
    def statistic(self) -> float:
        return self._result.params.statistic

    @property
    def df(self) -> int:
        return self._result.params.df

    @property
    def p_value(self) -> float:
        return self._result.params.p_value

    @property
    def n_groups(self) -> int:
        return self._result.params.n_groups

    @property
    def observed(self):
        return self._result.params.observed

    @property
    def expected(self):
        return self._result.params.expected

    @property
    def n_per_group(self):
        return self._result.params.n_per_group

    @property
    def rho(self) -> float:
        return self._result.params.rho

    @property
    def group_labels(self):
        return self._result.params.group_labels

    @property
    def backend_name(self) -> str:
        return self._result.backend_name

    @property
    def timing(self):
        return self._result.timing

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._result.warnings

    def summary(self) -> str:
        """R-style summary of log-rank test."""
        lines = []
        lines.append("Call: survdiff()")
        lines.append("")

        # Group table
        lines.append(f"  {'':>12s}  {'N':>6s}  {'Observed':>10s}  {'Expected':>10s}  {'(O-E)^2/E':>10s}")
        for i in range(self.n_groups):
            oe = ((self.observed[i] - self.expected[i]) ** 2
                  / self.expected[i]) if self.expected[i] > 0 else 0
            label = str(self.group_labels[i])
            lines.append(
                f"  {label:>12s}  {self.n_per_group[i]:6.0f}  "
                f"{self.observed[i]:10.1f}  {self.expected[i]:10.1f}  "
                f"{oe:10.3f}"
            )

        lines.append("")
        lines.append(
            f"  Chisq= {self.statistic:.4f} on {self.df} degrees of freedom, "
            f"p= {self.p_value:.4g}"
        )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"LogRankSolution(chisq={self.statistic:.4f}, "
            f"df={self.df}, p={self.p_value:.4g})"
        )


class CoxSolution(SolutionReprMixin):
    """Cox proportional hazards solution.

    Properties mirror R's coxph() output.
    """

    __slots__ = ('_result', '_names', '_design')

    def __init__(self, _result: Result[CoxParams], _names: tuple[str, ...] | None = None,
                 _design=None) -> None:
        self._result = _result
        self._names = _names
        # The fitted SurvivalDesign, retained for post-fit diagnostics that
        # need the data (cox_zph). Same pattern as GLMSolution._design.
        self._design = _design

    @property
    def coefficients(self):
        return self._result.params.coefficients

    @property
    def coef(self) -> dict[str, float]:
        """Named coefficient mapping (like R's coef())."""
        names = self._names or tuple(f"x{i}" for i in range(len(self.coefficients)))
        return dict(zip(names, self.coefficients.tolist()))

    @property
    def hr(self) -> dict[str, float]:
        """Named hazard ratio mapping (exp(coef))."""
        names = self._names or tuple(f"x{i}" for i in range(len(self.hazard_ratios)))
        return dict(zip(names, self.hazard_ratios.tolist()))

    @property
    def hazard_ratios(self):
        return self._result.params.hazard_ratios

    @property
    def standard_errors(self):
        return self._result.params.standard_errors

    @property
    def z_values(self):
        return self._result.params.z_values

    @property
    def p_values(self):
        return self._result.params.p_values

    @property
    def loglik(self):
        return self._result.params.loglik

    @property
    def concordance(self) -> float:
        return self._result.params.concordance

    @property
    def n_events(self) -> int:
        return self._result.params.n_events

    @property
    def n_observations(self) -> int:
        return self._result.params.n_observations

    @property
    def n_iter(self) -> int:
        return self._result.params.n_iter

    @property
    def converged(self) -> bool:
        return self._result.params.converged

    @property
    def ties(self) -> str:
        return self._result.params.ties

    @property
    def conf_level(self) -> float:
        """Confidence level for ``conf_int`` (default 0.95)."""
        return self._result.params.conf_level

    @property
    def n_strata(self) -> int:
        """Number of strata (1 for an unstratified fit)."""
        return self._result.params.n_strata

    @property
    def robust(self) -> bool:
        """Whether ``standard_errors`` are the sandwich (robust) estimator."""
        return self._result.params.robust

    @property
    def naive_standard_errors(self):
        """Model-based SEs when ``robust`` — else the same as
        ``standard_errors``."""
        naive = self._result.params.naive_standard_errors
        return naive if naive is not None else self.standard_errors

    @property
    def conf_int(self):
        """Wald confidence intervals for the coefficients, shape (p, 2).

        ``coef ± z * se`` on the coefficient (log-hazard-ratio) scale, where
        ``z`` is the normal quantile for ``conf_level`` (Cox inference is
        asymptotic-normal). ``exp(conf_int)`` gives hazard-ratio intervals.
        """
        import numpy as np
        from scipy import stats

        z = stats.norm.ppf((1.0 + self.conf_level) / 2.0)
        coef = self.coefficients
        se = self.standard_errors
        return np.column_stack([coef - z * se, coef + z * se])

    @property
    def backend_name(self) -> str:
        return self._result.backend_name

    @property
    def timing(self):
        return self._result.timing

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._result.warnings

    def summary(self) -> str:
        """R-style summary of Cox PH fit."""
        import numpy as np

        p = len(self.coefficients)
        names = self._names or tuple(f"x{i}" for i in range(p))
        max_name_len = max(len(n) for n in names)
        col_w = max(max_name_len, 10)

        lines = []
        lines.append("Call: coxph()")
        lines.append("")
        strata_note = f", {self.n_strata} strata" if self.n_strata > 1 else ""
        lines.append(
            f"  n= {self.n_observations}, "
            f"number of events= {self.n_events}{strata_note}"
        )
        lines.append("")

        # Coefficient table. When robust, show both the model-based se(coef)
        # and the robust se (as R's summary.coxph does), with z/p from robust.
        se_label = "robust se" if self.robust else "se(coef)"
        header = (f"  {'':{col_w}s}  {'coef':>10s}  {'exp(coef)':>10s}  "
                  f"{'se(coef)':>10s}  ")
        if self.robust:
            header += f"{'robust se':>10s}  "
        header += f"{'z':>10s}  {'Pr(>|z|)':>12s}"
        lines.append(header)
        naive_se = self.naive_standard_errors
        for i, name in enumerate(names):
            row = (f"  {name:>{col_w}s}  {self.coefficients[i]:10.6f}  "
                   f"{self.hazard_ratios[i]:10.6f}  "
                   f"{naive_se[i]:10.6f}  ")
            if self.robust:
                row += f"{self.standard_errors[i]:10.6f}  "
            row += f"{self.z_values[i]:10.4f}  {self.p_values[i]:12.4g}"
            lines.append(row)

        # HR confidence intervals (like R's summary.coxph), at conf_level.
        ci = np.exp(self.conf_int)                 # hazard-ratio scale
        lvl = f"{self.conf_level:.2f}".lstrip("0")  # e.g. ".95"
        lines.append("")
        lines.append(
            f"  {'':{col_w}s}  {'exp(coef)':>10s}  {'exp(-coef)':>10s}  "
            f"{'lower ' + lvl:>10s}  {'upper ' + lvl:>10s}"
        )
        for i, name in enumerate(names):
            lines.append(
                f"  {name:>{col_w}s}  {self.hazard_ratios[i]:10.4f}  "
                f"{1.0 / self.hazard_ratios[i]:10.4f}  "
                f"{ci[i, 0]:10.4f}  {ci[i, 1]:10.4f}"
            )

        lines.append("")
        lines.append(
            f"  Concordance= {self.concordance:.4f}"
        )
        lr_stat = 2 * (self.loglik[1] - self.loglik[0])
        lines.append(
            f"  Likelihood ratio test= {lr_stat:.4f} on {p} df"
        )
        lines.append(
            f"  n= {self.n_observations}, "
            f"number of events= {self.n_events}"
        )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"CoxSolution(n={self.n_observations}, "
            f"events={self.n_events}, "
            f"concordance={self.concordance:.4f})"
        )


class DiscreteTimeSolution(SolutionReprMixin):
    """Discrete-time survival model solution.

    Properties mirror the logistic regression on person-period data.
    """

    __slots__ = ('_result', '_names', '_conf_level')

    def __init__(self, _result: Result[DiscreteTimeParams],
                 _names: tuple[str, ...] | None = None,
                 _conf_level: float = 0.95) -> None:
        self._result = _result
        self._names = _names
        self._conf_level = _conf_level

    @property
    def coefficients(self):
        return self._result.params.coefficients

    @property
    def coef(self) -> dict[str, float]:
        """Named coefficient mapping."""
        names = self._names or tuple(f"x{i}" for i in range(len(self.coefficients)))
        return dict(zip(names, self.coefficients.tolist()))

    @property
    def hr(self) -> dict[str, float]:
        """Named hazard ratio mapping (exp(coef))."""
        names = self._names or tuple(f"x{i}" for i in range(len(self.hazard_ratios)))
        return dict(zip(names, self.hazard_ratios.tolist()))

    @property
    def standard_errors(self):
        return self._result.params.standard_errors

    @property
    def z_values(self):
        return self._result.params.z_values

    @property
    def p_values(self):
        return self._result.params.p_values

    @property
    def conf_level(self) -> float:
        """Confidence level for ``conf_int`` (default 0.95)."""
        return self._conf_level

    @property
    def conf_int(self):
        """Wald confidence intervals for the covariate coefficients, shape (p, 2).

        ``coef ± z * se`` on the (discrete-time log-hazard) coefficient scale,
        with the normal quantile for ``conf_level`` (the person-period logistic
        fit is asymptotic-normal). ``exp(conf_int)`` gives discrete-time
        hazard-ratio intervals.
        """
        import numpy as np
        from scipy import stats

        z = stats.norm.ppf((1.0 + self._conf_level) / 2.0)
        coef = self._result.params.coefficients
        se = self._result.params.standard_errors
        return np.column_stack([coef - z * se, coef + z * se])

    @property
    def hazard_ratios(self):
        return self._result.params.hazard_ratios

    @property
    def baseline_hazard(self):
        return self._result.params.baseline_hazard

    @property
    def interval_labels(self):
        return self._result.params.interval_labels

    @property
    def person_period_n(self) -> int:
        return self._result.params.person_period_n

    @property
    def n_intervals(self) -> int:
        return self._result.params.n_intervals

    @property
    def n_observations(self) -> int:
        return self._result.params.n_observations

    @property
    def n_events(self) -> int:
        return self._result.params.n_events

    @property
    def glm_deviance(self) -> float:
        return self._result.params.glm_deviance

    @property
    def glm_aic(self) -> float:
        return self._result.params.glm_aic

    @property
    def converged(self) -> bool:
        """Whether the person-period binomial IRLS converged.

        ``False`` means the underlying logistic fit hit the iteration cap (or,
        for an all-censored design with no events, that no fit ran) — treat the
        coefficients as unreliable. Mirrors ``CoxSolution.converged``.
        """
        return self._result.params.converged

    @property
    def n_iter(self) -> int:
        """IRLS iterations used by the person-period binomial GLM.

        Mirrors ``CoxSolution.n_iter``. A value at the solver's iteration cap
        alongside ``converged == False`` indicates a fit that did not settle.
        """
        return self._result.params.n_iter

    @property
    def backend_name(self) -> str:
        return self._result.backend_name

    @property
    def timing(self):
        return self._result.timing

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._result.warnings

    def summary(self) -> str:
        """Summary of discrete-time survival model."""
        lines = []
        lines.append("Call: discrete_time()")
        lines.append("")
        lines.append(
            f"  n={self.n_observations}, events={self.n_events}, "
            f"intervals={self.n_intervals}, "
            f"person-period rows={self.person_period_n}"
        )
        lines.append("")

        p = len(self.coefficients)
        names = self._names or tuple(f"x{i}" for i in range(p))
        max_name_len = max(len(n) for n in names)
        col_w = max(max_name_len, 10)

        lines.append(
            f"  {'':{col_w}s}  {'coef':>10s}  {'exp(coef)':>10s}  "
            f"{'se(coef)':>10s}  {'z':>10s}  {'Pr(>|z|)':>12s}"
        )
        for i, name in enumerate(names):
            lines.append(
                f"  {name:>{col_w}s}  {self.coefficients[i]:10.6f}  "
                f"{self.hazard_ratios[i]:10.6f}  "
                f"{self.standard_errors[i]:10.6f}  "
                f"{self.z_values[i]:10.4f}  "
                f"{self.p_values[i]:12.4g}"
            )

        lines.append("")
        lines.append(f"  Deviance: {self.glm_deviance:.4f}")
        lines.append(f"  AIC: {self.glm_aic:.4f}")
        lines.append(f"  Converged: {self.converged} ({self.n_iter} iterations)")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DiscreteTimeSolution(n={self.n_observations}, "
            f"events={self.n_events}, "
            f"intervals={self.n_intervals})"
        )
