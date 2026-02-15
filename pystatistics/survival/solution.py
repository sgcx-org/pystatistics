"""
Solution wrappers for survival analysis results.

Each Solution wraps a Result[Params] and exposes user-friendly properties
with R-style summary() methods.
"""

from __future__ import annotations

from pystatistics.core.result import Result
from pystatistics.survival._common import (
    CoxParams,
    DiscreteTimeParams,
    KMParams,
    LogRankParams,
)


class KMSolution:
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
    def ci_lower(self):
        """Lower confidence bound for S(t)."""
        return self._result.params.ci_lower

    @property
    def ci_upper(self):
        """Upper confidence bound for S(t)."""
        return self._result.params.ci_upper

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
        """Median survival time (smallest t where S(t) <= 0.5)."""
        if len(self.survival) == 0:
            return None
        idx = self.survival <= 0.5
        if not idx.any():
            return None
        return float(self.time[idx][0])

    @property
    def backend_name(self) -> str:
        return self._result.backend_name

    @property
    def timing(self):
        return self._result.timing

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
            f"{'lower {ci_pct}%':>10s}  {'upper {ci_pct}%':>10s}"
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


class LogRankSolution:
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


class CoxSolution:
    """Cox proportional hazards solution.

    Properties mirror R's coxph() output.
    """

    __slots__ = ('_result',)

    def __init__(self, _result: Result[CoxParams]) -> None:
        self._result = _result

    @property
    def coefficients(self):
        return self._result.params.coefficients

    @property
    def hazard_ratios(self):
        return self._result.params.hazard_ratios

    @property
    def standard_errors(self):
        return self._result.params.standard_errors

    @property
    def z_statistics(self):
        return self._result.params.z_statistics

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
    def backend_name(self) -> str:
        return self._result.backend_name

    @property
    def timing(self):
        return self._result.timing

    def summary(self) -> str:
        """R-style summary of Cox PH fit."""
        lines = []
        lines.append("Call: coxph()")
        lines.append("")
        lines.append(
            f"  n= {self.n_observations}, "
            f"number of events= {self.n_events}"
        )
        lines.append("")

        # Coefficient table
        lines.append(
            f"  {'':>10s}  {'coef':>10s}  {'exp(coef)':>10s}  "
            f"{'se(coef)':>10s}  {'z':>10s}  {'Pr(>|z|)':>12s}"
        )
        p = len(self.coefficients)
        for i in range(p):
            name = f"x{i}"
            lines.append(
                f"  {name:>10s}  {self.coefficients[i]:10.6f}  "
                f"{self.hazard_ratios[i]:10.6f}  "
                f"{self.standard_errors[i]:10.6f}  "
                f"{self.z_statistics[i]:10.4f}  "
                f"{self.p_values[i]:12.4g}"
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


class DiscreteTimeSolution:
    """Discrete-time survival model solution.

    Properties mirror the logistic regression on person-period data.
    """

    __slots__ = ('_result',)

    def __init__(self, _result: Result[DiscreteTimeParams]) -> None:
        self._result = _result

    @property
    def coefficients(self):
        return self._result.params.coefficients

    @property
    def standard_errors(self):
        return self._result.params.standard_errors

    @property
    def z_statistics(self):
        return self._result.params.z_statistics

    @property
    def p_values(self):
        return self._result.params.p_values

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
    def backend_name(self) -> str:
        return self._result.backend_name

    @property
    def timing(self):
        return self._result.timing

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

        lines.append(
            f"  {'':>10s}  {'coef':>10s}  {'exp(coef)':>10s}  "
            f"{'se(coef)':>10s}  {'z':>10s}  {'Pr(>|z|)':>12s}"
        )
        p = len(self.coefficients)
        for i in range(p):
            name = f"x{i}"
            lines.append(
                f"  {name:>10s}  {self.coefficients[i]:10.6f}  "
                f"{self.hazard_ratios[i]:10.6f}  "
                f"{self.standard_errors[i]:10.6f}  "
                f"{self.z_statistics[i]:10.4f}  "
                f"{self.p_values[i]:12.4g}"
            )

        lines.append("")
        lines.append(f"  Deviance: {self.glm_deviance:.4f}")
        lines.append(f"  AIC: {self.glm_aic:.4f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DiscreteTimeSolution(n={self.n_observations}, "
            f"events={self.n_events}, "
            f"intervals={self.n_intervals})"
        )
