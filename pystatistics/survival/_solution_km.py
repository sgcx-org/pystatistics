"""
Kaplan-Meier solution wrapper.

KMSolution wraps a Result[KMParams] and exposes user-friendly properties
with an R-style summary() method mirroring survfit() output.
"""

from __future__ import annotations

from pystatistics.core.result import Result, SolutionReprMixin
from pystatistics.survival._common import KMParams


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
