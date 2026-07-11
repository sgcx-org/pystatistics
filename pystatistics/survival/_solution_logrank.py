"""
Log-rank test solution wrapper.

LogRankSolution wraps a Result[LogRankParams] and exposes user-friendly
properties with an R-style summary() method mirroring survdiff() output.
"""

from __future__ import annotations

from pystatistics.core.result import Result, SolutionReprMixin
from pystatistics.survival._common import LogRankParams


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
