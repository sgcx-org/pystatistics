"""
Hypothesis test solution types.

HTestSolution wraps Result[HTestParams] and provides R's print.htest format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result
from pystatistics.hypothesis._common import HTestParams

if TYPE_CHECKING:
    from pystatistics.hypothesis.design import HypothesisDesign


@dataclass
class HTestSolution:
    """
    User-facing hypothesis test results.

    Wraps Result[HTestParams] and provides R's print.htest output format
    via summary(). All standard htest fields are available as properties.
    """
    _result: Result[HTestParams]
    _design: 'HypothesisDesign | None'

    # --- Standard htest fields ---

    @property
    def statistic(self) -> float | None:
        """Test statistic value."""
        return self._result.params.statistic

    @property
    def statistic_name(self) -> str:
        """Name of the test statistic (e.g. 't', 'X-squared')."""
        return self._result.params.statistic_name

    @property
    def parameter(self) -> dict[str, float] | None:
        """Distribution parameters (e.g. {'df': 9})."""
        return self._result.params.parameter

    @property
    def p_value(self) -> float:
        """p-value of the test."""
        return self._result.params.p_value

    @property
    def conf_int(self) -> NDArray[np.floating[Any]] | None:
        """Confidence interval, shape (2,)."""
        return self._result.params.conf_int

    @property
    def conf_level(self) -> float:
        """Confidence level."""
        return self._result.params.conf_level

    @property
    def estimate(self) -> dict[str, float] | None:
        """Point estimate(s)."""
        return self._result.params.estimate

    @property
    def null_value(self) -> dict[str, float] | None:
        """Hypothesized value under H0."""
        return self._result.params.null_value

    @property
    def alternative(self) -> str:
        """Alternative hypothesis direction."""
        return self._result.params.alternative

    @property
    def method(self) -> str:
        """Human-readable method name."""
        return self._result.params.method

    @property
    def data_name(self) -> str:
        """Description of the data."""
        return self._result.params.data_name

    # --- Test-specific extras ---

    @property
    def extras(self) -> dict[str, Any] | None:
        """Test-specific additional outputs."""
        return self._result.params.extras

    @property
    def observed(self) -> NDArray | None:
        """For chisq_test: observed counts."""
        e = self._result.params.extras
        return e.get('observed') if e else None

    @property
    def expected(self) -> NDArray | None:
        """For chisq_test: expected counts under H0."""
        e = self._result.params.extras
        return e.get('expected') if e else None

    @property
    def residuals(self) -> NDArray | None:
        """For chisq_test: Pearson residuals."""
        e = self._result.params.extras
        return e.get('residuals') if e else None

    @property
    def stdres(self) -> NDArray | None:
        """For chisq_test: standardized residuals."""
        e = self._result.params.extras
        return e.get('stdres') if e else None

    # --- Metadata ---

    @property
    def info(self) -> dict[str, Any]:
        return self._result.info

    @property
    def timing(self) -> dict[str, float] | None:
        return self._result.timing

    @property
    def backend_name(self) -> str:
        return self._result.backend_name

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._result.warnings

    # --- Formatting ---

    def summary(self) -> str:
        """
        Format as R's print.htest output.

        Produces output like:
            Welch Two Sample t-test

        data:  x and y
        t = 2.2345, df = 17.43, p-value = 0.03891
        alternative hypothesis: true difference in means is not equal to 0
        95 percent confidence interval:
         0.1234567  4.5678901
        sample estimates:
        mean of x mean of y
         5.123456  2.789012
        """
        p = self._result.params
        lines = []

        # Method title (centered-ish, like R)
        lines.append(f"\t{p.method}")
        lines.append("")

        # Data line
        lines.append(f"data:  {p.data_name}")

        # Statistic line: "t = 2.2345, df = 17.43, p-value = 0.03891"
        parts = []
        if p.statistic is not None:
            parts.append(f"{p.statistic_name} = {p.statistic:.5g}")
        if p.parameter is not None:
            for name, val in p.parameter.items():
                parts.append(f"{name} = {val:.5g}")
        parts.append(f"p-value = {_format_pvalue(p.p_value)}")
        lines.append(", ".join(parts))

        # Alternative hypothesis
        if p.null_value is not None and p.null_value:
            nv_name = next(iter(p.null_value.keys()))
            nv_val = next(iter(p.null_value.values()))
            if p.alternative == "two.sided":
                lines.append(
                    f"alternative hypothesis: true {nv_name} "
                    f"is not equal to {nv_val:g}"
                )
            elif p.alternative == "less":
                lines.append(
                    f"alternative hypothesis: true {nv_name} "
                    f"is less than {nv_val:g}"
                )
            elif p.alternative == "greater":
                lines.append(
                    f"alternative hypothesis: true {nv_name} "
                    f"is greater than {nv_val:g}"
                )

        # Confidence interval
        if p.conf_int is not None:
            pct = int(p.conf_level * 100)
            lines.append(f"{pct} percent confidence interval:")
            lo, hi = p.conf_int
            lines.append(f" {_format_number(lo)}  {_format_number(hi)}")

        # Estimates
        if p.estimate is not None:
            lines.append("sample estimates:")
            names = list(p.estimate.keys())
            vals = list(p.estimate.values())
            lines.append(" ".join(f"{n:>14s}" for n in names))
            lines.append(" ".join(f"{v:14.7g}" for v in vals))

        lines.append("")
        return "\n".join(lines)

    def __repr__(self) -> str:
        p = self._result.params
        stat_str = ""
        if p.statistic is not None:
            stat_str = f", {p.statistic_name}={p.statistic:.4g}"
        return (
            f"HTestSolution(method={p.method!r}{stat_str}, "
            f"p_value={p.p_value:.4g})"
        )


def _format_pvalue(p: float) -> str:
    """Format p-value like R does."""
    if p < 2.2e-16:
        return "< 2.2e-16"
    if p < 0.001:
        return f"{p:.4e}"
    return f"{p:.4g}"


def _format_number(x: float) -> str:
    """Format a number, handling infinity."""
    if np.isinf(x):
        return "-Inf" if x < 0 else "Inf"
    return f"{x:.7g}"
