"""
Solution wrappers for Monte Carlo results.

BootstrapSolution and PermutationSolution wrap Result[P] and provide
convenient accessors and R-style summary output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result
from pystatistics.montecarlo._common import BootParams, PermutationParams

if TYPE_CHECKING:
    from pystatistics.montecarlo.design import BootstrapDesign, PermutationDesign


@dataclass
class BootstrapSolution:
    """
    User-facing bootstrap results.

    Matches R's boot object output: t0, t, bias, SE, plus CI if computed.
    summary() produces R's print.boot format.
    """
    _result: Result[BootParams]
    _design: 'BootstrapDesign'

    # --- Core boot fields ---

    @property
    def t0(self) -> NDArray[np.floating[Any]]:
        """Observed statistic(s) on original data, shape (k,)."""
        return self._result.params.t0

    @property
    def t(self) -> NDArray[np.floating[Any]]:
        """Bootstrap replicates, shape (R, k)."""
        return self._result.params.t

    @property
    def R(self) -> int:
        """Number of bootstrap replicates."""
        return self._result.params.R

    @property
    def bias(self) -> NDArray[np.floating[Any]]:
        """Bootstrap bias estimate: mean(t) - t0, shape (k,)."""
        return self._result.params.bias

    @property
    def se(self) -> NDArray[np.floating[Any]]:
        """Bootstrap standard error: sd(t), shape (k,)."""
        return self._result.params.se

    @property
    def ci(self) -> dict[str, NDArray] | None:
        """Confidence intervals keyed by type, or None if not computed."""
        return self._result.params.ci

    @property
    def ci_conf_level(self) -> float | None:
        """Confidence level used for CI computation."""
        return self._result.params.ci_conf_level

    # --- Metadata ---

    @property
    def data(self) -> NDArray:
        """Original data."""
        return self._design.data

    @property
    def sim(self) -> str:
        """Simulation type used."""
        return self._design.sim

    @property
    def seed(self) -> int | None:
        """Random seed used."""
        return self._design.seed

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

    # --- Display ---

    def summary(self) -> str:
        """
        R-style print.boot output.

        Produces:
            ORDINARY NONPARAMETRIC BOOTSTRAP

            Bootstrap Statistics :
                original       bias    std. error
            t1*  5.12345    0.01234     0.56789
            t2*  3.45678   -0.00567     0.34567
        """
        lines = []

        # Title
        sim_name = {
            "ordinary": "ORDINARY NONPARAMETRIC BOOTSTRAP",
            "balanced": "BALANCED BOOTSTRAP",
            "parametric": "PARAMETRIC BOOTSTRAP",
        }.get(self.sim, "BOOTSTRAP")
        lines.append(f"\n{sim_name}\n")

        # Call info
        lines.append(
            f"Call: boot(data, statistic, R={self.R}, "
            f"sim=\"{self.sim}\")"
        )
        lines.append("")

        # Statistics table
        lines.append("Bootstrap Statistics :")

        k = len(self.t0)
        header = f"{'':>8s} {'original':>14s} {'bias':>14s} {'std. error':>14s}"
        lines.append(header)

        for i in range(k):
            label = f"t{i+1}*"
            lines.append(
                f"{label:>8s} {self.t0[i]:14.5f} {self.bias[i]:14.5f} "
                f"{self.se[i]:14.5f}"
            )

        # CI if available
        if self.ci is not None:
            lines.append("")
            conf_pct = int((self.ci_conf_level or 0.95) * 100)
            for ci_type, ci_vals in self.ci.items():
                lines.append(f"{conf_pct}% {ci_type} CI:")
                for i in range(k):
                    label = f"t{i+1}*"
                    lines.append(
                        f"  {label}: ({ci_vals[i, 0]:.5f}, "
                        f"{ci_vals[i, 1]:.5f})"
                    )

        return "\n".join(lines)

    def __repr__(self) -> str:
        k = len(self.t0)
        return (
            f"BootstrapSolution(R={self.R}, k={k}, "
            f"sim={self.sim!r}, backend={self.backend_name!r})"
        )


@dataclass
class PermutationSolution:
    """
    User-facing permutation test results.

    Provides observed statistic, permutation distribution, and p-value.
    """
    _result: Result[PermutationParams]
    _design: 'PermutationDesign'

    # --- Core fields ---

    @property
    def observed_stat(self) -> float:
        """Test statistic on original (unpermuted) data."""
        return self._result.params.observed_stat

    @property
    def perm_stats(self) -> NDArray[np.floating[Any]]:
        """Permutation distribution, shape (R,)."""
        return self._result.params.perm_stats

    @property
    def p_value(self) -> float:
        """Permutation p-value with Phipson-Smyth correction."""
        return self._result.params.p_value

    @property
    def R(self) -> int:
        """Number of permutations."""
        return self._result.params.R

    @property
    def alternative(self) -> str:
        """Alternative hypothesis direction."""
        return self._result.params.alternative

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

    # --- Display ---

    def summary(self) -> str:
        """Permutation test summary."""
        lines = [
            "\nPERMUTATION TEST",
            "",
            f"Number of permutations: {self.R}",
            f"Observed statistic: {self.observed_stat:.6g}",
            f"p-value ({self.alternative}): {self.p_value:.4g}",
            "",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PermutationSolution(R={self.R}, "
            f"observed={self.observed_stat:.4g}, "
            f"p_value={self.p_value:.4g})"
        )
