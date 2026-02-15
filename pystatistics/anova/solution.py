"""
User-facing ANOVA solution types.

Each solution wraps a Result[Params] and provides convenient accessors,
formatted summary output (matching R conventions), and the ANOVA table.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pystatistics.core.result import Result
from pystatistics.anova._common import (
    AnovaParams,
    AnovaRMParams,
    AnovaTableRow,
    AnovaRMTableRow,
    LeveneParams,
    PostHocParams,
    PostHocComparison,
    SphericitySummary,
)


# =====================================================================
# AnovaSolution  (one-way, factorial, ANCOVA)
# =====================================================================


@dataclass
class AnovaSolution:
    """
    User-facing result for between-subjects ANOVA.

    Produced by anova_oneway() and anova().
    """
    _result: Result[AnovaParams]

    @property
    def table(self) -> tuple[AnovaTableRow, ...]:
        """ANOVA table (list of rows: term, df, SS, MS, F, p)."""
        return self._result.params.table

    @property
    def ss_type(self) -> int:
        return self._result.params.ss_type

    @property
    def n_obs(self) -> int:
        return self._result.params.n_obs

    @property
    def grand_mean(self) -> float:
        return self._result.params.grand_mean

    @property
    def residual_df(self) -> int:
        return self._result.params.residual_df

    @property
    def residual_ss(self) -> float:
        return self._result.params.residual_ss

    @property
    def residual_ms(self) -> float:
        return self._result.params.residual_ms

    @property
    def eta_squared(self) -> dict[str, float]:
        return self._result.params.eta_squared

    @property
    def partial_eta_squared(self) -> dict[str, float]:
        return self._result.params.partial_eta_squared

    @property
    def group_means(self) -> dict[str, dict[str, float]]:
        return self._result.params.group_means

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

    def summary(self) -> str:
        """Generate R-style ANOVA summary table."""
        lines = [
            f"Analysis of Variance Table (Type {self.ss_type} SS)",
            "=" * 72,
            f"Observations: {self.n_obs}",
            "",
            f"{'Source':<20} {'Df':>6} {'Sum Sq':>14} {'Mean Sq':>14} {'F value':>10} {'Pr(>F)':>12}",
            "-" * 72,
        ]

        for row in self.table:
            if row.f_value is not None:
                sig = _significance_stars(row.p_value)
                lines.append(
                    f"{row.term:<20} {row.df:>6} {row.sum_sq:>14.4f} "
                    f"{row.mean_sq:>14.4f} {row.f_value:>10.4f} "
                    f"{row.p_value:>12.4e} {sig}"
                )
            else:
                lines.append(
                    f"{row.term:<20} {row.df:>6} {row.sum_sq:>14.4f} "
                    f"{row.mean_sq:>14.4f}"
                )

        lines.append("-" * 72)
        lines.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

        # Effect sizes
        if self.eta_squared:
            lines.append("")
            lines.append("Effect sizes:")
            for term in self.eta_squared:
                eta = self.eta_squared[term]
                partial = self.partial_eta_squared.get(term, eta)
                lines.append(
                    f"  {term}: eta^2 = {eta:.4f}, partial eta^2 = {partial:.4f}"
                )

        return "\n".join(lines)

    def __repr__(self) -> str:
        terms = [row.term for row in self.table if row.term != 'Residuals']
        return (
            f"AnovaSolution(type={self.ss_type}, n={self.n_obs}, "
            f"terms={terms})"
        )


# =====================================================================
# AnovaRMSolution  (repeated measures)
# =====================================================================


@dataclass
class AnovaRMSolution:
    """
    User-facing result for repeated-measures ANOVA.

    Produced by anova_rm().
    """
    _result: Result[AnovaRMParams]

    @property
    def table(self) -> tuple[AnovaRMTableRow, ...]:
        return self._result.params.table

    @property
    def n_subjects(self) -> int:
        return self._result.params.n_subjects

    @property
    def n_obs(self) -> int:
        return self._result.params.n_obs

    @property
    def sphericity(self) -> tuple[SphericitySummary, ...]:
        return self._result.params.sphericity

    @property
    def correction(self) -> str:
        return self._result.params.correction

    @property
    def eta_squared(self) -> dict[str, float]:
        return self._result.params.eta_squared

    @property
    def partial_eta_squared(self) -> dict[str, float]:
        return self._result.params.partial_eta_squared

    @property
    def info(self) -> dict[str, Any]:
        return self._result.info

    @property
    def timing(self) -> dict[str, float] | None:
        return self._result.timing

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._result.warnings

    def summary(self) -> str:
        """Generate R-style repeated-measures ANOVA summary."""
        lines = [
            "Repeated-Measures ANOVA",
            "=" * 72,
            f"Subjects: {self.n_subjects}",
            f"Observations: {self.n_obs}",
            f"Correction: {self.correction}",
            "",
            f"{'Source':<20} {'Df':>8} {'Sum Sq':>14} {'Mean Sq':>14} {'F':>10} {'p':>12}",
            "-" * 72,
        ]

        for row in self.table:
            if row.f_value is not None:
                sig = _significance_stars(row.p_value)
                lines.append(
                    f"{row.term:<20} {row.df:>8.2f} {row.sum_sq:>14.4f} "
                    f"{row.mean_sq:>14.4f} {row.f_value:>10.4f} "
                    f"{row.p_value:>12.4e} {sig}"
                )
            else:
                lines.append(
                    f"{row.term:<20} {row.df:>8.2f} {row.sum_sq:>14.4f} "
                    f"{row.mean_sq:>14.4f}"
                )

        # Sphericity
        if self.sphericity:
            lines.append("")
            lines.append("Mauchly's Test of Sphericity:")
            lines.append(f"  {'Factor':<16} {'W':>8} {'p':>12} {'GG eps':>10} {'HF eps':>10}")
            for s in self.sphericity:
                lines.append(
                    f"  {s.factor:<16} {s.mauchly_w:>8.4f} {s.p_value:>12.4e} "
                    f"{s.gg_epsilon:>10.4f} {s.hf_epsilon:>10.4f}"
                )

            # Corrected p-values
            lines.append("")
            lines.append("Corrected p-values:")
            for row in self.table:
                if row.gg_p_value is not None:
                    lines.append(
                        f"  {row.term}: GG p = {row.gg_p_value:.4e}, "
                        f"HF p = {row.hf_p_value:.4e}"
                    )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"AnovaRMSolution(n_subjects={self.n_subjects}, "
            f"n_obs={self.n_obs}, correction={self.correction!r})"
        )


# =====================================================================
# LeveneSolution
# =====================================================================


@dataclass
class LeveneSolution:
    """
    User-facing result for Levene's test.

    Produced by levene_test().
    """
    _result: Result[LeveneParams]

    @property
    def f_value(self) -> float:
        return self._result.params.f_value

    @property
    def p_value(self) -> float:
        return self._result.params.p_value

    @property
    def df_between(self) -> int:
        return self._result.params.df_between

    @property
    def df_within(self) -> int:
        return self._result.params.df_within

    @property
    def center(self) -> str:
        return self._result.params.center

    @property
    def group_vars(self) -> dict[str, float]:
        return self._result.params.group_vars

    def summary(self) -> str:
        variant = "Brown-Forsythe" if self.center == 'median' else "Levene"
        lines = [
            f"{variant} Test for Homogeneity of Variances",
            "=" * 50,
            f"F({self.df_between}, {self.df_within}) = {self.f_value:.4f}, "
            f"p = {self.p_value:.4e}",
            "",
            f"Center: {self.center}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"LeveneSolution(F={self.f_value:.4f}, "
            f"p={self.p_value:.4e}, center={self.center!r})"
        )


# =====================================================================
# PostHocSolution
# =====================================================================


@dataclass
class PostHocSolution:
    """
    User-facing result for post-hoc comparisons.

    Produced by anova_posthoc().
    """
    _result: Result[PostHocParams]

    @property
    def method(self) -> str:
        return self._result.params.method

    @property
    def comparisons(self) -> tuple[PostHocComparison, ...]:
        return self._result.params.comparisons

    @property
    def conf_level(self) -> float:
        return self._result.params.conf_level

    @property
    def factor(self) -> str:
        return self._result.params.factor

    def summary(self) -> str:
        method_names = {
            'tukey': "Tukey HSD",
            'bonferroni': "Bonferroni Pairwise Comparisons",
            'dunnett': "Dunnett's Test",
        }
        title = method_names.get(self.method, self.method)

        lines = [
            title,
            "=" * 72,
            f"Factor: {self.factor}",
            f"Confidence level: {self.conf_level:.0%}",
            "",
            f"{'Comparison':<25} {'diff':>10} {'lwr':>12} {'upr':>12} {'p adj':>12}",
            "-" * 72,
        ]

        for c in self.comparisons:
            label = f"{c.group2}-{c.group1}"
            sig = _significance_stars(c.p_value)
            lines.append(
                f"{label:<25} {c.diff:>10.4f} {c.ci_lower:>12.4f} "
                f"{c.ci_upper:>12.4f} {c.p_value:>12.4e} {sig}"
            )

        lines.append("-" * 72)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PostHocSolution(method={self.method!r}, "
            f"n_comparisons={len(self.comparisons)})"
        )


# =====================================================================
# Helpers
# =====================================================================


def _significance_stars(p: float | None) -> str:
    """Return significance stars for a p-value."""
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""
