"""
Common result types for multivariate analysis.

Frozen dataclasses following the PyStatistics pattern:
- PCAResult: principal component analysis results
- FactorResult: factor analysis results
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PCAResult:
    """Result from principal component analysis.

    Matches the structure of R's ``prcomp()`` return value.

    Attributes:
        sdev: Standard deviations of principal components (length min(n, p)).
        rotation: Loadings matrix (p x n_components) -- columns are eigenvectors.
        center: Column means used for centering (length p).
        scale: Column SDs used for scaling (None if scale=False).
        x: Scores matrix (n x n_components).
        n_obs: Number of observations.
        n_vars: Number of variables.
        var_names: Variable names, or None.
    """

    sdev: NDArray
    rotation: NDArray
    center: NDArray
    scale: NDArray | None
    x: NDArray
    n_obs: int
    n_vars: int
    var_names: tuple[str, ...] | None

    @property
    def explained_variance_ratio(self) -> NDArray:
        """Proportion of variance explained by each component."""
        variances = self.sdev ** 2
        return variances / np.sum(variances)

    @property
    def cumulative_variance_ratio(self) -> NDArray:
        """Cumulative proportion of variance explained."""
        return np.cumsum(self.explained_variance_ratio)

    def summary(self) -> str:
        """R-style summary matching ``summary(prcomp(...))``.

        Returns:
            Formatted string with standard deviations, proportion of variance,
            and cumulative proportion for each component.
        """
        n_comp = len(self.sdev)
        labels = [f"PC{i+1}" for i in range(n_comp)]

        header = "Importance of components:"
        col_width = max(12, max(len(lab) for lab in labels) + 2)

        def _fmt_row(name: str, values: NDArray) -> str:
            parts = [f"{name:<30s}"]
            for v in values:
                parts.append(f"{v:>{col_width}.6f}")
            return "".join(parts)

        rows = [
            header,
            _fmt_row("Standard deviation", self.sdev),
            _fmt_row("Proportion of Variance", self.explained_variance_ratio),
            _fmt_row("Cumulative Proportion", self.cumulative_variance_ratio),
        ]
        # Column labels
        label_row = " " * 30 + "".join(f"{lab:>{col_width}s}" for lab in labels)
        rows.insert(1, label_row)
        return "\n".join(rows)


@dataclass(frozen=True)
class FactorResult:
    """Result from factor analysis.

    Matches the structure of R's ``factanal()`` return value.

    Attributes:
        loadings: Rotated loadings matrix (p x n_factors).
        uniquenesses: Uniqueness for each variable (length p).
        communalities: 1 - uniquenesses.
        rotation_matrix: Rotation matrix, or None if no rotation.
        chi_sq: Goodness-of-fit chi-squared statistic, or None.
        p_value: p-value for chi-sq test, or None.
        dof: Degrees of freedom.
        n_factors: Number of factors extracted.
        n_obs: Number of observations.
        n_vars: Number of variables.
        var_names: Variable names, or None.
        method: Estimation method (e.g. 'ml').
        rotation_method: Rotation method (e.g. 'varimax', 'promax', 'none').
        converged: Whether the optimisation converged.
        n_iter: Number of iterations used.
        objective: Final objective function value.
    """

    loadings: NDArray
    uniquenesses: NDArray
    communalities: NDArray
    rotation_matrix: NDArray | None
    chi_sq: float | None
    p_value: float | None
    dof: int
    n_factors: int
    n_obs: int
    n_vars: int
    var_names: tuple[str, ...] | None
    method: str
    rotation_method: str
    converged: bool
    n_iter: int
    objective: float

    def summary(self) -> str:
        """R-style summary matching ``print(factanal(...))``.

        Returns:
            Formatted string with loadings, uniquenesses, SS loadings,
            proportion and cumulative variance, and the chi-squared test.
        """
        p, m = self.loadings.shape
        names = list(self.var_names) if self.var_names else [f"V{i+1}" for i in range(p)]
        factor_labels = [f"Factor{j+1}" for j in range(m)]
        name_width = max(len(n) for n in names) + 2
        col_width = 10

        lines: list[str] = []
        lines.append(f"Factor analysis with {m} factor(s), method: {self.method}")
        lines.append("")
        lines.append("Loadings:")

        # Header
        header = " " * name_width + "".join(f"{fl:>{col_width}s}" for fl in factor_labels)
        lines.append(header)

        # Loadings rows (suppress small values like R does)
        for i, name in enumerate(names):
            row = f"{name:<{name_width}s}"
            for j in range(m):
                val = self.loadings[i, j]
                if abs(val) < 0.1:
                    row += " " * col_width
                else:
                    row += f"{val:>{col_width}.3f}"
            row += f"{self.uniquenesses[i]:>{col_width}.3f}"
            lines.append(row)

        # Uniquenesses header in last column
        lines[3] = lines[3] + f"{'Uniquenesses':>{col_width + 4}s}"

        # SS loadings
        ss_loadings = np.sum(self.loadings ** 2, axis=0)
        prop_var = ss_loadings / p
        cum_var = np.cumsum(prop_var)

        lines.append("")
        row_ss = " " * name_width + "".join(f"{v:>{col_width}.3f}" for v in ss_loadings)
        row_pv = " " * name_width + "".join(f"{v:>{col_width}.3f}" for v in prop_var)
        row_cv = " " * name_width + "".join(f"{v:>{col_width}.3f}" for v in cum_var)
        lines.append(f"{'SS loadings':<{name_width}s}" + row_ss[name_width:])
        lines.append(f"{'Proportion Var':<{name_width}s}" + row_pv[name_width:])
        lines.append(f"{'Cumulative Var':<{name_width}s}" + row_cv[name_width:])

        # Chi-squared test
        if self.chi_sq is not None:
            lines.append("")
            lines.append(
                f"Test of the hypothesis that {m} factor(s) are sufficient."
            )
            lines.append(
                f"The chi square statistic is {self.chi_sq:.2f} "
                f"on {self.dof} degree(s) of freedom."
            )
            if self.p_value is not None:
                lines.append(f"The p-value is {self.p_value:.4g}")

        return "\n".join(lines)
