"""Result types for the low-rank / GRM mixed model (``grm_lmm``).

A GRM mixed model has a single variance component with a *low-rank* covariance
K = WW'/M (a genomic relatedness matrix, or any reduced-rank random effect).
Its result carries the fixed-effect table, the two variance components
(σ²_g genetic / σ²_e residual), the narrow-sense heritability h² =
σ²_g/(σ²_g+σ²_e), and the genetic-value BLUPs.

This is a *different model* from ``lmm``: it is not the lme4-style sparse-design
LMM and is never described as "LMM on a GPU". It exists for the genomics /
quantitative-genetics regime where the random effect is a dense low-rank factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from pystatistics.core.result import Result, SolutionReprMixin


@dataclass(frozen=True)
class GRMParams:
    """Parameter payload for a fitted low-rank / GRM mixed model."""
    # Fixed effects
    coefficients: NDArray
    coefficient_names: tuple[str, ...]
    se: NDArray
    z_values: NDArray                  # Wald z (asymptotic-normal reference)
    p_values: NDArray

    # Variance components
    var_genetic: float                 # σ²_g
    var_residual: float                # σ²_e
    variance_ratio: float              # γ = σ²_g / σ²_e
    heritability: float                # h² = σ²_g / (σ²_g + σ²_e)

    # Model fit
    log_likelihood: float
    reml: bool
    aic: float
    bic: float
    n_obs: int
    n_covariates: int
    rank: int                          # M, the rank of the low-rank factor

    # Convergence
    converged: bool
    n_iter: int

    # Random effects
    genetic_values: NDArray            # BLUP of g = Zu (n,)

    # Predictions
    fitted_values: NDArray
    residuals: NDArray

    # Internal
    theta: float                       # sqrt(variance ratio), the profiled param


class GRMSolution(SolutionReprMixin):
    """Solution wrapper for a fitted low-rank / GRM mixed model.

    Exposes the uniform accessors (``coefficients``, ``standard_errors``,
    ``z_values``, ``p_values``, ``conf_int``, ``fitted_values``, ``residuals``,
    ``converged``, ``n_iter``, ``backend_name``) plus the quantitative-genetics
    quantities (``heritability``, ``var_genetic``, ``var_residual``,
    ``variance_ratio``, ``genetic_values``).
    """

    def __init__(self, _result: Result[GRMParams], _conf_level: float = 0.95):
        self._result = _result
        self._conf_level = _conf_level

    @property
    def params(self) -> GRMParams:
        return self._result.params

    # --- Fixed effects ---

    @property
    def coefficients(self) -> NDArray:
        return self.params.coefficients

    @property
    def coef(self) -> dict[str, float]:
        return dict(zip(self.params.coefficient_names, self.params.coefficients))

    @property
    def standard_errors(self) -> NDArray:
        return self.params.se

    @property
    def z_values(self) -> NDArray:
        """Wald z-statistics for fixed effects (asymptotic-normal reference)."""
        return self.params.z_values

    @property
    def p_values(self) -> NDArray:
        return self.params.p_values

    @property
    def conf_level(self) -> float:
        return self._conf_level

    @property
    def conf_int(self) -> NDArray:
        """Wald confidence intervals for the fixed effects, shape (p, 2).

        ``β ± z * SE`` with the normal quantile for ``conf_level`` (GRM REML
        fixed-effect inference is asymptotic-normal).
        """
        z = stats.norm.ppf((1.0 + self._conf_level) / 2.0)
        coef = self.coefficients
        se = self.standard_errors
        return np.column_stack([coef - z * se, coef + z * se])

    # --- Variance components / quantitative genetics ---

    @property
    def heritability(self) -> float:
        """Narrow-sense heritability h² = σ²_g / (σ²_g + σ²_e)."""
        return self.params.heritability

    @property
    def var_genetic(self) -> float:
        """Genetic variance component σ²_g."""
        return self.params.var_genetic

    @property
    def var_residual(self) -> float:
        """Residual variance component σ²_e."""
        return self.params.var_residual

    @property
    def variance_ratio(self) -> float:
        """Variance ratio γ = σ²_g / σ²_e."""
        return self.params.variance_ratio

    @property
    def genetic_values(self) -> NDArray:
        """BLUP of the genetic values g (n,)."""
        return self.params.genetic_values

    # --- Model fit ---

    @property
    def log_likelihood(self) -> float:
        return self.params.log_likelihood

    @property
    def aic(self) -> float:
        return self.params.aic

    @property
    def bic(self) -> float:
        return self.params.bic

    @property
    def fitted_values(self) -> NDArray:
        return self.params.fitted_values

    @property
    def residuals(self) -> NDArray:
        return self.params.residuals

    @property
    def converged(self) -> bool:
        return self.params.converged

    @property
    def n_iter(self) -> int:
        return self.params.n_iter

    @property
    def backend_name(self) -> str:
        return self._result.backend_name

    # --- Display ---

    def summary(self) -> str:
        """R-style summary for the low-rank / GRM mixed model."""
        p = self.params
        method = 'REML' if p.reml else 'ML'
        lines = []
        lines.append(f"Low-rank / GRM mixed model fit by {method}")
        lines.append(f" Backend: {self.backend_name}")
        lines.append("")
        lines.append("Variance components:")
        lines.append(f"  {'Genetic (σ²_g)':<20s} {p.var_genetic:12.5f}")
        lines.append(f"  {'Residual (σ²_e)':<20s} {p.var_residual:12.5f}")
        lines.append(f"  {'Heritability (h²)':<20s} {p.heritability:12.5f}")
        lines.append("")
        lines.append(f"Number of obs: {p.n_obs}, rank (M): {p.rank}")
        lines.append("")
        lines.append("Fixed effects:")
        header = (f" {'':>15s} {'Estimate':>12s} {'Std. Error':>12s} "
                  f"{'z value':>10s} {'Pr(>|z|)':>10s}")
        lines.append(header)
        for i, name in enumerate(p.coefficient_names):
            lines.append(
                f" {name:>15s} {p.coefficients[i]:12.5f} {p.se[i]:12.5f} "
                f"{p.z_values[i]:10.3f} {p.p_values[i]:10.3g}"
            )
        lines.append("")
        lines.append(f"{method} criterion: {-2 * p.log_likelihood:.2f}   "
                     f"AIC: {p.aic:.2f}  BIC: {p.bic:.2f}")
        if not p.converged:
            lines.append("")
            lines.append("WARNING: Model did not converge")
        return '\n'.join(lines)

    def __repr__(self) -> str:
        p = self.params
        return (f"GRMSolution(h²={p.heritability:.3f}, "
                f"n={p.n_obs}, rank={p.rank}, "
                f"backend={self.backend_name!r})")
