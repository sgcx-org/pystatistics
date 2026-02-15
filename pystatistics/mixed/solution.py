"""
Solution wrappers for mixed models.

LMMSolution and GLMMSolution wrap Result[LMMParams] / Result[GLMMParams]
and provide R-style summary output, property accessors for common
quantities, and model comparison via likelihood ratio tests.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from pystatistics.core.result import Result
from pystatistics.mixed._common import LMMParams, GLMMParams, VarCompSummary


def _significance_stars(p: float) -> str:
    """Return significance stars like R."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '.'
    else:
        return ' '


def _format_pvalue(p: float) -> str:
    """Format p-value like R."""
    if p < 2e-16:
        return '< 2e-16'
    elif p < 0.001:
        return f'{p:.2e}'
    else:
        return f'{p:.4f}'


class LMMSolution:
    """Solution wrapper for a fitted linear mixed model.

    Provides R-style summary output matching lmerTest::summary(),
    property accessors for fixed effects, random effects, ICC,
    and model comparison via likelihood ratio test.
    """

    def __init__(self, _result: Result[LMMParams]):
        self._result = _result

    @property
    def params(self) -> LMMParams:
        return self._result.params

    # --- Fixed effects ---

    @property
    def coefficients(self) -> NDArray:
        """Fixed effect estimates β̂."""
        return self.params.coefficients

    @property
    def fixef(self) -> dict[str, float]:
        """Fixed effects as name → value dict."""
        return dict(zip(self.params.coefficient_names, self.params.coefficients))

    @property
    def se(self) -> NDArray:
        """Standard errors of fixed effects."""
        return self.params.se

    @property
    def t_values(self) -> NDArray:
        """t-statistics for fixed effects."""
        return self.params.t_values

    @property
    def p_values(self) -> NDArray:
        """p-values for fixed effects (Satterthwaite df)."""
        return self.params.p_values

    @property
    def df_satterthwaite(self) -> NDArray:
        """Satterthwaite denominator df for each fixed effect."""
        return self.params.df_satterthwaite

    # --- Random effects ---

    @property
    def ranef(self) -> dict[str, NDArray]:
        """Random effects (BLUPs / conditional modes) per grouping factor."""
        return self.params.random_effects

    @property
    def var_components(self) -> tuple[VarCompSummary, ...]:
        """Variance component summaries."""
        return self.params.var_components

    @property
    def icc(self) -> dict[str, float]:
        """Intraclass correlation coefficient per grouping factor.

        ICC = σ²_group / (σ²_group + σ²_residual)

        For models with random slopes, uses the intercept variance only.
        """
        sigma_sq_resid = self.params.residual_variance
        result = {}
        for vc in self.params.var_components:
            if vc.name in ('(Intercept)', '1'):
                key = vc.group
                if key not in result:
                    result[key] = vc.variance / (vc.variance + sigma_sq_resid)
        return result

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

    # --- Model comparison ---

    def compare(self, other: 'LMMSolution') -> str:
        """Likelihood ratio test between two nested models.

        Both models should be fit with ML (reml=False) for valid LRT.

        Args:
            other: The other model to compare against.

        Returns:
            Formatted LRT summary string.
        """
        if self.params.reml or other.params.reml:
            import warnings
            warnings.warn(
                "Likelihood ratio test requires ML (not REML) fits for "
                "valid comparison. Refit with reml=False.",
                UserWarning,
                stacklevel=2,
            )

        # Determine which model has more parameters
        n_params_self = len(self.params.theta) + len(self.params.coefficients)
        n_params_other = len(other.params.theta) + len(other.params.coefficients)

        if n_params_self >= n_params_other:
            full, reduced = self, other
            n_full, n_reduced = n_params_self, n_params_other
        else:
            full, reduced = other, self
            n_full, n_reduced = n_params_other, n_params_self

        chi_sq = -2.0 * (reduced.log_likelihood - full.log_likelihood)
        chi_sq = max(chi_sq, 0.0)
        df = n_full - n_reduced
        if df <= 0:
            df = 1
        p_value = float(stats.chi2.sf(chi_sq, df))

        lines = [
            "Likelihood Ratio Test",
            "=" * 50,
            f"  Reduced model logLik: {reduced.log_likelihood:.4f}  "
            f"(df = {n_reduced})",
            f"  Full model logLik:    {full.log_likelihood:.4f}  "
            f"(df = {n_full})",
            f"  Chi-squared: {chi_sq:.4f}  on {df} df",
            f"  p-value: {_format_pvalue(p_value)}",
        ]
        return '\n'.join(lines)

    # --- Summary ---

    def summary(self) -> str:
        """R-style summary matching lmerTest::summary(lmer(...))."""
        params = self.params
        method = 'REML' if params.reml else 'ML'

        lines = []
        lines.append(f"Linear mixed model fit by {method}")
        lines.append("")

        # Random effects table
        lines.append("Random effects:")
        lines.append(f" {'Groups':<12s} {'Name':<15s} {'Variance':>10s} "
                      f"{'Std.Dev.':>10s} {'Corr':>6s}")

        prev_group = None
        for vc in params.var_components:
            grp_label = vc.group if vc.group != prev_group else ''
            corr_str = f'{vc.corr:6.2f}' if vc.corr is not None else ''
            lines.append(
                f" {grp_label:<12s} {vc.name:<15s} {vc.variance:10.4f} "
                f"{vc.std_dev:10.4f} {corr_str}"
            )
            prev_group = vc.group

        lines.append(
            f" {'Residual':<12s} {'':<15s} {params.residual_variance:10.4f} "
            f"{params.residual_std:10.4f}"
        )
        lines.append("")

        # Number of obs and groups
        group_parts = ', '.join(
            f'{name}: {n}' for name, n in params.n_groups.items()
        )
        lines.append(
            f"Number of obs: {params.n_obs}, groups: {group_parts}"
        )
        lines.append("")

        # Fixed effects table
        lines.append("Fixed effects:")
        header = (f" {'':>15s} {'Estimate':>10s} {'Std. Error':>10s} "
                  f"{'df':>10s} {'t value':>10s} {'Pr(>|t|)':>10s} {'':>4s}")
        lines.append(header)

        for i, name in enumerate(params.coefficient_names):
            p_str = _format_pvalue(params.p_values[i])
            stars = _significance_stars(params.p_values[i])
            lines.append(
                f" {name:>15s} {params.coefficients[i]:10.4f} "
                f"{params.se[i]:10.4f} {params.df_satterthwaite[i]:10.2f} "
                f"{params.t_values[i]:10.3f} {p_str:>10s} {stars}"
            )

        lines.append("---")
        lines.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        lines.append("")

        # Model fit
        lines.append(f"{'REML' if params.reml else 'ML'} criterion at convergence: "
                      f"{-2 * params.log_likelihood:.1f}")
        lines.append(f"AIC: {params.aic:.1f}, BIC: {params.bic:.1f}")

        if not params.converged:
            lines.append("")
            lines.append("WARNING: Model did not converge")

        return '\n'.join(lines)

    def __repr__(self) -> str:
        method = 'REML' if self.params.reml else 'ML'
        nfe = len(self.params.coefficients)
        nre = len(self.params.var_components)
        return (
            f"LMMSolution({method}, "
            f"n={self.params.n_obs}, "
            f"fixed={nfe}, "
            f"random={nre} var components)"
        )


class GLMMSolution:
    """Solution wrapper for a fitted generalized linear mixed model.

    Same interface as LMMSolution plus family-specific properties.
    Uses Wald z-statistics (not Satterthwaite t) for inference.
    """

    def __init__(self, _result: Result[GLMMParams]):
        self._result = _result

    @property
    def params(self) -> GLMMParams:
        return self._result.params

    # --- Fixed effects ---

    @property
    def coefficients(self) -> NDArray:
        return self.params.coefficients

    @property
    def fixef(self) -> dict[str, float]:
        return dict(zip(self.params.coefficient_names, self.params.coefficients))

    @property
    def se(self) -> NDArray:
        return self.params.se

    @property
    def z_values(self) -> NDArray:
        """Wald z-statistics for fixed effects."""
        return self.params.t_values

    @property
    def p_values(self) -> NDArray:
        return self.params.p_values

    # --- Random effects ---

    @property
    def ranef(self) -> dict[str, NDArray]:
        return self.params.random_effects

    @property
    def var_components(self) -> tuple[VarCompSummary, ...]:
        return self.params.var_components

    @property
    def icc(self) -> dict[str, float]:
        """ICC on the latent (link) scale.

        For GLMM, ICC is computed on the link scale:
        ICC = σ²_group / (σ²_group + π²/3) for logistic
        ICC = σ²_group / (σ²_group + 1) for probit
        """
        # Use distribution-specific residual variance
        family = self.params.family_name.lower()
        link = self.params.link_name.lower()
        if family == 'binomial' and link == 'logit':
            sigma_sq_resid = np.pi**2 / 3.0
        elif family == 'binomial' and link == 'probit':
            sigma_sq_resid = 1.0
        else:
            sigma_sq_resid = 1.0  # default for other families

        result = {}
        for vc in self.params.var_components:
            if vc.name in ('(Intercept)', '1'):
                key = vc.group
                if key not in result:
                    result[key] = vc.variance / (vc.variance + sigma_sq_resid)
        return result

    # --- Model fit ---

    @property
    def log_likelihood(self) -> float:
        return self.params.log_likelihood

    @property
    def deviance(self) -> float:
        return self.params.deviance

    @property
    def aic(self) -> float:
        return self.params.aic

    @property
    def bic(self) -> float:
        return self.params.bic

    @property
    def fitted_values(self) -> NDArray:
        """Fitted values on the response scale (μ̂)."""
        return self.params.fitted_values

    @property
    def linear_predictor(self) -> NDArray:
        """Linear predictor (η̂ = Xβ̂ + Zb̂)."""
        return self.params.linear_predictor

    @property
    def residuals(self) -> NDArray:
        return self.params.residuals

    @property
    def converged(self) -> bool:
        return self.params.converged

    # --- Summary ---

    def summary(self) -> str:
        """R-style summary matching lme4::summary(glmer(...))."""
        params = self.params

        lines = []
        lines.append(
            f"Generalized linear mixed model fit by ML "
            f"(Laplace Approximation)"
        )
        lines.append(f" Family: {params.family_name} ( {params.link_name} )")
        lines.append("")

        # Random effects
        lines.append("Random effects:")
        lines.append(f" {'Groups':<12s} {'Name':<15s} {'Variance':>10s} "
                      f"{'Std.Dev.':>10s}")
        prev_group = None
        for vc in params.var_components:
            grp_label = vc.group if vc.group != prev_group else ''
            lines.append(
                f" {grp_label:<12s} {vc.name:<15s} {vc.variance:10.4f} "
                f"{vc.std_dev:10.4f}"
            )
            prev_group = vc.group
        lines.append("")

        # Number of obs
        group_parts = ', '.join(
            f'{name}: {n}' for name, n in params.n_groups.items()
        )
        lines.append(
            f"Number of obs: {params.n_obs}, groups: {group_parts}"
        )
        lines.append("")

        # Fixed effects (Wald z-test)
        lines.append("Fixed effects:")
        header = (f" {'':>15s} {'Estimate':>10s} {'Std. Error':>10s} "
                  f"{'z value':>10s} {'Pr(>|z|)':>10s} {'':>4s}")
        lines.append(header)

        for i, name in enumerate(params.coefficient_names):
            p_str = _format_pvalue(params.p_values[i])
            stars = _significance_stars(params.p_values[i])
            lines.append(
                f" {name:>15s} {params.coefficients[i]:10.4f} "
                f"{params.se[i]:10.4f} "
                f"{params.t_values[i]:10.3f} {p_str:>10s} {stars}"
            )

        lines.append("---")
        lines.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        lines.append("")

        lines.append(f"AIC: {params.aic:.1f}, BIC: {params.bic:.1f}")
        lines.append(f"Deviance: {params.deviance:.4f}")

        if not params.converged:
            lines.append("")
            lines.append("WARNING: Model did not converge")

        return '\n'.join(lines)

    def __repr__(self) -> str:
        nfe = len(self.params.coefficients)
        nre = len(self.params.var_components)
        return (
            f"GLMMSolution({self.params.family_name}({self.params.link_name}), "
            f"n={self.params.n_obs}, "
            f"fixed={nfe}, "
            f"random={nre} var components)"
        )
