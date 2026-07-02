"""
User-facing GAM solution with R-style summary output.

Wraps a :class:`Result[GAMParams]` and a list of :class:`SmoothInfo`
objects, exposing convenient properties and an ``mgcv``-style
``summary()`` method.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from pystatistics.core.result import Result, SolutionReprMixin
from pystatistics.gam._common import GAMParams, SmoothInfo
from pystatistics.regression._formatting import significance_stars


class GAMSolution(SolutionReprMixin):
    """User-facing GAM result with convenient properties and R-style output.

    Wraps a ``Result[GAMParams]`` and provides:

    * Coefficient access.
    * Smooth-term information (EDF, approximate significance tests).
    * R-style ``summary()`` matching ``mgcv::summary.gam()``.
    """

    __slots__ = ("_result", "_smooth_infos", "_names")

    def __init__(
        self,
        _result: Result[GAMParams],
        _smooth_infos: list[SmoothInfo],
        _names: list[str] | None = None,
    ) -> None:
        """Initialise the solution wrapper.

        Args:
            _result: The generic ``Result`` envelope carrying ``GAMParams``.
            _smooth_infos: One ``SmoothInfo`` per smooth term.
            _names: Optional names for parametric coefficients.
        """
        self._result = _result
        self._smooth_infos = _smooth_infos
        self._names = _names

    # ------------------------------------------------------------------
    # Properties delegating to GAMParams
    # ------------------------------------------------------------------

    @property
    def params(self) -> GAMParams:
        """The underlying frozen parameter payload."""
        return self._result.params

    @property
    def coefficients(self) -> NDArray:
        """Full coefficient vector (parametric + basis)."""
        return self._result.params.coefficients

    @property
    def fitted_values(self) -> NDArray:
        """Response-scale predictions."""
        return self._result.params.fitted_values

    @property
    def residuals(self) -> NDArray:
        """Working residuals (y - mu_hat)."""
        return self._result.params.residuals

    @property
    def edf(self) -> NDArray:
        """Effective degrees of freedom per smooth term."""
        return self._result.params.edf

    @property
    def total_edf(self) -> float:
        """Total effective degrees of freedom."""
        return self._result.params.total_edf

    @property
    def smooth_terms(self) -> list[SmoothInfo]:
        """Information about each smooth term."""
        return list(self._smooth_infos)

    @property
    def deviance(self) -> float:
        """Model deviance."""
        return self._result.params.deviance

    @property
    def null_deviance(self) -> float:
        """Null-model deviance."""
        return self._result.params.null_deviance

    @property
    def deviance_explained(self) -> float:
        """Proportion of null deviance explained: ``1 - dev / null_dev``."""
        if self.null_deviance == 0.0:
            return 0.0
        return 1.0 - self.deviance / self.null_deviance

    @property
    def aic(self) -> float:
        """Akaike information criterion."""
        return self._result.params.aic

    @property
    def gcv(self) -> float:
        """GCV score."""
        return self._result.params.gcv

    @property
    def scale(self) -> float:
        """Estimated dispersion parameter."""
        return self._result.params.scale

    @property
    def ubre(self) -> float:
        """UBRE / scaled-AIC score (the criterion mgcv's GCV.Cp minimises
        for known-scale families)."""
        return self._result.params.ubre

    @property
    def lambdas(self) -> NDArray:
        """Selected (or user-fixed) smoothing parameters, one per smooth.

        Reported in the same penalty scaling mgcv reports ``sp`` in; divide
        by ``params.s_scales`` for function-space units.
        """
        return self._result.params.lambdas

    @property
    def covariance(self) -> NDArray:
        """Bayesian posterior covariance of the coefficients (mgcv ``Vp``)."""
        return self._result.params.covariance

    @property
    def se(self) -> NDArray:
        """Standard errors for ALL coefficients (posterior, mgcv-style)."""
        return np.sqrt(np.maximum(
            np.diag(self._result.params.covariance), 0.0,
        ))

    @property
    def reml_score(self) -> float | None:
        """Laplace REML criterion (``None`` unless ``method='REML'``)."""
        return self._result.params.reml_score

    @property
    def converged(self) -> bool:
        """Whether the P-IRLS algorithm converged."""
        return self._result.params.converged

    @property
    def outer_converged(self) -> bool:
        """Whether the smoothing-parameter search converged off-boundary."""
        return self._result.params.outer_converged

    @property
    def n_iter(self) -> int:
        """Number of P-IRLS iterations executed."""
        return self._result.params.n_iter

    # ------------------------------------------------------------------
    # R-squared (adjusted)
    # ------------------------------------------------------------------

    @property
    def r_squared_adj(self) -> float:
        """Adjusted R-squared based on deviance explained.

        Matches the formula used by ``mgcv``::

            R_adj = 1 - (deviance / df_resid) / (null_deviance / df_null)
        """
        n = self._result.params.n_obs
        df_resid = max(n - self.total_edf, 1.0)
        df_null = max(n - 1.0, 1.0)
        if self.null_deviance == 0.0:
            return 0.0
        return 1.0 - (self.deviance / df_resid) / (self.null_deviance / df_null)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """R-style summary matching ``summary(gam(...))``.

        Includes:

        * Family and link function.
        * Parametric coefficient table with standard errors and tests.
        * Approximate significance tests for each smooth term.
        * Adjusted R-squared, deviance explained, GCV, and scale.

        Returns:
            Multi-line summary string.
        """
        p = self._result.params
        lines: list[str] = []

        lines.append(f"Family: {p.family_name}")
        lines.append(f"Link function: {p.link_name}")
        lines.append("")

        # Formula reconstruction: parametric terms first, then smooths.
        n_param = (
            self._smooth_infos[0].coef_indices[0]
            if self._smooth_infos else len(p.coefficients)
        )
        if self._names:
            param_terms = [
                nm for nm in self._names
                if nm.strip("()") not in ("Intercept", "intercept")
            ]
        else:
            # Unnamed: report the count honestly rather than inventing names.
            param_terms = (
                [f"X[{i}]" for i in range(1, n_param)] if n_param > 1 else []
            )
        pieces = param_terms + [si.term_name for si in self._smooth_infos]
        formula = f"y ~ {' + '.join(pieces)}" if pieces else "y ~ 1"
        lines.append(f"Formula: {formula}")
        lines.append("")

        # Parametric coefficients
        lines.append("Parametric coefficients:")
        # names length is validated against the parametric block in gam();
        # n_param always comes from the model structure, never from names.
        param_names = (
            list(self._names) if self._names
            else [f"B[{i}]" for i in range(n_param)]
        )
        stat_label = "z" if p.dispersion_fixed else "t"
        lines.append(
            f"{'':>16s} {'Estimate':>10s} {'Std. Error':>10s} "
            f"{stat_label + ' value':>10s} {'Pr(>|' + stat_label + '|)':>10s}"
        )

        # Known-scale families: z reference; estimated scale: t with the
        # residual df (mgcv's summary.gam convention).
        scale_known = p.dispersion_fixed
        resid_df = max(p.n_obs - p.total_edf, 1.0)
        for i, name in enumerate(param_names):
            coef_val = float(p.coefficients[i])
            se = self._param_se(i)
            z_val = coef_val / se if se > 0 else 0.0
            if scale_known:
                p_val = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z_val)))
            else:
                p_val = 2.0 * float(sp_stats.t.sf(abs(z_val), resid_df))
            stars = significance_stars(p_val)
            p_str = f"{p_val:.4e}" if p_val >= 1e-4 else "<2e-16"
            lines.append(
                f"{name:>16s} {coef_val:10.4f} {se:10.4f} "
                f"{z_val:10.3f} {p_str:>10s} {stars}"
            )
        lines.append("")

        # Approximate significance of smooth terms (mgcv labels the
        # statistic Chi.sq for fixed dispersion, F when it is estimated).
        lines.append("Approximate significance of smooth terms:")
        smooth_stat = "Chi.sq" if p.dispersion_fixed else "F"
        lines.append(
            f"{'':>16s} {'edf':>8s} {'Ref.df':>8s} "
            f"{smooth_stat:>10s} {'p-value':>10s}"
        )
        for si in self._smooth_infos:
            stars = significance_stars(si.p_value)
            p_str = f"{si.p_value:.4e}" if si.p_value >= 1e-4 else "<2e-16"
            lines.append(
                f"{si.term_name:>16s} {si.edf:8.3f} {si.ref_df:8.3f} "
                f"{si.chi_sq:10.2f} {p_str:>10s} {stars}"
            )
        lines.append("")

        # Diagnostics footer
        dev_expl_pct = self.deviance_explained * 100.0
        r2 = self.r_squared_adj
        lines.append(
            f"R-sq.(adj) = {r2:.3f}   "
            f"Deviance explained = {dev_expl_pct:.1f}%"
        )
        if p.method == "REML" and p.reml_score is not None:
            lines.append(
                f"-REML = {p.reml_score:.4g}  "
                f"Scale est. = {p.scale:.4g}  n = {p.n_obs}"
            )
        elif p.dispersion_fixed:
            # Known scale: UBRE was the selection criterion (mgcv GCV.Cp).
            lines.append(
                f"UBRE = {p.ubre:.4g}  Scale est. = {p.scale:.4g}  "
                f"n = {p.n_obs}"
            )
        else:
            lines.append(
                f"GCV = {p.gcv:.4g}  Scale est. = {p.scale:.4g}  n = {p.n_obs}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _param_se(self, idx: int) -> float:
        """Standard error for the *idx*-th coefficient.

        From the stored Bayesian posterior covariance
        ``V_beta = scale * (X'WX + sum lam*S)^{-1}`` (mgcv's ``Vp``) —
        the same quantity ``summary.gam`` uses for its parametric table.
        """
        v = float(self._result.params.covariance[idx, idx])
        return float(np.sqrt(v)) if v > 0.0 else 0.0

    def __repr__(self) -> str:
        p = self._result.params
        n_smooth = len(self._smooth_infos)
        return (
            f"GAMSolution(family={p.family_name!r}, "
            f"n_smooths={n_smooth}, "
            f"deviance_explained={self.deviance_explained:.1%})"
        )
