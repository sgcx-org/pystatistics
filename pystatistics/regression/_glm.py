"""
GLM parameter payload and solution.

GLMParams: immutable data computed by GLM backends (IRLS).
GLMSolution: user-facing results with R-style summary.
"""

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result
from pystatistics.regression._formatting import significance_stars

if TYPE_CHECKING:
    from pystatistics.regression.design import Design


@dataclass(frozen=True)
class GLMParams:
    """
    Parameter payload for GLM (IRLS).

    This is the immutable data computed by GLM backends.
    """
    coefficients: NDArray[np.floating[Any]]
    fitted_values: NDArray[np.floating[Any]]        # mu (response scale)
    linear_predictor: NDArray[np.floating[Any]]      # eta (link scale)
    residuals_working: NDArray[np.floating[Any]]
    residuals_deviance: NDArray[np.floating[Any]]
    residuals_pearson: NDArray[np.floating[Any]]
    residuals_response: NDArray[np.floating[Any]]    # y - mu
    deviance: float
    null_deviance: float
    aic: float
    dispersion: float
    rank: int
    df_residual: int
    df_null: int
    n_iter: int
    converged: bool
    family_name: str
    link_name: str


@dataclass
class GLMSolution:
    """
    User-facing GLM results.

    Wraps the backend Result and provides convenient accessors for
    all GLM-specific outputs including deviance, AIC, and multiple
    residual types.
    """
    _result: Result[GLMParams]
    _design: 'Design'

    # Optional variable names (set by fit() via names= kwarg)
    _names: tuple[str, ...] | None = None

    # Cached computations
    _standard_errors: NDArray[np.floating[Any]] | None = None
    _test_statistics: NDArray[np.floating[Any]] | None = None
    _p_values: NDArray[np.floating[Any]] | None = None

    @property
    def coefficients(self) -> NDArray[np.floating[Any]]:
        return self._result.params.coefficients

    @property
    def coef(self) -> dict[str, float]:
        """Named coefficient mapping (like R's coef() or statsmodels .params)."""
        names = self._names or tuple(f"B[{i}]" for i in range(len(self.coefficients)))
        return dict(zip(names, self.coefficients.tolist()))

    @property
    def fitted_values(self) -> NDArray[np.floating[Any]]:
        """Fitted values on the response scale (mu)."""
        return self._result.params.fitted_values

    @property
    def linear_predictor(self) -> NDArray[np.floating[Any]]:
        """Linear predictor eta = X @ beta."""
        return self._result.params.linear_predictor

    @property
    def residuals_deviance(self) -> NDArray[np.floating[Any]]:
        """Deviance residuals (signed)."""
        return self._result.params.residuals_deviance

    @property
    def residuals_pearson(self) -> NDArray[np.floating[Any]]:
        """Pearson residuals: (y - mu) / sqrt(V(mu))."""
        return self._result.params.residuals_pearson

    @property
    def residuals_working(self) -> NDArray[np.floating[Any]]:
        """Working residuals from the final IRLS iteration."""
        return self._result.params.residuals_working

    @property
    def residuals_response(self) -> NDArray[np.floating[Any]]:
        """Response residuals: y - mu."""
        return self._result.params.residuals_response

    @property
    def deviance(self) -> float:
        """Residual deviance."""
        return self._result.params.deviance

    @property
    def null_deviance(self) -> float:
        """Null deviance (intercept-only model)."""
        return self._result.params.null_deviance

    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        return self._result.params.aic

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        return self.aic - 2.0 * self.rank + self.rank * np.log(self._design.n)

    @property
    def dispersion(self) -> float:
        """Dispersion parameter.

        Fixed at 1.0 for Binomial and Poisson. Estimated from data for Gaussian.
        """
        return self._result.params.dispersion

    @property
    def rank(self) -> int:
        return self._result.params.rank

    @property
    def df_residual(self) -> int:
        return self._result.params.df_residual

    @property
    def df_null(self) -> int:
        return self._result.params.df_null

    @property
    def converged(self) -> bool:
        return self._result.params.converged

    @property
    def n_iter(self) -> int:
        return self._result.params.n_iter

    @property
    def family_name(self) -> str:
        return self._result.params.family_name

    @property
    def link_name(self) -> str:
        return self._result.params.link_name

    @property
    def standard_errors(self) -> NDArray[np.floating[Any]]:
        """Standard errors of coefficients.

        Computed as sqrt(dispersion * diag((X'WX)^{-1})) where W are
        the final IRLS weights. Uses QR from the final iteration when
        available (CPU). GPU path uses direct (X'WX)^{-1} inversion,
        which is mathematically equivalent.
        """
        if self._standard_errors is not None:
            return self._standard_errors

        p = len(self.coefficients)
        disp = self.dispersion

        pivot = self._result.info.get('pivot')
        R = self._result.info.get('R')

        if R is not None and pivot is not None:
            from scipy.linalg import solve_triangular
            pivot = np.array(pivot)
            R_sq = R[:p, :p]
            try:
                R_inv = solve_triangular(R_sq, np.eye(p), lower=False)
                XtWX_inv_diag = np.sum(R_inv ** 2, axis=1)
                se_pivoted = np.sqrt(disp * XtWX_inv_diag)
            except np.linalg.LinAlgError:
                se_pivoted = np.full(p, np.nan, dtype=np.float64)

            se = np.empty(p, dtype=np.float64)
            se[pivot] = se_pivoted
            self._standard_errors = se
        else:
            # NOT A FALLBACK: mathematically equivalent to QR path.
            # GPU backends store X'WX directly instead of QR factors.
            XtWX = self._result.info.get('XtWX')
            if XtWX is not None:
                try:
                    XtWX_inv = np.linalg.inv(XtWX)
                    self._standard_errors = np.sqrt(disp * np.diag(XtWX_inv))
                except np.linalg.LinAlgError:
                    self._standard_errors = np.full(p, np.nan, dtype=np.float64)
            else:
                self._standard_errors = np.full(p, np.nan, dtype=np.float64)

        return self._standard_errors

    @property
    def _uses_z_test(self) -> bool:
        """Whether to use z-test (fixed dispersion) or t-test (estimated)."""
        return self._result.params.family_name in ('binomial', 'poisson')

    @property
    def test_statistics(self) -> NDArray[np.floating[Any]]:
        """Wald test statistics (z for fixed dispersion, t for estimated)."""
        if self._test_statistics is not None:
            return self._test_statistics

        se = self.standard_errors
        with np.errstate(divide='ignore', invalid='ignore'):
            stat = self.coefficients / se
            stat = np.where(np.isfinite(stat), stat, np.nan)
        self._test_statistics = stat
        return self._test_statistics

    @property
    def p_values(self) -> NDArray[np.floating[Any]]:
        """Two-sided p-values.

        Uses z-distribution for Binomial/Poisson (fixed dispersion=1).
        Uses t-distribution for Gaussian (estimated dispersion).
        """
        if self._p_values is not None:
            return self._p_values

        from scipy import stats as sp_stats

        stat = self.test_statistics

        if self._uses_z_test:
            pv = 2.0 * sp_stats.norm.sf(np.abs(stat))
        else:
            df = self.df_residual
            if df <= 0:
                pv = np.full(len(stat), np.nan, dtype=np.float64)
            else:
                pv = 2.0 * sp_stats.t.sf(np.abs(stat), df=df)

        self._p_values = np.where(np.isfinite(stat), pv, np.nan)
        return self._p_values

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
        """Generate R-style GLM summary output."""
        stat_name = "z value" if self._uses_z_test else "t value"
        p_header = "Pr(>|z|)" if self._uses_z_test else "Pr(>|t|)"
        p = len(self.coefficients)
        names = self._names or tuple(f"B[{i}]" for i in range(p))
        max_name_len = max(len(n) for n in names)
        col_w = max(max_name_len + 2, 15)

        lines = [
            "GLM Results",
            "=" * 72,
            f"Family: {self.family_name} ({self.link_name} link)",
            f"Observations: {self._design.n}",
            f"Predictors: {self._design.p}",
            f"Converged: {self.converged} ({self.n_iter} iterations)",
            "",
            f"Deviance: {self.deviance:.4f}  (Null: {self.null_deviance:.4f})",
            f"AIC: {self.aic:.4f}",
            f"Dispersion: {self.dispersion:.6f}",
            "",
            "Coefficients:",
            "-" * 72,
            f"  {'':<{col_w}s} {'Estimate':>14s} {'Std.Error':>12s} {stat_name:>10s} {p_header:>12s}",
            "-" * 72,
        ]

        for name, coef, se, stat, pv in zip(
            names, self.coefficients, self.standard_errors,
            self.test_statistics, self.p_values
        ):
            se_str = f"{se:12.6f}" if not np.isnan(se) else "         NA"
            stat_str = f"{stat:10.3f}" if not np.isnan(stat) else "        NA"
            pv_str = f"{pv:12.4e}" if not np.isnan(pv) else "          NA"
            sig = significance_stars(pv)
            lines.append(
                f"  {name:<{col_w}s} {coef:14.6f} {se_str} {stat_str} {pv_str} {sig}"
            )

        lines.append("-" * 72)
        lines.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        lines.append(f"Backend: {self.backend_name}")
        if self.timing:
            lines.append(f"Time: {self.timing.get('total_seconds', 0):.4f}s")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GLMSolution(family={self.family_name!r}, "
            f"n={self._design.n}, p={self._design.p}, "
            f"deviance={self.deviance:.4f}, converged={self.converged})"
        )
