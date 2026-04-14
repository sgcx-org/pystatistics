"""
Linear regression parameter payload and solution.

LinearParams: immutable data computed by backends.
LinearSolution: user-facing results with R-style summary.
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
class LinearParams:
    """
    Parameter payload for linear regression.

    This is the immutable data computed by backends.
    """
    coefficients: NDArray[np.floating[Any]]
    residuals: NDArray[np.floating[Any]]
    fitted_values: NDArray[np.floating[Any]]
    rss: float
    tss: float
    rank: int
    df_residual: int


@dataclass
class LinearSolution:
    """
    User-facing regression results.

    Wraps the backend Result and provides convenient accessors
    for all regression outputs including standard errors and t-statistics.
    """
    _result: Result[LinearParams]
    _design: 'Design'

    # Optional variable names (set by fit() via names= kwarg)
    _names: tuple[str, ...] | None = None

    # Cached computations
    _standard_errors: NDArray[np.floating[Any]] | None = None
    _t_statistics: NDArray[np.floating[Any]] | None = None
    _p_values: NDArray[np.floating[Any]] | None = None

    @property
    def coefficients(self) -> NDArray[np.floating[Any]]:
        return self._result.params.coefficients

    @property
    def coef(self) -> dict[str, float]:
        """Named coefficient mapping (like R's coef() or statsmodels .params).

        Returns dict mapping variable names to coefficient values.
        Falls back to 'B[0]', 'B[1]'... when names are not available.
        """
        names = self._names or tuple(f"B[{i}]" for i in range(len(self.coefficients)))
        return dict(zip(names, self.coefficients.tolist()))

    @property
    def residuals(self) -> NDArray[np.floating[Any]]:
        return self._result.params.residuals

    @property
    def fitted_values(self) -> NDArray[np.floating[Any]]:
        return self._result.params.fitted_values

    @property
    def rss(self) -> float:
        return self._result.params.rss

    @property
    def tss(self) -> float:
        return self._result.params.tss

    @property
    def r_squared(self) -> float:
        if self.tss == 0:
            return 1.0 if self.rss == 0 else 0.0
        return 1.0 - (self.rss / self.tss)

    @property
    def adjusted_r_squared(self) -> float:
        n = self._design.n
        p = self._result.params.rank
        if n - p <= 0 or self.tss == 0:
            return self.r_squared
        return 1.0 - (1.0 - self.r_squared) * (n - 1) / (n - p)

    @property
    def residual_std_error(self) -> float:
        df = self._result.params.df_residual
        if df <= 0:
            return 0.0
        return float(np.sqrt(self.rss / df))

    @property
    def standard_errors(self) -> NDArray[np.floating[Any]]:
        """
        Standard errors of coefficients.

        For CPU QR backend: uses R^{-1} from QR decomposition to compute
        (X'X)^{-1} = R^{-1} R^{-T}, matching R's backsolve(R, diag(p)) exactly.

        For GPU backend (no R available): uses np.linalg.inv(X'X).
        # NOT A FALLBACK: mathematically equivalent to QR path,
        # just a different computation route since GPU doesn't store QR factors.

        For rank-deficient matrices, aliased coefficients get NaN standard errors.
        """
        if self._standard_errors is not None:
            return self._standard_errors

        df = self._result.params.df_residual
        rank = self._result.params.rank
        p = len(self.coefficients)

        if df <= 0:
            self._standard_errors = np.full(p, np.nan, dtype=np.float64)
            return self._standard_errors

        sigma_sq = self.rss / df

        pivot = self._result.info.get('pivot')
        R = self._result.info.get('R')

        if R is not None and pivot is not None:
            # QR-based SE computation (matches R exactly)
            from scipy.linalg import solve_triangular
            pivot = np.array(pivot)

            if rank < p:
                R_active = R[:rank, :rank]
                try:
                    R_inv = solve_triangular(
                        R_active, np.eye(rank), lower=False
                    )
                    XtX_inv_diag = np.sum(R_inv ** 2, axis=1)
                    se_pivoted = np.sqrt(sigma_sq * XtX_inv_diag)
                except np.linalg.LinAlgError:
                    se_pivoted = np.full(rank, np.nan, dtype=np.float64)

                se = np.full(p, np.nan, dtype=np.float64)
                se[pivot[:rank]] = se_pivoted
            else:
                R_sq = R[:p, :p]
                try:
                    R_inv = solve_triangular(
                        R_sq, np.eye(p), lower=False
                    )
                    XtX_inv_diag = np.sum(R_inv ** 2, axis=1)
                    se_pivoted = np.sqrt(sigma_sq * XtX_inv_diag)
                except np.linalg.LinAlgError:
                    se_pivoted = np.full(p, np.nan, dtype=np.float64)

                se = np.empty(p, dtype=np.float64)
                se[pivot] = se_pivoted

            self._standard_errors = se
        else:
            # NOT A FALLBACK: mathematically equivalent to QR path.
            # GPU backends don't store QR factors, so we compute
            # (X'X)^{-1} directly. Same result, different route.
            try:
                XtX_inv = np.linalg.inv(self._design.XtX())
                self._standard_errors = np.sqrt(sigma_sq * np.diag(XtX_inv))
            except np.linalg.LinAlgError:
                self._standard_errors = np.full(p, np.nan, dtype=np.float64)

        return self._standard_errors

    @property
    def t_statistics(self) -> NDArray[np.floating[Any]]:
        """t-statistics for coefficients."""
        if self._t_statistics is not None:
            return self._t_statistics

        se = self.standard_errors
        with np.errstate(divide='ignore', invalid='ignore'):
            t = self.coefficients / se
            t = np.where(np.isfinite(t), t, np.nan)
        self._t_statistics = t
        return self._t_statistics

    @property
    def p_values(self) -> NDArray[np.floating[Any]]:
        """Two-sided p-values for coefficient t-tests."""
        if self._p_values is not None:
            return self._p_values

        from scipy import stats

        t = self.t_statistics
        df = self.df_residual

        if df <= 0:
            self._p_values = np.full(len(t), np.nan, dtype=np.float64)
            return self._p_values

        pv = 2.0 * stats.t.sf(np.abs(t), df=df)
        self._p_values = np.where(np.isfinite(t), pv, np.nan)
        return self._p_values

    @property
    def rank(self) -> int:
        return self._result.params.rank

    @property
    def df_residual(self) -> int:
        return self._result.params.df_residual

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
        """Generate R-style summary output."""
        p = len(self.coefficients)
        names = self._names or tuple(f"B[{i}]" for i in range(p))
        max_name_len = max(len(n) for n in names)
        col_w = max(max_name_len + 2, 15)

        rq = np.quantile(self.residuals, [0.0, 0.25, 0.5, 0.75, 1.0])

        lines = [
            "Linear Regression Results",
            "=" * 72,
            f"Observations: {self._design.n}",
            f"Predictors: {self._design.p}",
            f"Rank: {self.rank}",
            "",
            "Residuals:",
            f"  {'Min':>10s} {'1Q':>10s} {'Median':>10s} {'3Q':>10s} {'Max':>10s}",
            f"  {rq[0]:10.4f} {rq[1]:10.4f} {rq[2]:10.4f} {rq[3]:10.4f} {rq[4]:10.4f}",
            "",
            "Coefficients:",
            "-" * 72,
            f"  {'':<{col_w}s} {'Estimate':>14s} {'Std.Error':>12s} {'t value':>10s} {'Pr(>|t|)':>12s}",
            "-" * 72,
        ]

        for name, coef, se, t, pv in zip(
            names, self.coefficients, self.standard_errors,
            self.t_statistics, self.p_values
        ):
            if np.isnan(coef):
                lines.append(f"  {name:<{col_w}s} (aliased)")
            else:
                se_str = f"{se:12.6f}" if not np.isnan(se) else "         NA"
                t_str = f"{t:10.3f}" if not np.isnan(t) else "        NA"
                pv_str = f"{pv:12.4e}" if not np.isnan(pv) else "          NA"
                sig = significance_stars(pv)
                lines.append(
                    f"  {name:<{col_w}s} {coef:14.6f} {se_str} {t_str} {pv_str} {sig}"
                )

        lines.append("-" * 72)
        lines.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        lines.append("")
        lines.append(
            f"Residual standard error: {self.residual_std_error:.4f} on {self.df_residual} degrees of freedom"
        )
        lines.append(
            f"Multiple R-squared: {self.r_squared:.6f},  "
            f"Adjusted R-squared: {self.adjusted_r_squared:.6f}"
        )

        df_model = self.rank - 1
        if df_model > 0 and self.df_residual > 0 and self.tss > 0:
            from scipy import stats as sp_stats
            f_stat = (self.r_squared / df_model) / ((1.0 - self.r_squared) / self.df_residual)
            f_pval = sp_stats.f.sf(f_stat, df_model, self.df_residual)
            lines.append(
                f"F-statistic: {f_stat:.2f} on {df_model} and {self.df_residual} DF,  "
                f"p-value: {f_pval:.4e}"
            )

        lines.append(f"Backend: {self.backend_name}")
        if self.timing:
            lines.append(f"Time: {self.timing.get('total_seconds', 0):.4f}s")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"LinearSolution(n={self._design.n}, p={self._design.p}, "
            f"rank={self.rank}, r_squared={self.r_squared:.4f})"
        )
