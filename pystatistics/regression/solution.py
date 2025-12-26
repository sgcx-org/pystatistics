"""
Regression solution containers.
"""

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result

if TYPE_CHECKING:
    from pystatistics.regression.design import Design


@dataclass(frozen=True)
class LinearParams:
    """Parameter payload for linear regression."""
    coefficients: NDArray[np.floating[Any]]
    residuals: NDArray[np.floating[Any]]
    fitted_values: NDArray[np.floating[Any]]
    rss: float
    tss: float
    rank: int
    df_residual: int


@dataclass
class LinearSolution:
    """User-facing regression results."""
    _result: Result[LinearParams]
    _design: 'Design'
    
    @property
    def coefficients(self) -> NDArray[np.floating[Any]]:
        return self._result.params.coefficients
    
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
        df = self._result.params.df_residual
        if df <= 0:
            return np.zeros_like(self.coefficients)
        sigma_sq = self.rss / df
        XtX_inv = np.linalg.inv(self._design.XtX())
        return np.sqrt(sigma_sq * np.diag(XtX_inv))
    
    @property
    def t_statistics(self) -> NDArray[np.floating[Any]]:
        se = self.standard_errors
        with np.errstate(divide='ignore', invalid='ignore'):
            t = self.coefficients / se
            t = np.where(np.isfinite(t), t, 0.0)
        return t
    
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
        lines = [
            "Linear Regression Results",
            "=" * 50,
            f"Observations: {self._design.n}",
            f"Predictors: {self._design.p}",
            f"R-squared: {self.r_squared:.6f}",
            f"Adj. R-squared: {self.adjusted_r_squared:.6f}",
            f"Residual Std. Error: {self.residual_std_error:.6f} on {self.df_residual} DF",
            "",
            "Coefficients:",
            "-" * 50,
        ]
        for i, (coef, se, t) in enumerate(zip(self.coefficients, self.standard_errors, self.t_statistics)):
            lines.append(f"  β[{i}]: {coef:12.6f}  SE: {se:10.6f}  t: {t:8.3f}")
        lines.append("-" * 50)
        lines.append(f"Backend: {self.backend_name}")
        if self.timing:
            lines.append(f"Time: {self.timing.get('total_seconds', 0):.4f}s")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"LinearSolution(n={self._design.n}, p={self._design.p}, r_squared={self.r_squared:.4f})"
