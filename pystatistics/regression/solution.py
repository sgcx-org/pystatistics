"""
Regression solution containers.

LinearParams holds the raw parameter estimates.
LinearSolution wraps the Result envelope and provides typed accessors.
"""

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result

if TYPE_CHECKING:
    from pystatistics.regression.design import RegressionDesign


@dataclass(frozen=True)
class LinearParams:
    """
    Parameter payload for linear regression.
    
    Attributes:
        coefficients: Estimated coefficients β (p,)
        residuals: Residuals y - Xβ (n,)
        fitted_values: Fitted values Xβ (n,)
        rss: Residual sum of squares ||y - Xβ||²
        tss: Total sum of squares ||y - ȳ||²
        rank: Numerical rank of X
        df_residual: Residual degrees of freedom (n - rank)
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
    User-facing wrapper for linear regression results.
    
    Provides typed access to parameters and convenience methods.
    This is what fit() returns to users.
    """
    _result: Result[LinearParams]
    _design: 'RegressionDesign'  # Keep reference for summary statistics
    
    # === Parameter Access ===
    
    @property
    def coefficients(self) -> NDArray[np.floating[Any]]:
        """Estimated coefficients β."""
        return self._result.params.coefficients
    
    @property
    def residuals(self) -> NDArray[np.floating[Any]]:
        """Residuals (y - Xβ)."""
        return self._result.params.residuals
    
    @property
    def fitted_values(self) -> NDArray[np.floating[Any]]:
        """Fitted values (Xβ)."""
        return self._result.params.fitted_values
    
    # === Goodness of Fit ===
    
    @property
    def rss(self) -> float:
        """Residual sum of squares."""
        return self._result.params.rss
    
    @property
    def tss(self) -> float:
        """Total sum of squares."""
        return self._result.params.tss
    
    @property
    def r_squared(self) -> float:
        """Coefficient of determination R²."""
        if self.tss == 0:
            return 1.0 if self.rss == 0 else 0.0
        return 1.0 - (self.rss / self.tss)
    
    @property
    def adjusted_r_squared(self) -> float:
        """Adjusted R² accounting for number of predictors."""
        n = self._design.n
        p = self._result.params.rank
        if n - p <= 0 or self.tss == 0:
            return self.r_squared
        return 1.0 - (1.0 - self.r_squared) * (n - 1) / (n - p)
    
    @property
    def residual_std_error(self) -> float:
        """Residual standard error (σ̂)."""
        df = self._result.params.df_residual
        if df <= 0:
            return 0.0
        return float(np.sqrt(self.rss / df))
    
    # === Inference ===
    
    @property
    def standard_errors(self) -> NDArray[np.floating[Any]]:
        """Standard errors of coefficients."""
        # SE(β̂) = σ̂ * sqrt(diag((X'X)^{-1}))
        df = self._result.params.df_residual
        if df <= 0:
            return np.zeros_like(self.coefficients)
        sigma_sq = self.rss / df
        XtX_inv = np.linalg.inv(self._design.XtX())
        return np.sqrt(sigma_sq * np.diag(XtX_inv))
    
    @property
    def t_statistics(self) -> NDArray[np.floating[Any]]:
        """t-statistics for coefficient significance tests."""
        se = self.standard_errors
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            t_stats = self.coefficients / se
            t_stats = np.where(np.isfinite(t_stats), t_stats, 0.0)
        return t_stats
    
    # === Metadata ===
    
    @property
    def rank(self) -> int:
        """Numerical rank of design matrix."""
        return self._result.params.rank
    
    @property
    def df_residual(self) -> int:
        """Residual degrees of freedom."""
        return self._result.params.df_residual
    
    @property
    def info(self) -> dict[str, Any]:
        """Backend and computation info."""
        return self._result.info
    
    @property
    def timing(self) -> dict[str, float] | None:
        """Execution timing breakdown."""
        return self._result.timing
    
    @property
    def backend_name(self) -> str:
        """Name of backend that produced this result."""
        return self._result.backend_name
    
    @property
    def warnings(self) -> tuple[str, ...]:
        """Any warnings generated during computation."""
        return self._result.warnings
    
    # === Display ===
    
    def summary(self) -> str:
        """Human-readable summary of regression results."""
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
        
        se = self.standard_errors
        t_stats = self.t_statistics
        
        for i, (coef, stderr, t) in enumerate(zip(self.coefficients, se, t_stats)):
            lines.append(f"  β[{i}]: {coef:12.6f}  SE: {stderr:10.6f}  t: {t:8.3f}")
        
        lines.append("-" * 50)
        lines.append(f"Backend: {self.backend_name}")
        
        if self.timing:
            lines.append(f"Time: {self.timing.get('total_seconds', 0):.4f}s")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (
            f"LinearSolution(n={self._design.n}, p={self._design.p}, "
            f"r_squared={self.r_squared:.4f}, backend='{self.backend_name}')"
        )
