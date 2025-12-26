"""
Regression design (data container).

RegressionDesign implements the DataSource protocol and holds validated
regression data (X, y) along with computed quantities needed by backends.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.validation import (
    check_finite,
    check_1d,
    check_2d,
    check_consistent_length,
    check_min_samples,
)


@dataclass(frozen=True)
class RegressionDesign:
    """
    Immutable, validated container for regression data.
    
    Use the build() classmethod for construction—it performs all validation.
    Direct instantiation via __init__ is for internal use only.
    
    Implements the DataSource protocol for compatibility with shared tooling.
    
    Attributes:
        _X: Design matrix (n x p), private
        _y: Response vector (n,), private
        _n: Number of observations
        _p: Number of predictors
    """
    _X: NDArray[np.floating[Any]]
    _y: NDArray[np.floating[Any]]
    _n: int
    _p: int
    
    # Supported capabilities for this DataSource
    _CAPABILITIES: frozenset[str] = frozenset({
        'materialize',
        'second_pass',
        'sufficient_stats',
    })
    
    @classmethod
    def build(
        cls,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
    ) -> 'RegressionDesign':
        """
        Construct RegressionDesign with full validation.
        
        All validation happens here. After construction, the design
        is guaranteed to be valid and backends can trust the data.
        
        Args:
            X: Design matrix (n x p), already converted to array
            y: Response vector (n,), already converted to array
            
        Returns:
            Validated RegressionDesign instance
            
        Raises:
            ValidationError: If inputs fail validation
            DimensionError: If dimensions are inconsistent
        """
        # Validate X
        check_2d(X, 'X')
        check_finite(X, 'X')
        
        # Validate y
        check_1d(y, 'y')
        check_finite(y, 'y')
        
        # Check consistency
        check_consistent_length(X, y, names=('X', 'y'))
        
        n, p = X.shape
        
        # Need at least p observations for OLS
        check_min_samples(X, p, 'X')
        
        return cls(_X=X, _y=y, _n=n, _p=p)
    
    # === DataSource Protocol ===
    
    @property
    def n_observations(self) -> int:
        """Number of observations (rows in X)."""
        return self._n
    
    @property
    def metadata(self) -> dict[str, Any]:
        """Design metadata."""
        return {
            'n': self._n,
            'p': self._p,
            'X_dtype': str(self._X.dtype),
            'y_dtype': str(self._y.dtype),
        }
    
    def supports(self, capability: str) -> bool:
        """Check if capability is supported."""
        return capability in self._CAPABILITIES
    
    # === Domain-Specific Accessors ===
    
    @property
    def X(self) -> NDArray[np.floating[Any]]:
        """Design matrix (n x p)."""
        return self._X
    
    @property
    def y(self) -> NDArray[np.floating[Any]]:
        """Response vector (n,)."""
        return self._y
    
    @property
    def n(self) -> int:
        """Number of observations."""
        return self._n
    
    @property
    def p(self) -> int:
        """Number of predictors."""
        return self._p
    
    # === Sufficient Statistics (computed on demand) ===
    
    def XtX(self) -> NDArray[np.floating[Any]]:
        """Compute X'X (p x p)."""
        return self._X.T @ self._X
    
    def Xty(self) -> NDArray[np.floating[Any]]:
        """Compute X'y (p,)."""
        return self._X.T @ self._y
    
    def yty(self) -> float:
        """Compute y'y (scalar)."""
        return float(self._y @ self._y)
