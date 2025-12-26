"""
Regression Design.

Design wraps a DataSource and extracts X (design matrix) and y (response).
It knows it's building a regression—DataSource doesn't.

Like a furniture maker visiting the lumber yard: "I need these logs
for making chairs." The lumber yard just provides logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.datasource import DataSource, CAPABILITY_GPU_NATIVE
from pystatistics.core.validation import check_finite, check_2d, check_1d, check_consistent_length, check_min_samples


@dataclass(frozen=True)
class Design:
    """
    Regression design matrix specification.
    
    Wraps a DataSource and provides X, y for regression.
    Immutable after construction.
    
    Construction:
        Design.from_datasource(ds, y='target')           # X = all other columns
        Design.from_datasource(ds, x=['a','b'], y='c')  # X = specified columns
        Design.from_datasource(ds)                        # Uses ds['X'] and ds['y']
        Design.from_arrays(X, y)                          # Direct from arrays
    """
    _X: NDArray[np.floating[Any]]
    _y: NDArray[np.floating[Any]]
    _n: int
    _p: int
    _source: DataSource | None = None
    
    @classmethod
    def from_datasource(
        cls,
        source: DataSource,
        *,
        x: str | list[str] | None = None,
        y: str | None = None,
    ) -> Design:
        """
        Build Design from DataSource.
        
        Args:
            source: The DataSource
            x: Predictor column(s). If None and source has 'X', uses that.
               If None and y is specified, uses all columns except y.
            y: Response column. If None, uses 'y' from source.
        
        Returns:
            Design ready for regression
        
        Assumes good faith: garbage in, garbage out.
        """
        # Get y
        if y is not None:
            y_arr = source.get(y)
        elif source.has('y'):
            y_arr = source.get('y')
        else:
            raise ValueError("Must specify y or DataSource must have 'y'")
        
        # Get X
        if x is not None:
            if isinstance(x, str):
                X_arr = source.get(x)
            else:
                X_arr = source.get_columns(x)
        elif source.has('X'):
            X_arr = source.get('X')
        elif y is not None:
            # X = all columns except y
            x_cols = [c for c in source.columns if c != y]
            if not x_cols:
                raise ValueError("No predictor columns available")
            X_arr = source.get_columns(x_cols)
        else:
            raise ValueError("Must specify x or DataSource must have 'X'")
        
        # Convert tensors to numpy if needed
        if hasattr(X_arr, 'cpu'):
            X_arr = X_arr.cpu().numpy()
        if hasattr(y_arr, 'cpu'):
            y_arr = y_arr.cpu().numpy()
        
        X_arr = np.asarray(X_arr, dtype=np.float64)
        y_arr = np.asarray(y_arr, dtype=np.float64)
        
        return cls._build(X_arr, y_arr, source=source)
    
    @classmethod
    def from_arrays(cls, X: NDArray, y: NDArray) -> Design:
        """Build Design directly from arrays."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return cls._build(X, y, source=None)
    
    @classmethod
    def _build(cls, X: NDArray, y: NDArray, source: DataSource | None) -> Design:
        """Internal builder with validation."""
        # Ensure correct shapes
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        
        # Validate
        check_2d(X, 'X')
        check_1d(y, 'y')
        check_finite(X, 'X')
        check_finite(y, 'y')
        check_consistent_length(X, y, names=('X', 'y'))
        
        n, p = X.shape
        check_min_samples(X, p, 'X')
        
        return cls(_X=X, _y=y, _n=n, _p=p, _source=source)
    
    # === Properties ===
    
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
    
    @property
    def n_observations(self) -> int:
        """Alias for n (DataSource protocol compatibility)."""
        return self._n
    
    @property
    def metadata(self) -> dict[str, Any]:
        """Design metadata."""
        return {'n': self._n, 'p': self._p, 'has_source': self._source is not None}
    
    def supports(self, capability: str) -> bool:
        """Check capability (DataSource protocol compatibility)."""
        if capability == CAPABILITY_GPU_NATIVE and self._source:
            return self._source.supports(CAPABILITY_GPU_NATIVE)
        return capability in {'materialize', 'second_pass', 'sufficient_stats'}
    
    # === Sufficient Statistics ===
    
    def XtX(self) -> NDArray[np.floating[Any]]:
        """X'X (p x p)."""
        return self._X.T @ self._X
    
    def Xty(self) -> NDArray[np.floating[Any]]:
        """X'y (p,)."""
        return self._X.T @ self._y
    
    def yty(self) -> float:
        """y'y (scalar)."""
        return float(self._y @ self._y)
    
    def __repr__(self) -> str:
        return f"Design(n={self._n}, p={self._p})"
