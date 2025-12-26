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

from pystatistics.core.datasource import DataSource
from pystatistics.core.capabilities import CAPABILITY_GPU_NATIVE, CAPABILITY_REPEATABLE
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
            y_arr = source[y]
        elif 'y' in source:
            y_arr = source['y']
        else:
            raise ValueError("Must specify y or DataSource must have 'y'")
        
        # Get X
        if x is not None:
            if isinstance(x, str):
                X_arr = source[x]
            else:
                # Multiple columns - stack them
                X_arr = _get_columns(source, x)
        elif 'X' in source:
            X_arr = source['X']
        elif y is not None:
            # X = all columns except y
            x_cols = sorted(k for k in source.keys() if k != y)
            if not x_cols:
                raise ValueError("No predictor columns available")
            X_arr = _get_columns(source, x_cols)
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
    def source(self) -> DataSource | None:
        """Original DataSource, if available."""
        return self._source
    
    def supports(self, capability: str) -> bool:
        """Check if underlying data supports a capability."""
        if self._source is not None:
            return self._source.supports(capability)
        # Arrays in memory support these
        return capability in (CAPABILITY_REPEATABLE,)
    
    def XtX(self) -> NDArray[np.floating[Any]]:
        """Compute X'X (for standard errors)."""
        return self._X.T @ self._X
    
    def Xty(self) -> NDArray[np.floating[Any]]:
        """Compute X'y."""
        return self._X.T @ self._y


def _get_columns(source: DataSource, names: list[str]) -> NDArray:
    """Stack multiple columns from DataSource into a matrix."""
    arrays = []
    for name in names:
        arr = source[name]
        if hasattr(arr, 'cpu'):
            arr = arr.cpu().numpy()
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arrays.append(arr)
    return np.hstack(arrays)
