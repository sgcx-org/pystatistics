"""
MVNDesign: data wrapper for MVN MLE with missing values.

Wraps a data matrix and provides validation and metadata for
the MVN MLE pipeline. Follows pystatistics Design pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ValidationError


@dataclass(frozen=True)
class MVNDesign:
    """
    Design for multivariate normal MLE with missing data.

    Wraps a data matrix (n observations x p variables) that may contain
    NaN values representing missing data. Immutable after construction.

    Construction:
        MVNDesign.from_array(data)
        MVNDesign.from_datasource(ds, columns=['a', 'b', 'c'])
    """
    _data: NDArray[np.floating[Any]]
    _n: int
    _p: int

    @classmethod
    def from_array(cls, data) -> MVNDesign:
        """
        Build MVNDesign from array-like data.

        Parameters
        ----------
        data : array-like
            2D data matrix. Can be numpy array, pandas DataFrame,
            or any array-like with .values attribute.
        """
        if hasattr(data, 'values'):
            data_array = np.asarray(data.values, dtype=np.float64)
        else:
            data_array = np.asarray(data, dtype=np.float64)

        return cls._build(data_array)

    @classmethod
    def from_datasource(cls, source, *, columns: list[str] | None = None) -> MVNDesign:
        """
        Build MVNDesign from a DataSource.

        Parameters
        ----------
        source : DataSource
            Data source providing columns
        columns : list of str, optional
            Column names to include. If None, uses all columns.
        """
        if columns is not None:
            arrays = []
            for col in columns:
                arr = source[col]
                if hasattr(arr, 'cpu'):
                    arr = arr.cpu().numpy()
                arr = np.asarray(arr, dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                arrays.append(arr)
            data_array = np.hstack(arrays)
        elif 'data' in source:
            data_array = np.asarray(source['data'], dtype=np.float64)
        else:
            keys = list(source.keys())
            if not keys:
                raise ValidationError("DataSource has no columns")
            arrays = []
            for key in keys:
                arr = source[key]
                if hasattr(arr, 'cpu'):
                    arr = arr.cpu().numpy()
                arr = np.asarray(arr, dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                arrays.append(arr)
            data_array = np.hstack(arrays)

        return cls._build(data_array)

    @classmethod
    def _build(cls, data: NDArray) -> MVNDesign:
        """Internal builder with validation."""
        if data.ndim != 2:
            raise ValidationError(
                f"Data must be 2D (observations x variables), got {data.ndim}D"
            )

        n, p = data.shape

        if n < 2:
            raise ValidationError(f"Need at least 2 observations, got {n}")

        if p < 1:
            raise ValidationError(f"Need at least 1 variable, got {p}")

        # Check for inf values (NaN is expected)
        non_finite_mask = ~np.isfinite(data) & ~np.isnan(data)
        if np.any(non_finite_mask):
            loc = np.where(non_finite_mask)
            raise ValidationError(
                f"Data contains infinite values (first at row {loc[0][0]}, col {loc[1][0]})"
            )

        # Check for all-NaN rows
        all_nan_rows = np.all(np.isnan(data), axis=1)
        if np.any(all_nan_rows):
            raise ValidationError(
                f"Data contains {int(np.sum(all_nan_rows))} rows with all missing values"
            )

        # Check for all-NaN columns
        all_nan_cols = np.all(np.isnan(data), axis=0)
        if np.any(all_nan_cols):
            col_idx = np.where(all_nan_cols)[0][0]
            raise ValidationError(
                f"Variable at column {col_idx} is completely missing"
            )

        return cls(_data=data, _n=n, _p=p)

    @property
    def data(self) -> NDArray[np.floating[Any]]:
        """Data matrix (n x p), may contain NaN."""
        return self._data

    @property
    def n(self) -> int:
        """Number of observations."""
        return self._n

    @property
    def p(self) -> int:
        """Number of variables."""
        return self._p

    @property
    def n_missing(self) -> int:
        """Total number of missing values."""
        return int(np.sum(np.isnan(self._data)))

    @property
    def missing_rate(self) -> float:
        """Overall missing rate (0.0 to 1.0)."""
        return self.n_missing / (self._n * self._p)

    @property
    def has_missing(self) -> bool:
        """Whether data has any missing values."""
        return self.n_missing > 0

    def __repr__(self) -> str:
        return (
            f"MVNDesign(n={self._n}, p={self._p}, "
            f"missing_rate={self.missing_rate:.1%})"
        )
