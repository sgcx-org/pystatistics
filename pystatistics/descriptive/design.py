"""
DescriptiveDesign: data wrapper for descriptive statistics.

Wraps a data matrix and provides validation and metadata for
the descriptive statistics pipeline. Follows pystatistics Design pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ValidationError


@dataclass(frozen=True)
class DescriptiveDesign:
    """
    Design for descriptive statistics.

    Wraps a data matrix (n observations x p variables) that may contain
    NaN values representing missing data. Immutable after construction.

    Construction:
        DescriptiveDesign.from_array(data)
        DescriptiveDesign.from_datasource(ds, columns=['a', 'b', 'c'])
    """
    _data: NDArray[np.floating[Any]]
    _n: int
    _p: int
    _columns: tuple[str, ...] | None

    @classmethod
    def from_array(cls, data) -> DescriptiveDesign:
        """
        Build DescriptiveDesign from array-like data.

        Parameters
        ----------
        data : array-like
            1D or 2D data matrix. Can be numpy array, pandas DataFrame,
            or any array-like with .values attribute. 1D input is reshaped
            to (n, 1).
        """
        if hasattr(data, 'values'):
            columns = tuple(str(c) for c in data.columns) if hasattr(data, 'columns') else None
            data_array = np.asarray(data.values, dtype=np.float64)
        else:
            columns = None
            data_array = np.asarray(data, dtype=np.float64)

        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)

        return cls._build(data_array, columns=columns)

    @classmethod
    def from_datasource(cls, source, *, columns: list[str] | None = None) -> DescriptiveDesign:
        """
        Build DescriptiveDesign from a DataSource.

        Parameters
        ----------
        source : DataSource
            Data source providing columns.
        columns : list of str, optional
            Column names to include. If None, uses all columns.
        """
        if columns is not None:
            col_names = tuple(columns)
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
            col_names = None
            data_array = np.asarray(source['data'], dtype=np.float64)
        else:
            keys = list(source.keys())
            if not keys:
                raise ValidationError("DataSource has no columns")
            col_names = tuple(keys)
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

        return cls._build(data_array, columns=col_names)

    @classmethod
    def _build(
        cls,
        data: NDArray,
        columns: tuple[str, ...] | None = None,
    ) -> DescriptiveDesign:
        """Internal builder with validation."""
        if data.ndim != 2:
            raise ValidationError(
                f"Data must be 2D (observations x variables), got {data.ndim}D"
            )

        n, p = data.shape

        if n < 1:
            raise ValidationError(f"Need at least 1 observation, got {n}")

        if p < 1:
            raise ValidationError(f"Need at least 1 variable, got {p}")

        # Check for inf values (NaN is expected and handled by use= parameter)
        non_finite_mask = np.isinf(data)
        if np.any(non_finite_mask):
            loc = np.where(non_finite_mask)
            raise ValidationError(
                f"Data contains infinite values (first at row {loc[0][0]}, col {loc[1][0]})"
            )

        return cls(_data=data, _n=n, _p=p, _columns=columns)

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
    def columns(self) -> tuple[str, ...] | None:
        """Column names, or None if not available."""
        return self._columns

    @property
    def n_missing(self) -> int:
        """Total number of missing values."""
        return int(np.sum(np.isnan(self._data)))

    @property
    def has_missing(self) -> bool:
        """Whether data has any missing values."""
        return bool(np.any(np.isnan(self._data)))

    def __repr__(self) -> str:
        missing = f", missing={self.n_missing}" if self.has_missing else ""
        return f"DescriptiveDesign(n={self._n}, p={self._p}{missing})"
