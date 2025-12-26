"""
Universal DataSource for PyStatistics.

DataSource is the "I have data" abstraction. It doesn't know or care
what domain consumes it. It just provides data access.

Like a lumber yard: provides raw logs. Doesn't care if you're making
furniture, paper, or two-by-fours.

Usage:
    from pystatistics import DataSource
    
    ds = DataSource.from_arrays(X=X, y=y)
    ds = DataSource.from_file("data.csv")
    ds = DataSource.from_dataframe(df)
    ds = DataSource.from_tensors(X=X_gpu, y=y_gpu)
    
    # Access arrays
    ds.keys()  # frozenset({'X', 'y'})
    X = ds['X']
    y = ds['y']
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.capabilities import (
    CAPABILITY_MATERIALIZED,
    CAPABILITY_STREAMING,
    CAPABILITY_GPU_NATIVE,
    CAPABILITY_REPEATABLE,
    CAPABILITY_SUFFICIENT_STATISTICS,
)

if TYPE_CHECKING:
    import pandas as pd
    import torch


@dataclass
class DataSource:
    """
    Universal data container. Domain-agnostic.
    
    Construct via factory classmethods, not directly.
    
    The lumber yard analogy: DataSource has data (logs). It doesn't know
    or care what you're building—furniture (regression), paper (MVN MLE),
    or two-by-fours (survival analysis).
    """
    _data: dict[str, Any]
    _capabilities: frozenset[str]
    _metadata: dict[str, Any] = field(default_factory=dict)
    
    # === Array Access ===
    
    def keys(self) -> frozenset[str]:
        """
        Return the names of all available arrays.
        
        Returns:
            frozenset of array names
            
        Example:
            >>> ds = DataSource.from_arrays(X=X, y=y)
            >>> ds.keys()
            frozenset({'X', 'y'})
        """
        return frozenset(k for k in self._data.keys() if not k.startswith('_'))
    
    def __getitem__(self, key: str) -> Any:
        """
        Access a named array.
        
        Args:
            key: Array name
            
        Returns:
            The array
            
        Raises:
            KeyError: If key not found, with helpful message listing available keys
            
        Example:
            >>> ds = DataSource.from_arrays(X=X, y=y)
            >>> X = ds['X']
            >>> ds['Z']  # KeyError: "DataSource has no array 'Z'. Available: {'X', 'y'}"
        """
        if key not in self._data:
            available = self.keys()
            raise KeyError(
                f"DataSource has no array '{key}'. Available: {available}"
            )
        return self._data[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists."""
        return key in self._data
    
    # === Properties ===
    
    @property
    def n_observations(self) -> int:
        """Number of statistical units (rows)."""
        return self._metadata.get('n_observations', 0)
    
    @property
    def metadata(self) -> dict[str, Any]:
        """Domain-agnostic metadata."""
        return self._metadata.copy()
    
    def supports(self, capability: str) -> bool:
        """
        Check if this DataSource supports a capability.
        
        Args:
            capability: Use constants from pystatistics.core.capabilities
            
        Returns:
            True if supported, False otherwise
            
        Note:
            Unknown capabilities return False, never raise.
        """
        return capability in self._capabilities
    
    # === Factory Methods ===
    
    @classmethod
    def from_arrays(
        cls,
        *,
        X: NDArray | None = None,
        y: NDArray | None = None,
        data: NDArray | None = None,
        columns: list[str] | None = None,
        **named_arrays: NDArray,
    ) -> DataSource:
        """Construct from NumPy arrays."""
        storage: dict[str, Any] = {}
        n_obs: int | None = None
        
        if X is not None:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            storage['X'] = X
            n_obs = X.shape[0]
            
        if y is not None:
            y = np.asarray(y, dtype=np.float64)
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.ravel()
            storage['y'] = y
            n_obs = n_obs or y.shape[0]
        
        if data is not None:
            data = np.asarray(data, dtype=np.float64)
            n_obs = n_obs or data.shape[0]
            if columns is not None:
                for i, col in enumerate(columns):
                    storage[col] = data[:, i]
            else:
                storage['_data'] = data
        
        for name, arr in named_arrays.items():
            storage[name] = np.asarray(arr, dtype=np.float64)
            n_obs = n_obs or storage[name].shape[0]
        
        return cls(
            _data=storage,
            _capabilities=frozenset({CAPABILITY_MATERIALIZED, CAPABILITY_REPEATABLE}),
            _metadata={'n_observations': n_obs, 'source': 'arrays'},
        )
    
    @classmethod
    def from_file(cls, path: str | Path, *, columns: list[str] | None = None) -> DataSource:
        """Construct from file (CSV, NPY)."""
        path = Path(path)
        suffix = path.suffix.lower()
        
        if suffix in ('.csv', '.tsv'):
            import pandas as pd
            df = pd.read_csv(path, usecols=columns)
            return cls.from_dataframe(df, source_path=str(path))
        elif suffix == '.npy':
            data = np.load(path)
            return cls.from_arrays(data=data, columns=columns)
        else:
            raise ValidationError(f"Unknown file format: {suffix}")
    
    @classmethod
    def from_dataframe(cls, df: 'pd.DataFrame', *, source_path: str | None = None) -> DataSource:
        """Construct from pandas DataFrame."""
        storage: dict[str, Any] = {}
        
        for col in df.columns:
            storage[col] = df[col].to_numpy(dtype=np.float64)
        
        metadata = {
            'n_observations': len(df),
            'source': 'dataframe',
            'columns': list(df.columns),
        }
        if source_path:
            metadata['source_path'] = source_path
            
        return cls(
            _data=storage,
            _capabilities=frozenset({CAPABILITY_MATERIALIZED, CAPABILITY_REPEATABLE}),
            _metadata=metadata,
        )
    
    @classmethod
    def from_tensors(
        cls,
        *,
        X: 'torch.Tensor | None' = None,
        y: 'torch.Tensor | None' = None,
        **named_tensors: 'torch.Tensor',
    ) -> DataSource:
        """Construct from PyTorch tensors (already on GPU)."""
        import torch
        
        storage: dict[str, Any] = {}
        n_obs: int | None = None
        device: str | None = None
        
        if X is not None:
            storage['X'] = X
            n_obs = X.shape[0]
            device = str(X.device)
            
        if y is not None:
            storage['y'] = y
            n_obs = n_obs or y.shape[0]
            device = device or str(y.device)
        
        for name, tensor in named_tensors.items():
            storage[name] = tensor
            n_obs = n_obs or tensor.shape[0]
            device = device or str(tensor.device)
        
        capabilities = {CAPABILITY_MATERIALIZED, CAPABILITY_REPEATABLE}
        if device and device != 'cpu':
            capabilities.add(CAPABILITY_GPU_NATIVE)
        
        return cls(
            _data=storage,
            _capabilities=frozenset(capabilities),
            _metadata={
                'n_observations': n_obs,
                'source': 'tensors',
                'device': device,
            },
        )
    
    @classmethod  
    def build(cls, *args, **kwargs) -> DataSource:
        """
        Convenience factory that dispatches to appropriate from_* method.
        
        Examples:
            DataSource.build(X=X, y=y)  # from_arrays
            DataSource.build("data.csv")  # from_file
        """
        if args and isinstance(args[0], (str, Path)):
            return cls.from_file(args[0], **kwargs)
        else:
            return cls.from_arrays(**kwargs)
