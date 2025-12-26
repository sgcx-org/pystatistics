"""
Universal DataSource for PyStatistics.

DataSource is the "I have data" abstraction. It doesn't know or care
what domain consumes it. It just provides data access.

Like a lumber yard: provides raw logs. Doesn't care if you're making
furniture, paper, or two-by-fours.

Usage:
    from pystatistics import DataSource
    
    ds = DataSource.build(X, y)
    ds = DataSource.build("data.csv")
    ds = DataSource.from_arrays(X=X, y=y)
    ds = DataSource.from_file("data.csv")
    ds = DataSource.from_dataframe(df)
    ds = DataSource.from_tensors(X=X_gpu, y=y_gpu)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ValidationError

if TYPE_CHECKING:
    import pandas as pd
    import torch


CAPABILITY_MATERIALIZE = 'materialize'
CAPABILITY_STREAM = 'stream'
CAPABILITY_GPU_NATIVE = 'gpu_native'
CAPABILITY_SECOND_PASS = 'second_pass'


@dataclass
class DataSource:
    """
    Universal data container. Domain-agnostic.
    
    Construct via factory classmethods, not directly.
    """
    _data: dict[str, Any]
    _capabilities: frozenset[str]
    _metadata: dict[str, Any] = field(default_factory=dict)
    
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
            _capabilities=frozenset({CAPABILITY_MATERIALIZE, CAPABILITY_SECOND_PASS}),
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
            arr = df[col].to_numpy()
            if np.issubdtype(arr.dtype, np.number):
                arr = arr.astype(np.float64)
            storage[str(col)] = arr
        
        metadata = {'n_observations': len(df), 'source': 'dataframe', 'columns': list(df.columns)}
        if source_path:
            metadata['path'] = source_path
        
        return cls(
            _data=storage,
            _capabilities=frozenset({CAPABILITY_MATERIALIZE, CAPABILITY_SECOND_PASS}),
            _metadata=metadata,
        )
    
    @classmethod
    def from_tensors(cls, *, X: 'torch.Tensor | None' = None, y: 'torch.Tensor | None' = None, **tensors: 'torch.Tensor') -> DataSource:
        """Construct from PyTorch tensors."""
        import torch
        
        storage: dict[str, Any] = {}
        device = None
        n_obs = None
        
        if X is not None:
            storage['X'] = X
            device = X.device
            n_obs = X.shape[0]
        if y is not None:
            storage['y'] = y
            device = device or y.device
            n_obs = n_obs or y.shape[0]
        for name, t in tensors.items():
            storage[name] = t
            device = device or t.device
            n_obs = n_obs or t.shape[0]
        
        caps = {CAPABILITY_MATERIALIZE, CAPABILITY_SECOND_PASS}
        if device is not None and device.type == 'cuda':
            caps.add(CAPABILITY_GPU_NATIVE)
        
        return cls(
            _data=storage,
            _capabilities=frozenset(caps),
            _metadata={'n_observations': n_obs, 'source': 'tensors', 'device': str(device)},
        )
    
    @classmethod
    def build(cls, *args, **kwargs) -> DataSource:
        """Smart constructor - infers appropriate factory."""
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False
            torch = None
        
        if len(args) == 1 and not kwargs:
            arg = args[0]
            if isinstance(arg, (str, Path)):
                return cls.from_file(arg)
            if hasattr(arg, 'to_numpy'):
                return cls.from_dataframe(arg)
            if isinstance(arg, np.ndarray):
                return cls.from_arrays(data=arg)
        
        if len(args) == 2 and not kwargs:
            X, y = args
            if has_torch and isinstance(X, torch.Tensor):
                return cls.from_tensors(X=X, y=y)
            return cls.from_arrays(X=np.asarray(X), y=np.asarray(y))
        
        if kwargs:
            if has_torch and any(isinstance(v, torch.Tensor) for v in kwargs.values()):
                return cls.from_tensors(**kwargs)
            return cls.from_arrays(**kwargs)
        
        raise ValidationError("Could not infer DataSource type")
    
    # === Data Access ===
    
    @property
    def n_observations(self) -> int:
        return self._metadata.get('n_observations', 0)
    
    @property
    def columns(self) -> list[str]:
        return [k for k in self._data.keys() if not k.startswith('_')]
    
    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata.copy()
    
    def supports(self, capability: str) -> bool:
        return capability in self._capabilities
    
    def has(self, name: str) -> bool:
        return name in self._data
    
    def get(self, name: str) -> Any:
        if name not in self._data:
            raise KeyError(f"'{name}' not found. Available: {self.columns}")
        return self._data[name]
    
    def get_columns(self, names: list[str]) -> NDArray:
        """Stack multiple columns into matrix."""
        arrays = []
        for name in names:
            arr = self.get(name)
            if hasattr(arr, 'cpu'):
                arr = arr.cpu().numpy()
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            arrays.append(arr)
        return np.hstack(arrays)
    
    def __repr__(self) -> str:
        cols = ', '.join(self.columns[:5])
        if len(self.columns) > 5:
            cols += f', ... ({len(self.columns)} total)'
        return f"DataSource(n={self.n_observations}, columns=[{cols}])"
