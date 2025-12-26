#!/usr/bin/env python3
"""
PyStatistics Architecture Refactor Script
==========================================

This script refactors the pystatistics codebase according to the 
PyStatistics Architecture Document v1.0.

Changes:
1. Rename core/backends/ → core/compute/ (reserve "backend" for domain-specific)
2. Create core/capabilities.py (single source of truth for capability strings)
3. Add provenance field to Result
4. Add keys() and __getitem__ with helpful KeyError to DataSource
5. Update capability constants to new naming (materialized, streaming, repeatable)
6. Update LinearParams to make residuals/fitted_values Optional
7. Update all imports throughout the codebase

Usage:
    cd /path/to/pystatistics
    python refactor_architecture.py

    # Or dry-run first:
    python refactor_architecture.py --dry-run

Author: PyStatistics Team
Date: December 26, 2025
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

# =============================================================================
# Configuration
# =============================================================================

# Old import paths -> New import paths
IMPORT_REPLACEMENTS = [
    # core/backends -> core/compute
    ('from pystatistics.core.backends.device import', 'from pystatistics.core.compute.device import'),
    ('from pystatistics.core.backends.timing import', 'from pystatistics.core.compute.timing import'),
    ('from pystatistics.core.backends.linalg.qr import', 'from pystatistics.core.compute.linalg.qr import'),
    ('from pystatistics.core.backends.linalg.cholesky import', 'from pystatistics.core.compute.linalg.cholesky import'),
    ('from pystatistics.core.backends.linalg.svd import', 'from pystatistics.core.compute.linalg.svd import'),
    ('from pystatistics.core.backends.linalg.solve import', 'from pystatistics.core.compute.linalg.solve import'),
    ('from pystatistics.core.backends.linalg.determinant import', 'from pystatistics.core.compute.linalg.determinant import'),
    ('from pystatistics.core.backends.linalg import', 'from pystatistics.core.compute.linalg import'),
    ('from pystatistics.core.backends import', 'from pystatistics.core.compute import'),
    ('import pystatistics.core.backends', 'import pystatistics.core.compute'),
    
    # Old capability constants -> new capability constants (in datasource.py)
    ('CAPABILITY_MATERIALIZE', 'CAPABILITY_MATERIALIZED'),
    ('CAPABILITY_STREAM', 'CAPABILITY_STREAMING'),
    ('CAPABILITY_SECOND_PASS', 'CAPABILITY_REPEATABLE'),
    ('CAPABILITY_GPU_NATIVE', 'CAPABILITY_GPU_NATIVE'),  # unchanged
    
    # Update supports() calls with string literals
    ("supports('materialize')", "supports(CAPABILITY_MATERIALIZED)"),
    ("supports('stream')", "supports(CAPABILITY_STREAMING)"),
    ("supports('second_pass')", "supports(CAPABILITY_REPEATABLE)"),
    ("supports('gpu_native')", "supports(CAPABILITY_GPU_NATIVE)"),
    ("supports('sufficient_stats')", "supports(CAPABILITY_SUFFICIENT_STATISTICS)"),
]

# =============================================================================
# New File Contents
# =============================================================================

CAPABILITIES_PY = '''\
"""
Capability string constants for PyStatistics.

This module is the SINGLE SOURCE OF TRUTH for capability strings.
Import from here, never use raw strings.

Usage:
    from pystatistics.core.capabilities import (
        CAPABILITY_MATERIALIZED,
        CAPABILITY_STREAMING,
        CAPABILITY_REPEATABLE,
    )
    
    if ds.supports(CAPABILITY_MATERIALIZED):
        X, y = ds['X'], ds['y']
"""

# Data can be returned as full numpy arrays in memory
CAPABILITY_MATERIALIZED = 'materialized'

# Data can be yielded in batches (for datasets larger than RAM)
CAPABILITY_STREAMING = 'streaming'

# Data is already on GPU as PyTorch tensors
CAPABILITY_GPU_NATIVE = 'gpu_native'

# Data can be iterated multiple times (for residuals, fitted values)
CAPABILITY_REPEATABLE = 'repeatable'

# Sufficient statistics can be computed/provided directly
CAPABILITY_SUFFICIENT_STATISTICS = 'sufficient_statistics'

# All capabilities as a frozenset for validation
ALL_CAPABILITIES = frozenset({
    CAPABILITY_MATERIALIZED,
    CAPABILITY_STREAMING,
    CAPABILITY_GPU_NATIVE,
    CAPABILITY_REPEATABLE,
    CAPABILITY_SUFFICIENT_STATISTICS,
})

__all__ = [
    'CAPABILITY_MATERIALIZED',
    'CAPABILITY_STREAMING', 
    'CAPABILITY_GPU_NATIVE',
    'CAPABILITY_REPEATABLE',
    'CAPABILITY_SUFFICIENT_STATISTICS',
    'ALL_CAPABILITIES',
]
'''

COMPUTE_INIT_PY = '''\
"""
Shared compute infrastructure for PyStatistics.

This module provides hardware detection, timing utilities, and linear algebra
kernels that are shared across all domain-specific backends.

IMPORTANT: This is NOT where domain-specific backends live. Those go in
{domain}/backends/. This module contains shared NUMERIC infrastructure.

Submodules:
    device: Hardware detection and device selection
    timing: Execution timing utilities
    precision: Numerical precision constants and utilities
    linalg: Linear algebra kernels (QR, Cholesky, SVD, etc.)
"""

from pystatistics.core.compute.device import (
    DeviceInfo,
    detect_gpu,
    get_cpu_info,
    select_device,
)
from pystatistics.core.compute.timing import Timer, timed

__all__ = [
    # Device detection
    "DeviceInfo",
    "detect_gpu",
    "get_cpu_info",
    "select_device",
    # Timing
    "Timer",
    "timed",
]
'''

COMPUTE_LINALG_INIT_PY = '''\
"""
Linear algebra kernels for PyStatistics.

This module provides CPU and GPU implementations of core linear algebra
operations used across all statistical domains.

All functions follow these conventions:
    - CPU functions use NumPy/SciPy (LAPACK under the hood)
    - GPU functions use PyTorch and return NumPy arrays (data moved to CPU)
    - Each operation returns a structured result dataclass
    - Errors are raised immediately with clear messages

Submodules:
    qr: QR decomposition
    cholesky: Cholesky decomposition (stub)
    svd: Singular value decomposition (stub)
    solve: Triangular and symmetric solvers (stub)
    determinant: Log-determinant computation (stub)
"""

from pystatistics.core.compute.linalg.qr import (
    QRResult,
    qr_cpu,
    qr_gpu,
    qr_solve_cpu,
    qr_solve_gpu,
)

__all__ = [
    # QR decomposition
    "QRResult",
    "qr_cpu",
    "qr_gpu",
    "qr_solve_cpu",
    "qr_solve_gpu",
]
'''

# Updated Result class with provenance
RESULT_PY = '''\
"""
Generic result container for all PyStatistics computations.

The Result class provides a standardized envelope that all domain-specific
results use. This enables shared tooling for timing, logging, reproducibility,
and serialization while allowing domains to define their own parameter structures.

Design decisions:
    - Generic over parameter payload P for type safety
    - info dict for flexible metadata (converged, iterations, diagnostics)
    - timing is optional (don't burden unit tests)
    - provenance for reproducibility (versions, device, algorithm)
    - Immutable (frozen=True) for reproducibility
"""

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Any

P = TypeVar('P')  # Parameter payload type


def _default_provenance() -> dict[str, Any]:
    """Generate minimal provenance metadata."""
    import pystatistics
    provenance = {
        'pystatistics_version': pystatistics.__version__,
    }
    try:
        import numpy as np
        provenance['numpy_version'] = np.__version__
    except ImportError:
        pass
    try:
        import torch
        provenance['torch_version'] = torch.__version__
    except ImportError:
        pass
    return provenance


@dataclass(frozen=True)
class Result(Generic[P]):
    """
    Immutable result envelope for statistical computations.
    
    Type Parameters:
        P: The domain-specific parameter payload type
        
    Attributes:
        params: Domain-specific parameters (coefficients, estimates, etc.)
        info: Structured metadata (method, convergence, diagnostics)
        timing: Execution timing breakdown, or None if not measured
        backend_name: Identifier of the backend that produced this result
        warnings: Non-fatal issues encountered during computation
        provenance: Reproducibility metadata (versions, device, algorithm)
        
    The frozen=True ensures results are immutable after creation, which is
    important for reproducibility and prevents accidental modification.
    
    Examples:
        >>> # Direct method (no convergence notion)
        >>> Result(
        ...     params=LinearParams(coefficients=beta),
        ...     info={'method': 'qr', 'rank': 5},
        ...     timing={'total_seconds': 0.01},
        ...     backend_name='cpu_qr'
        ... )
        
        >>> # Iterative method
        >>> Result(
        ...     params=MVNParams(mu=mu, sigma=sigma),
        ...     info={'method': 'em', 'converged': True, 'iterations': 23},
        ...     timing={'total_seconds': 0.5, 'e_step': 0.3, 'm_step': 0.2},
        ...     backend_name='cpu_em'
        ... )
    """
    params: P
    info: dict[str, Any]
    timing: dict[str, float] | None
    backend_name: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict[str, Any] = field(default_factory=_default_provenance)
    
    def has_warning(self, substring: str) -> bool:
        """Check if any warning contains the given substring."""
        return any(substring in w for w in self.warnings)
'''

# Updated DataSource with keys() and __getitem__
DATASOURCE_PY = '''\
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
'''

# Updated core/__init__.py
CORE_INIT_PY = '''\
"""
Core infrastructure for PyStatistics.
"""

from pystatistics.core.datasource import DataSource
from pystatistics.core.result import Result
from pystatistics.core.capabilities import (
    CAPABILITY_MATERIALIZED,
    CAPABILITY_STREAMING,
    CAPABILITY_GPU_NATIVE,
    CAPABILITY_REPEATABLE,
    CAPABILITY_SUFFICIENT_STATISTICS,
)
from pystatistics.core.exceptions import (
    PyStatisticsError,
    ValidationError,
    DimensionError,
    NumericalError,
    SingularMatrixError,
    NotPositiveDefiniteError,
    ConvergenceError,
)

__all__ = [
    "DataSource",
    "Result",
    # Capabilities
    "CAPABILITY_MATERIALIZED",
    "CAPABILITY_STREAMING",
    "CAPABILITY_GPU_NATIVE",
    "CAPABILITY_REPEATABLE",
    "CAPABILITY_SUFFICIENT_STATISTICS",
    # Exceptions
    "PyStatisticsError",
    "ValidationError",
    "DimensionError",
    "NumericalError",
    "SingularMatrixError",
    "NotPositiveDefiniteError",
    "ConvergenceError",
]
'''

# Updated regression/design.py imports
DESIGN_PY_HEADER = '''\
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
'''

# Updated regression/backends/cpu.py imports
CPU_BACKEND_IMPORTS = '''\
"""
CPU reference backend for linear regression.
"""

from typing import Any
import numpy as np

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.core.compute.linalg.qr import qr_solve_cpu, qr_cpu
from pystatistics.regression.design import Design
from pystatistics.regression.solution import LinearParams
'''

# Updated regression/solvers.py imports  
SOLVERS_IMPORTS = '''\
"""
Solver dispatch for regression.
"""

from typing import Literal
import numpy as np
from numpy.typing import ArrayLike

from pystatistics.core.compute.device import select_device
from pystatistics.regression.design import Design
from pystatistics.regression.solution import LinearSolution
from pystatistics.regression.backends.cpu import CPUQRBackend
'''

# =============================================================================
# Refactoring Functions
# =============================================================================

def find_pystatistics_root() -> Path:
    """Find the pystatistics package root directory."""
    cwd = Path.cwd()
    
    # Check if we're in the root
    if (cwd / 'pystatistics' / '__init__.py').exists():
        return cwd
    
    # Check if we're in pystatistics/
    if (cwd / '__init__.py').exists() and cwd.name == 'pystatistics':
        return cwd.parent
    
    raise FileNotFoundError(
        "Cannot find pystatistics package root. "
        "Run this script from the repository root."
    )


def backup_directory(src: Path, dry_run: bool) -> Path:
    """Create a backup of a directory."""
    backup_path = src.parent / f"{src.name}_backup"
    if dry_run:
        print(f"  [DRY-RUN] Would backup {src} -> {backup_path}")
        return backup_path
    
    if backup_path.exists():
        shutil.rmtree(backup_path)
    shutil.copytree(src, backup_path)
    print(f"  Backed up {src} -> {backup_path}")
    return backup_path


def rename_directory(old: Path, new: Path, dry_run: bool) -> None:
    """Rename a directory."""
    if dry_run:
        print(f"  [DRY-RUN] Would rename {old} -> {new}")
        return
    
    if old.exists():
        old.rename(new)
        print(f"  Renamed {old} -> {new}")


def write_file(path: Path, content: str, dry_run: bool) -> None:
    """Write content to a file."""
    if dry_run:
        print(f"  [DRY-RUN] Would write {path}")
        return
    
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"  Wrote {path}")


def update_imports_in_file(filepath: Path, replacements: List[Tuple[str, str]], dry_run: bool) -> int:
    """Update imports in a single file. Returns number of replacements made."""
    if not filepath.exists():
        return 0
    
    content = filepath.read_text()
    original = content
    count = 0
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            count += content.count(new) - original.count(new)
    
    if content != original:
        if dry_run:
            print(f"  [DRY-RUN] Would update {filepath} ({count} replacements)")
        else:
            filepath.write_text(content)
            print(f"  Updated {filepath} ({count} replacements)")
    
    return count


def find_python_files(root: Path) -> List[Path]:
    """Find all Python files in directory tree."""
    return list(root.rglob("*.py"))


# =============================================================================
# Main Refactoring Steps
# =============================================================================

def step1_rename_backends_to_compute(root: Path, dry_run: bool) -> None:
    """Step 1: Rename core/backends/ to core/compute/"""
    print("\n" + "=" * 70)
    print("STEP 1: Rename core/backends/ -> core/compute/")
    print("=" * 70)
    
    backends_dir = root / 'pystatistics' / 'core' / 'backends'
    compute_dir = root / 'pystatistics' / 'core' / 'compute'
    
    if not backends_dir.exists():
        if compute_dir.exists():
            print("  Already renamed (core/compute/ exists)")
            return
        else:
            print("  ERROR: Neither core/backends/ nor core/compute/ found!")
            return
    
    # Backup first
    backup_directory(backends_dir, dry_run)
    
    # Rename
    rename_directory(backends_dir, compute_dir, dry_run)


def step2_create_capabilities_module(root: Path, dry_run: bool) -> None:
    """Step 2: Create core/capabilities.py"""
    print("\n" + "=" * 70)
    print("STEP 2: Create core/capabilities.py")
    print("=" * 70)
    
    capabilities_path = root / 'pystatistics' / 'core' / 'capabilities.py'
    write_file(capabilities_path, CAPABILITIES_PY, dry_run)


def step3_update_compute_init(root: Path, dry_run: bool) -> None:
    """Step 3: Update core/compute/__init__.py"""
    print("\n" + "=" * 70)
    print("STEP 3: Update core/compute/__init__.py")
    print("=" * 70)
    
    init_path = root / 'pystatistics' / 'core' / 'compute' / '__init__.py'
    write_file(init_path, COMPUTE_INIT_PY, dry_run)
    
    # Also update linalg/__init__.py
    linalg_init = root / 'pystatistics' / 'core' / 'compute' / 'linalg' / '__init__.py'
    write_file(linalg_init, COMPUTE_LINALG_INIT_PY, dry_run)


def step4_update_result(root: Path, dry_run: bool) -> None:
    """Step 4: Update core/result.py with provenance field"""
    print("\n" + "=" * 70)
    print("STEP 4: Update core/result.py (add provenance)")
    print("=" * 70)
    
    result_path = root / 'pystatistics' / 'core' / 'result.py'
    write_file(result_path, RESULT_PY, dry_run)


def step5_update_datasource(root: Path, dry_run: bool) -> None:
    """Step 5: Update core/datasource.py with keys() and __getitem__"""
    print("\n" + "=" * 70)
    print("STEP 5: Update core/datasource.py (add keys(), __getitem__)")
    print("=" * 70)
    
    datasource_path = root / 'pystatistics' / 'core' / 'datasource.py'
    write_file(datasource_path, DATASOURCE_PY, dry_run)


def step6_update_core_init(root: Path, dry_run: bool) -> None:
    """Step 6: Update core/__init__.py"""
    print("\n" + "=" * 70)
    print("STEP 6: Update core/__init__.py")
    print("=" * 70)
    
    init_path = root / 'pystatistics' / 'core' / '__init__.py'
    write_file(init_path, CORE_INIT_PY, dry_run)


def step7_update_all_imports(root: Path, dry_run: bool) -> None:
    """Step 7: Update all imports throughout codebase"""
    print("\n" + "=" * 70)
    print("STEP 7: Update imports throughout codebase")
    print("=" * 70)
    
    pystatistics_dir = root / 'pystatistics'
    tests_dir = root / 'tests'
    
    all_files = find_python_files(pystatistics_dir)
    if tests_dir.exists():
        all_files.extend(find_python_files(tests_dir))
    
    total_replacements = 0
    for filepath in all_files:
        # Skip the files we just wrote
        if filepath.name in ('capabilities.py', 'datasource.py', 'result.py'):
            if 'core' in str(filepath):
                continue
        
        count = update_imports_in_file(filepath, IMPORT_REPLACEMENTS, dry_run)
        total_replacements += count
    
    print(f"\n  Total import replacements: {total_replacements}")


def step8_update_regression_design(root: Path, dry_run: bool) -> None:
    """Step 8: Update regression/design.py header"""
    print("\n" + "=" * 70)
    print("STEP 8: Update regression/design.py imports")
    print("=" * 70)
    
    design_path = root / 'pystatistics' / 'regression' / 'design.py'
    if not design_path.exists():
        print(f"  WARNING: {design_path} not found")
        return
    
    content = design_path.read_text()
    
    # Find where the imports end (look for the first class or function definition)
    match = re.search(r'^(@dataclass|class |def )', content, re.MULTILINE)
    if match:
        # Get everything after imports
        rest = content[match.start():]
        new_content = DESIGN_PY_HEADER + '\n\n' + rest
        
        if dry_run:
            print(f"  [DRY-RUN] Would update {design_path}")
        else:
            design_path.write_text(new_content)
            print(f"  Updated {design_path}")


def step9_verify_structure(root: Path) -> None:
    """Step 9: Verify the new structure"""
    print("\n" + "=" * 70)
    print("STEP 9: Verify new structure")
    print("=" * 70)
    
    expected_files = [
        'pystatistics/core/capabilities.py',
        'pystatistics/core/datasource.py',
        'pystatistics/core/result.py',
        'pystatistics/core/compute/__init__.py',
        'pystatistics/core/compute/device.py',
        'pystatistics/core/compute/timing.py',
        'pystatistics/core/compute/linalg/__init__.py',
        'pystatistics/core/compute/linalg/qr.py',
    ]
    
    missing = []
    for f in expected_files:
        path = root / f
        if path.exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} (MISSING)")
            missing.append(f)
    
    should_not_exist = [
        'pystatistics/core/backends/',
    ]
    
    for f in should_not_exist:
        path = root / f
        if path.exists():
            print(f"  ⚠ {f} still exists (should be removed)")
        else:
            print(f"  ✓ {f} removed")
    
    if missing:
        print(f"\n  WARNING: {len(missing)} files missing!")
    else:
        print("\n  All expected files present!")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Refactor PyStatistics according to Architecture Document v1.0'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be done without making changes'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("PYSTATISTICS ARCHITECTURE REFACTOR")
    print("=" * 70)
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")
    
    try:
        root = find_pystatistics_root()
        print(f"Found pystatistics root: {root}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    
    # Execute all steps
    step1_rename_backends_to_compute(root, args.dry_run)
    step2_create_capabilities_module(root, args.dry_run)
    step3_update_compute_init(root, args.dry_run)
    step4_update_result(root, args.dry_run)
    step5_update_datasource(root, args.dry_run)
    step6_update_core_init(root, args.dry_run)
    step7_update_all_imports(root, args.dry_run)
    step8_update_regression_design(root, args.dry_run)
    
    if not args.dry_run:
        step9_verify_structure(root)
    
    print("\n" + "=" * 70)
    if args.dry_run:
        print("DRY RUN COMPLETE - Run without --dry-run to apply changes")
    else:
        print("REFACTOR COMPLETE")
        print("\nNext steps:")
        print("  1. Run tests: pytest tests/")
        print("  2. Fix any remaining import errors")
        print("  3. Commit: git add . && git commit -m 'Refactor: core/backends -> core/compute'")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    exit(main())