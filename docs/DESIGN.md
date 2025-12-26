# PyStatistics Architecture Document

**Software Version:** 1.0
**Document Version:** 1.2  
**Date:** December 26, 2025  
**Author:** PyStatistics Team  
**Status:** Final Design Specification

---

## Implementation Status

This section tracks what has been built vs. what is planned. The architecture described in this document is the **target specification**; implementation proceeds incrementally.

### Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Implemented and validated |
| 🔨 | In progress |
| 📋 | Planned (architecture defined, not yet built) |
| ❌ | Not planned for v1.0 |

### Core Infrastructure (`pystatistics/core/`)

| Component | Status | Notes |
|-----------|--------|-------|
| `datasource.py` | ✅ | `DataSource` class with factory methods |
| `capabilities.py` | 📋 | Capability constants (needs creation) |
| `result.py` | ✅ | `Result[P]` generic envelope |
| `protocols.py` | ✅ | `DataSource`, `Backend` protocols |
| `exceptions.py` | ✅ | Full exception hierarchy |
| `validation.py` | ✅ | Input validators |
| `compute/device.py` | 🔨 | Rename from `backends/device.py` |
| `compute/timing.py` | 🔨 | Rename from `backends/timing.py` |
| `compute/linalg/qr.py` | 🔨 | Rename from `backends/linalg/qr.py` |
| `compute/linalg/cholesky.py` | 📋 | Cholesky decomposition |
| `compute/linalg/svd.py` | 📋 | SVD decomposition |

**Pending Refactor:** Rename `core/backends/` → `core/compute/` to reserve "backend" for domain-specific Backend protocol implementations.

### Regression (`pystatistics/regression/`)

| Component | Status | Notes |
|-----------|--------|-------|
| `design.py` | ✅ | `Design` wrapper class |
| `solution.py` | ✅ | `LinearSolution`, `LinearParams` |
| `solvers.py` | ✅ | `fit()` dispatcher |
| `backends/cpu.py` | ✅ | CPU QR backend, validated against R |
| `backends/gpu.py` | 📋 | GPU PyTorch backend |
| GLM support | 📋 | Generalized linear models |
| Ridge regression | 📋 | L2 regularization |
| Streaming designs | 📋 | HDF5/chunked data |

### MVN MLE (`pystatistics/mvnmle/`)

| Component | Status | Notes |
|-----------|--------|-------|
| `data.py` | 📋 | `MVNData` wrapper class (not yet ported) |
| `solution.py` | 📋 | `MVNSolution`, `MVNParams` |
| `backends/cpu.py` | 📋 | Stub exists; port from PyMVNMLE |
| `backends/gpu.py` | 📋 | Stub exists |
| `datasets.py` | 📋 | Reference datasets (apple, missvals) |

**Note:** PyMVNMLE exists as a standalone validated package. The work here is to **port** it into the PyStatistics umbrella architecture, refactoring to use `DataSource` → `MVNData` → `Result[MVNParams]` pattern.

### Future Domains

| Domain | Status | Notes |
|--------|--------|-------|
| `survival/` | 📋 | Cox PH, Kaplan-Meier |
| `mixed/` | 📋 | LMM, GLMM |
| `timeseries/` | ❌ | Not planned for v1.0 |
| `hypothesis/` | ❌ | Not planned for v1.0 |

### Validation Status

| Test Suite | Status | Notes |
|------------|--------|-------|
| Regression vs R `lm()` | ✅ | Coefficients match to 1e-12 |
| MVN MLE vs R `mvnmle` | ✅ | In PyMVNMLE standalone; needs port |
| GPU ≡ CPU equivalence | 📋 | Pending GPU backend implementation |

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Implementation Status](#implementation-status)
3. [Design Principles](#design-principles)
4. [Core Abstractions](#core-abstractions)
5. [System Architecture](#system-architecture)
6. [Domain Extensions](#domain-extensions)
7. [Component Specifications](#component-specifications)
8. [Data Flow](#data-flow)
9. [API Reference](#api-reference)
10. [Implementation Guidelines](#implementation-guidelines)
11. [Testing Strategy](#testing-strategy)
12. [Performance Considerations](#performance-considerations)
13. [Future Extensions](#future-extensions)

---

## Executive Summary

PyStatistics is a unified GPU-accelerated statistical computing ecosystem designed for regulatory-grade statistical inference at scale. It provides a coherent umbrella architecture that enables multiple specialized statistical domains—regression, maximum likelihood estimation, survival analysis, and more—to share infrastructure while maintaining domain-specific APIs.

### The Problem

The scientific computing ecosystem lacks tools that simultaneously provide:
- Exact statistical inference on datasets too large for R/SAS
- Numerical correctness suitable for FDA submissions
- GPU acceleration without sacrificing statistical rigor
- A unified API across statistical domains

### The Solution

PyStatistics addresses this through a layered architecture:

1. **Core Layer** - Domain-agnostic infrastructure (DataSource, Result, Backend protocols)
2. **Domain Layer** - Specialized implementations (regression, mvnmle, survival)
3. **Backend Layer** - Hardware-specific computation (CPU reference, GPU acceleration)

### Key Innovation

Two orthogonal abstractions separate concerns cleanly across ALL statistical domains:

| Abstraction | Purpose | Example (Regression) | Example (MVN MLE) |
|-------------|---------|---------------------|-------------------|
| **DataSource** | "I have data" (the lumber yard) | Arrays, files, streams | Arrays, files, streams |
| **Domain Wrapper** | "I know what I need" (the buyer) | `Design` (X matrix + y) | `MVNData` (data + missingness) |
| **Backend** | "How to compute" | QR, Cholesky, GPU | EM, direct ML, GPU |
| **Result** | "What we found" | β̂, SE, residuals | μ̂, Σ̂, log-likelihood |

**The Lumber Yard Analogy:**
- **DataSource** = The lumber yard. It has logs. It doesn't care if you're making furniture, paper, or two-by-fours.
- **Domain Wrapper** (Design, MVNData, SurvivalData) = The buyer sent by the manufacturer. This person knows exactly what kind of logs are needed to make chairs. They visit the lumber yard, select the appropriate materials, and bring them back in the right form for their domain.

This enables:
- **Same data loading** for 1,000-row and 1,000,000-row datasets
- **Shared infrastructure** (timing, device detection, validation)
- **Domain-specific semantics** without contorting the API
- **GPU acceleration** without changing statistical estimators
- **Regulatory compliance** through explicit quality modes

### Target Audience

- Biostatisticians performing clinical trial analyses
- Computational biologists running GWAS on UK Biobank-scale data
- Quantitative researchers needing exact inference at scale
- Organizations requiring FDA-compliant statistical software

### Package Ecosystem

```
pystatistics/                 # Umbrella package (pip install pystatistics)
├── core/                     # Shared infrastructure
├── regression/               # Linear/GLM models
├── mvnmle/                   # Multivariate normal MLE
├── survival/                 # Survival analysis (planned)
├── timeseries/               # Time series (planned)
└── mixed/                    # Mixed models (planned)
```

---

## Implementation Status

This section tracks what has been built vs. what is planned. The architecture described in this document is the **target specification**; implementation proceeds incrementally.

### Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Implemented and validated |
| 🔨 | In progress |
| 📋 | Planned (architecture defined, not yet built) |
| ❌ | Not planned for v1.0 |

### Core Infrastructure (`pystatistics/core/`)

| Component | Status | Notes |
|-----------|--------|-------|
| `datasource.py` | ✅ | `DataSource` class with factory methods |
| `result.py` | ✅ | `Result[P]` generic envelope |
| `protocols.py` | ✅ | `DataSource`, `Backend` protocols |
| `exceptions.py` | ✅ | Full exception hierarchy |
| `validation.py` | ✅ | Input validators |
| `backends/device.py` | ✅ | GPU detection, device selection |
| `backends/timing.py` | ✅ | `Timer` class |
| `backends/linalg/qr.py` | ✅ | QR decomposition (CPU) |
| `backends/linalg/cholesky.py` | 📋 | Cholesky decomposition |
| `backends/linalg/svd.py` | 📋 | SVD decomposition |

### Regression (`pystatistics/regression/`)

| Component | Status | Notes |
|-----------|--------|-------|
| `design.py` | ✅ | `Design` wrapper class |
| `solution.py` | ✅ | `LinearSolution`, `LinearParams` |
| `solvers.py` | ✅ | `fit()` dispatcher |
| `backends/cpu.py` | ✅ | CPU QR backend, validated against R |
| `backends/gpu.py` | 📋 | GPU PyTorch backend |
| GLM support | 📋 | Generalized linear models |
| Ridge regression | 📋 | L2 regularization |
| Streaming designs | 📋 | HDF5/chunked data |

### MVN MLE (`pystatistics/mvnmle/`)

| Component | Status | Notes |
|-----------|--------|-------|
| `data.py` | 📋 | `MVNData` wrapper class (not yet ported) |
| `solution.py` | 📋 | `MVNSolution`, `MVNParams` |
| `backends/cpu.py` | 📋 | Stub exists; port from PyMVNMLE |
| `backends/gpu.py` | 📋 | Stub exists |
| `datasets.py` | 📋 | Reference datasets (apple, missvals) |

**Note:** PyMVNMLE exists as a standalone validated package. The work here is to **port** it into the PyStatistics umbrella architecture, refactoring to use `DataSource` → `MVNData` → `Result[MVNParams]` pattern.

### Future Domains

| Domain | Status | Notes |
|--------|--------|-------|
| `survival/` | 📋 | Cox PH, Kaplan-Meier |
| `mixed/` | 📋 | LMM, GLMM |
| `timeseries/` | ❌ | Not planned for v1.0 |
| `hypothesis/` | ❌ | Not planned for v1.0 |

### Validation Status

| Test Suite | Status | Notes |
|------------|--------|-------|
| Regression vs R `lm()` | ✅ | Coefficients match to 1e-12 |
| MVN MLE vs R `mvnmle` | ✅ | In PyMVNMLE standalone; needs port |
| GPU ≡ CPU equivalence | 📋 | Pending GPU backend implementation |

---

## Design Principles

### 1. Correctness > Fidelity > Performance > Convenience

**Principle:** The hierarchy of priorities is absolute and non-negotiable.

| Priority | Meaning | Example |
|----------|---------|---------|
| **Correctness** | Results are mathematically correct | Never silently switch estimators |
| **Fidelity** | Match R reference implementations | CPU backends bit-identical to R |
| **Performance** | Fast computation | GPU acceleration after correctness proven |
| **Convenience** | Nice API | Add sugar only after core is solid |

**Implication:** If OLS is ill-conditioned, we *refuse* (unless `unsafe=True`), we don't secretly switch to ridge.

### 2. Explicit Linear Algebra Over DSLs

**Principle:** Users pass design matrices directly, not formula strings.

**Rationale:**
- Formula parsing is complex, error-prone, and unnecessary
- Modern statisticians understand linear algebra
- Direct matrix passing eliminates an entire layer of bugs
- Simplifies validation and testing

**Implication:** No `"y ~ x1 + x2"` interface in v1. Users construct matrices explicitly.

### 3. One Abstraction, Many Representations

**Principle:** Dense, streamed, GPU-resident, and distributed are *representations*, not *APIs*.

**Rationale:**
- User shouldn't change code when data gets bigger
- Access pattern is an implementation detail
- Same mathematics should have same interface

**Implication:** `fit(design, ...)` works identically whether design is 1KB or 1TB.

### 4. Exact Math or Explicit Regularization

**Principle:** Never silently substitute one estimator for another.

**Rationale:**
- OLS and Ridge are different estimators with different properties
- Regulatory submissions require documenting exact methods used
- "Helpful" substitutions break statistical validity
- Automated pipelines depend on deterministic behavior

**Implication:** The system MUST REFUSE to silently modify the requested computation. Users must explicitly consent to any regularization.

### 5. Trust Your Neighbors

**Principle:** Each layer trusts the next layer's inputs.

**Rationale:**
- Validation happens ONCE at entry points
- Internal functions assume clean inputs
- Eliminates redundant checking
- Makes code easier to reason about

**Implication:** Backends don't re-validate what solvers already validated. Solvers don't re-validate what the public API validated.

### 6. Fail Fast, Fail Loud

**Principle:** No defaults except at top-level API. No silent fallbacks.

**Rationale:**
- Explicit is better than implicit
- Errors should be impossible to miss
- Each layer has one responsibility

**Implication:** Internal functions require ALL parameters explicitly. No "helpful" behavior that might mask problems.

### 7. Minimal Surface Area

**Principle:** One constructor, one solver, maximum functionality.

**Rationale:**
- Fewer ways to do things = fewer ways to do things wrong
- Easier to document, test, and maintain
- Reduces cognitive load on users

**Implication:** 
- `DataSource.from_arrays(...)`, `DataSource.from_file(...)` - factory methods for data
- `Design.from_datasource(...)` - domain-specific wrapper
- `fit(design, ...)` - single entry point per domain

### 8. Domain-Invariant Core, Domain-Specific Extensions

**Principle:** The core prescribes only what's truly universal; domains add their own semantics.

**Rationale:**
- Regression, MVN MLE, and survival analysis have different data structures
- Forcing a common "DesignMatrix" on MVN MLE would be contortion
- Capability-based data handles are truly domain-invariant

**Implication:** `DataSource` is minimal (n_observations, metadata, supports()). Domain-specific `Design` classes add the rest.

---

## Core Abstractions

### The DataSource Abstraction

**Purpose:** "I have data" - domain-agnostic data container.

**Philosophy:** The lumber yard. It has logs (data). It doesn't know or care what you're building—furniture (regression), paper (MVN MLE), or two-by-fours (survival analysis). It just provides raw materials in a consistent way.

### The Domain Wrapper Abstraction

**Purpose:** "I know what I need" - domain-specific data selector.

**Philosophy:** The buyer sent by the manufacturer. When the furniture company needs wood for chairs, they don't send a generic "go get wood" message—they send someone who knows exactly what species, dimensions, and grain patterns are required. This buyer visits the lumber yard (DataSource), selects the appropriate materials, and brings them back in the form the factory (backend) expects.

**Domain-Specific Names:**
| Domain | Wrapper Name | Rationale |
|--------|--------------|-----------|
| Regression | `Design` | Statistical term "design matrix" |
| MVN MLE | `MVNData` | Describes the data structure |
| Survival | `SurvivalData` | Describes the data structure |
| Mixed Models | `MixedData` | Describes the data structure |

**Why "Design" only for regression:** The term "design matrix" is specific to regression and experimental design. Using "Design" for MVN MLE or survival analysis would be a category error—those domains don't have design matrices in the statistical sense.

```python
from pystatistics import DataSource

# Multiple ways to create - same abstraction
ds = DataSource.from_arrays(X=X, y=y)
ds = DataSource.from_file("data.csv")
ds = DataSource.from_dataframe(df)
ds = DataSource.from_tensors(X=X_gpu, y=y_gpu)  # Already on GPU
```

**Key Properties:**
- `n_observations: int` - Number of statistical units
- `metadata: dict` - Domain-agnostic metadata
- `keys() -> frozenset[str]` - Available array names
- `supports(capability: str) -> bool` - Capability checking

**Array Access (Constrained Dict-Like Interface):**

DataSource provides dict-like access via `ds['X']`, but this is deliberately constrained:

```python
# DataSource knows its schema
ds = DataSource.from_arrays(X=X, y=y)
ds.keys()  # frozenset({'X', 'y'})

# Valid access
X = ds['X']  # Returns numpy array
y = ds['y']  # Returns numpy array

# Invalid access raises helpful KeyError
ds['Z']  # KeyError: "DataSource has no array 'Z'. Available: {'X', 'y'}"
```

**Why constrained access matters:**
- Prevents silent `'X'` vs `'x'` bugs
- Explicit schema makes the DataSource introspectable
- Domains know exactly what arrays are available before extraction

**Standard Capabilities:**

Capabilities are defined in `pystatistics/core/capabilities.py` as the single source of truth. All capability strings use consistent naming (nouns or adjectives describing state).

```python
# pystatistics/core/capabilities.py
"""Capability string constants. Import from here, never use raw strings."""

CAPABILITY_MATERIALIZED = 'materialized'      # Can return full data as numpy arrays
CAPABILITY_STREAMING = 'streaming'            # Can yield data in batches  
CAPABILITY_GPU_NATIVE = 'gpu_native'          # Can return data as PyTorch tensors on device
CAPABILITY_REPEATABLE = 'repeatable'          # Supports multiple iterations over data
CAPABILITY_SUFFICIENT_STATISTICS = 'sufficient_statistics'  # Can compute sufficient stats directly
```

| Capability | Meaning |
|------------|---------|
| `'materialized'` | Can return full data as numpy arrays in memory |
| `'streaming'` | Can yield data in batches (for datasets larger than RAM) |
| `'gpu_native'` | Can return data as PyTorch tensors already on device |
| `'repeatable'` | Supports multiple iterations over data (for residuals, etc.) |
| `'sufficient_statistics'` | Can compute sufficient statistics directly |

**Critical Design Decision:** Unknown capabilities MUST return `False`, never raise. This allows forward-compatible capability checking as new capabilities are added.

**Usage:**
```python
from pystatistics.core.capabilities import CAPABILITY_MATERIALIZED, CAPABILITY_REPEATABLE

if ds.supports(CAPABILITY_MATERIALIZED):
    X, y = ds.get_arrays('X', 'y')
```

### The Result Abstraction

**Purpose:** Standardized envelope for all statistical computations.

**Philosophy:** All domains return results in a common wrapper that enables shared tooling (timing, logging, reproducibility, serialization) while allowing domain-specific parameter structures.

```python
from dataclasses import dataclass, field
from typing import TypeVar, Generic

P = TypeVar('P')  # Parameter payload type

@dataclass(frozen=True)
class Result(Generic[P]):
    params: P                          # Domain-specific parameters
    info: dict[str, Any]               # Method, convergence, diagnostics
    timing: dict[str, float] | None    # Execution timing (optional)
    backend_name: str                  # Which backend produced this
    warnings: tuple[str, ...]          # Non-fatal issues
    provenance: dict[str, Any] = field(default_factory=dict)  # Reproducibility metadata
```

**Key Properties:**
- **Generic over P** - Type safety for domain-specific payloads
- **Immutable** - `frozen=True` prevents accidental modification
- **Timing optional** - Don't burden unit tests with timing requirements
- **Provenance for reproducibility** - Critical for regulatory compliance

**Provenance Field:**

The `provenance` dict captures metadata needed for reproducibility and regulatory audit trails:

```python
provenance = {
    'pystatistics_version': '0.1.0',
    'numpy_version': '1.24.0',
    'torch_version': '2.0.0',        # If GPU used
    'device': 'cuda:0',               # Or 'cpu', 'mps'
    'dtype': 'float64',
    'algorithm': 'qr',
    'input_hash': 'sha256:abc123...',  # Optional hash of input metadata (NOT data)
    'timestamp': '2025-12-26T15:30:00Z',
}
```

This matters immensely for FDA-style reproducibility requirements. Even if initially minimal, having the field established prevents future breaking changes.

**Domain-Specific Payloads:**
```python
# Regression
@dataclass
class LinearParams:
    coefficients: np.ndarray
    residuals: np.ndarray | None      # None if not repeatable
    fitted_values: np.ndarray | None  # None if not repeatable
    rss: float
    tss: float
    rank: int
    df_residual: int

# MVN MLE
@dataclass
class MVNParams:
    muhat: np.ndarray      # Mean estimate
    sigmahat: np.ndarray   # Covariance estimate
    loglik: float          # Log-likelihood at optimum
```

### The Backend Protocol

**Purpose:** Hardware-specific computation abstraction.

**Philosophy:** Backends are stateless transformers. They take domain-specific designs and produce results. All configuration happens through the design or at construction time.

```python
from typing import Protocol, TypeVar

D = TypeVar('D')  # DataSource/Design type
P = TypeVar('P')  # Parameter payload type

class Backend(Protocol[D, P]):
    @property
    def name(self) -> str:
        """Identifier: '{device}_{algorithm}' (e.g., 'cpu_qr', 'gpu_cholesky')"""
        ...
    
    def solve(self, design: D) -> Result[P]:
        """Execute computation. Never re-validates inputs."""
        ...
```

**Naming Convention:**
```
{device}_{algorithm}
├── cpu_qr        # CPU, QR decomposition
├── cpu_svd       # CPU, SVD decomposition  
├── cpu_em        # CPU, EM algorithm
├── gpu_cholesky  # GPU, Cholesky decomposition
└── gpu_qr        # GPU, QR decomposition
```

---

## System Architecture

### Directory Structure

```
pystatistics/
│
├── __init__.py                     # Package entry: DataSource, version
│
├── core/                           # Domain-agnostic infrastructure
│   ├── __init__.py                 # Core exports
│   ├── datasource.py               # Universal DataSource class
│   ├── capabilities.py             # Capability string constants (single source of truth)
│   ├── result.py                   # Generic Result[P] envelope
│   ├── protocols.py                # DataSource, Backend protocols
│   ├── exceptions.py               # Exception hierarchy
│   ├── validation.py               # Input validators
│   │
│   └── compute/                    # Shared compute infrastructure (NOT "backends")
│       ├── __init__.py
│       ├── device.py               # GPU detection, device selection
│       ├── timing.py               # Execution timing utilities
│       ├── precision.py            # Numerical constants (eps, etc.)
│       │
│       └── linalg/                 # Linear algebra kernels
│           ├── qr.py               # QR decomposition (CPU + GPU)
│           ├── cholesky.py         # Cholesky decomposition
│           ├── svd.py              # SVD decomposition
│           └── solve.py            # Triangular solvers
│
├── regression/                     # Linear and generalized linear models
│   ├── __init__.py                 # Public: Design, fit
│   ├── design.py                   # Design (wraps DataSource for regression)
│   ├── solution.py                 # LinearSolution, LinearParams
│   ├── solvers.py                  # fit() dispatcher
│   │
│   └── backends/                   # Domain-specific Backend implementations
│       ├── __init__.py
│       ├── cpu.py                  # CPU QR reference implementation
│       └── gpu.py                  # GPU PyTorch implementation
│
├── mvnmle/                         # Multivariate normal MLE
│   ├── __init__.py                 # Public: mlest, datasets
│   ├── data.py                     # MVNData (wraps DataSource for MVN)
│   ├── solution.py                 # MVNSolution, MVNParams
│   ├── algorithms/                 # EM, direct optimization
│   │   ├── em.py
│   │   └── direct.py
│   ├── datasets.py                 # Reference datasets (apple, missvals)
│   │
│   └── backends/                   # Domain-specific Backend implementations
│       ├── __init__.py
│       ├── cpu.py                  # NumPy/SciPy reference
│       └── gpu.py                  # PyTorch GPU implementation
│
├── survival/                       # Survival analysis (planned)
│   ├── __init__.py
│   ├── data.py                     # SurvivalData
│   └── backends/                   # Domain-specific backends
│
└── mixed/                          # Mixed models (planned)
    ├── __init__.py
    ├── data.py                     # MixedData
    └── backends/                   # Domain-specific backends
```

**Critical Naming Convention:**
- **`core/compute/`** - Shared numeric kernels (linalg, device detection, timing). These are NOT backends.
- **`{domain}/backends/`** - Domain-specific implementations of the `Backend` protocol.

This distinction prevents category confusion: "backend" always means "thing that implements Backend protocol for a domain."

### Layer Responsibilities

#### Layer 0: Core Infrastructure (`core/`)

**Responsibility:** Provide domain-agnostic utilities that all domains share.

| Component | Purpose |
|-----------|---------|
| `datasource.py` | Universal data container |
| `capabilities.py` | Capability string constants (single source of truth) |
| `result.py` | Generic result envelope |
| `protocols.py` | Type protocols for static checking |
| `exceptions.py` | Exception hierarchy |
| `validation.py` | Input validators |
| `compute/` | Device detection, timing, linalg kernels |

**Key Constraint:** Core NEVER imports from domain packages. Dependencies flow outward only.

#### Layer 1: Domain Public API (`{domain}/__init__.py`)

**Responsibility:** Define the user-facing API for each domain.

| Task | Owner |
|------|-------|
| Validate inputs ONCE | Public API |
| Dispatch to appropriate solver | Public API |
| Handle convenience shortcuts | Public API |

**Example (Regression):**
```python
# pystatistics/regression/__init__.py
from pystatistics.regression.design import Design
from pystatistics.regression.solvers import fit

__all__ = ["Design", "fit"]
```

#### Layer 2: Domain Wrapper (`{domain}/design.py` or `{domain}/data.py`)

**Responsibility:** Wrap DataSource with domain-specific extraction logic.

| Domain | Wrapper Class | File | Wraps DataSource To Extract |
|--------|---------------|------|----------------------------|
| Regression | `Design` | `design.py` | X matrix, y vector, has_intercept |
| MVN MLE | `MVNData` | `data.py` | Data matrix, missingness patterns |
| Survival | `SurvivalData` | `data.py` | Time, event, covariates |

**Key Constraints:** 
- Wrappers are IMMUTABLE after construction
- Wrappers ALWAYS hold a reference to their source DataSource
- Wrappers know what their domain needs; DataSource doesn't

#### Layer 3: Solvers (`{domain}/solvers.py`)

**Responsibility:** Coordinate between Design and Backend.

| Task | Owner |
|------|-------|
| Query Design capabilities | Solver |
| Choose access pattern | Solver |
| Select backend | Solver |
| Wrap Backend output in Solution | Solver |

#### Layer 4: Backends (`{domain}/backends/`)

**Responsibility:** Execute numerical computation.

| Task | Owner |
|------|-------|
| Implement algorithms | Backend |
| Return standardized Result | Backend |
| Handle hardware specifics | Backend |
| NEVER re-validate inputs | Backend |

---

## Domain Extensions

### How Domains Extend the Core

Each domain follows the same pattern:

```
1. DataSource              →  The lumber yard (domain-agnostic)
2. Domain Wrapper          →  The buyer (knows what the domain needs)
   (Design, MVNData, etc.)    Wraps DataSource, extracts domain-specific views
3. Domain Solver           →  Dispatches to backends
4. Domain Backend(s)       →  Hardware-specific computation
5. Domain Solution         →  User-facing result wrapper
```

### Regression Extension

**Purpose:** Linear and generalized linear models.

**Wrapper:** `Design` - Named after the statistical "design matrix" concept. Wraps DataSource and extracts X (design matrix) and y (response vector).

```python
from pystatistics import DataSource
from pystatistics.regression import Design, fit

# DataSource: The lumber yard
ds = DataSource.from_file("data.csv")

# Design: The buyer who knows regression needs X and y
design = Design.from_datasource(ds, y='target')

# fit: The factory that builds the result
result = fit(design)
print(result.coefficients)
print(result.standard_errors)
print(result.r_squared)
```

**Residuals Policy (Two-Pass Contract):**

Residuals and fitted values require a second pass through the data. The contract is explicit:

| Data Source Type | `supports('repeatable')` | Residuals Available |
|------------------|--------------------------|---------------------|
| Dense (in-memory) | `True` | ✅ Yes |
| Streaming (HDF5) with re-iteration | `True` | ✅ Yes |
| Streaming (one-pass only) | `False` | ❌ No |
| Sufficient statistics only | `False` | ❌ No |

The result contract makes this explicit:

```python
@dataclass
class LinearParams:
    coefficients: np.ndarray           # Always present
    residuals: np.ndarray | None       # None if not repeatable
    fitted_values: np.ndarray | None   # None if not repeatable
    ...

# In result.info:
result.info['residuals_available']  # True or False
```

**Why this matters:** Downstream code that assumes residuals exist will fail silently with streaming data. Making this explicit in the type signature (`Optional`) and in `info` prevents that class of bug.

**Backends:**
- `cpu_qr` - Reference implementation using LAPACK QR (R-compatible)
- `cpu_svd` - SVD-based (more stable for ill-conditioned problems)
- `gpu_qr` - PyTorch GPU implementation

### MVN MLE Extension

**Purpose:** Maximum likelihood estimation for multivariate normal data with missing values.

**Wrapper:** `MVNData` - NOT called "Design" because MVN MLE doesn't have a design matrix. Wraps DataSource and extracts the data matrix with missingness pattern analysis.

**MVNData is Heavier Than Design (And That's OK):**

Domain wrappers are uniform in *role* (they extract what the domain needs from DataSource), but NOT uniform in *shape*. MVN MLE has an extra axis that regression lacks: **missingness structure**.

`MVNData` caches:
```python
@dataclass(frozen=True)
class MVNData:
    _data: np.ndarray                    # The raw data matrix
    _source: DataSource                  # Reference to lumber yard
    
    # Missingness structure (computed once, cached)
    _pattern_ids: np.ndarray             # Which pattern each row belongs to
    _pattern_counts: dict[int, int]      # How many rows per pattern
    _pattern_masks: dict[int, np.ndarray]  # Boolean mask for each pattern
    _permutation_indices: np.ndarray     # Indices to group by pattern
    
    # Optional: per-pattern sufficient statistics
    _pattern_stats: dict[int, dict] | None
```

This is heavier than `Design` (which just stores X and y), and that's correct. Don't force symmetry where the domains genuinely differ.

```python
from pystatistics.mvnmle import mlest, datasets

# Classic usage (mimics R mvnmle)
result = mlest(datasets.apple)
print(result.muhat)       # Mean estimates
print(result.sigmahat)    # Covariance estimates
print(result.loglik)      # Log-likelihood

# Explicit DataSource usage
ds = DataSource.from_arrays(data=my_data)
result = mlest(ds, backend='gpu')
```

**Backends:**
- `cpu_em` - CPU EM algorithm (R-compatible)
- `cpu_direct` - Direct optimization with L-BFGS-B
- `gpu_em` - GPU-accelerated EM algorithm

### Future Extensions

| Domain | Wrapper Name | Purpose | Key Data Elements |
|--------|--------------|---------|-------------------|
| `survival` | `SurvivalData` | Survival analysis | Time, event, censoring indicator |
| `mixed` | `MixedData` | Mixed-effects models | Fixed effects, random effects, groups |
| `timeseries` | `TimeSeriesData` | Time series | Temporal ordering, lags, frequency |
| `hypothesis` | `TestData` | Hypothesis tests | Sample(s), null hypothesis specification |

---

## Component Specifications

### Exception Hierarchy

```python
PyStatisticsError                    # Base for all library exceptions
├── ValidationError                  # Input validation failed
│   └── DimensionError              # Array dimensions wrong
├── NumericalError                   # Computation failed
│   ├── SingularMatrixError         # Matrix not invertible
│   └── NotPositiveDefiniteError    # Cholesky failed
└── ConvergenceError                 # Iterative method didn't converge
```

**Design Principles:**
- Exceptions carry diagnostic attributes (not just messages)
- Error messages are actionable with actual values
- Never catch and re-raise with less information

```python
class SingularMatrixError(NumericalError):
    def __init__(
        self, 
        message: str,
        matrix_name: str | None = None,
        condition_number: float | None = None,
        rank: int | None = None,
        expected_rank: int | None = None
    ):
        super().__init__(message)
        self.matrix_name = matrix_name
        self.condition_number = condition_number
        self.rank = rank
        self.expected_rank = expected_rank
```

### Validation Utilities

**Location:** `pystatistics/core/validation.py`

**Philosophy:** Each function validates ONE thing. No silent coercion except `np.asarray` on array-likes.

| Function | Purpose |
|----------|---------|
| `check_array(array, name)` | Convert to numpy, reject object dtype |
| `check_finite(array, name)` | Reject NaN/Inf |
| `check_ndim(array, ndim, name)` | Verify dimensions |
| `check_1d(array, name)` | Verify 1-dimensional |
| `check_2d(array, name)` | Verify 2-dimensional |
| `check_consistent_length(*arrays)` | Verify same n_observations |
| `check_min_samples(n, min_n, name)` | Verify minimum sample size |

### Device Detection

**Location:** `pystatistics/core/compute/device.py`

```python
@dataclass
class DeviceInfo:
    device_type: Literal['cpu', 'cuda', 'mps']
    device_name: str
    memory_gb: float | None
    compute_capability: str | None  # CUDA compute capability

def detect_gpu() -> DeviceInfo | None:
    """Detect available GPU. Returns None if CPU-only."""
    
def select_device(preference: str) -> DeviceInfo:
    """
    Select device based on preference.
    
    Args:
        preference: 'auto', 'cpu', 'cuda', 'mps'
        
    Returns:
        DeviceInfo for selected device
    """
```

### Timing Utilities

**Location:** `pystatistics/core/compute/timing.py`

```python
class Timer:
    """Hierarchical timing context manager."""
    
    def start(self) -> None: ...
    def stop(self) -> None: ...
    
    @contextmanager
    def section(self, name: str) -> Generator[None, None, None]:
        """Time a named section."""
        
    def result(self) -> dict[str, float]:
        """Return timing breakdown."""
```

**Usage:**
```python
timer = Timer()
timer.start()

with timer.section('qr_decomposition'):
    Q, R = qr(X)
    
with timer.section('solve'):
    beta = solve_triangular(R, Q.T @ y)

timer.stop()
print(timer.result())
# {'total_seconds': 0.015, 'qr_decomposition': 0.010, 'solve': 0.005}
```

---

## Data Flow

### Regression Data Flow

```
User Input                    Layer 0: Core              Layer 1: Domain
───────────                   ──────────────             ───────────────

X, y arrays ──────────────────────────────────────────> Design.from_arrays()
     │                                                        │
     └─> DataSource.from_arrays() ──> DataSource             │
              (the lumber yard)           │                   │
                                          │         Design (the buyer)
                                          │         wraps DataSource,
                                          │         extracts X and y
                                          │                   │
                                          │                   v
                                          │            fit(design, ...)
                                          │                   │
                                          │                   v
                                          │            _get_backend()
                                          │                   │
                                          │                   v
                                          │            backend.solve()
                                          │                   │
                                          v                   v
                                      Shared linalg      Result[LinearParams]
                                      (qr_cpu, etc.)           │
                                                               v
                                                        LinearSolution
                                                               │
                                                               v
                                                        User accesses:
                                                        .coefficients
                                                        .standard_errors
                                                        .r_squared
                                                        .summary()
```

### MVN MLE Data Flow

```
User Input                    Layer 0: Core              Layer 1: Domain
───────────                   ──────────────             ───────────────

data array ───────────────────────────────────────────> mlest(data, ...)
     │                                                        │
     └─> DataSource.from_arrays() ──> DataSource             │
              (the lumber yard)           │                   │
                                          │         MVNData (the buyer)
                                          │         wraps DataSource,
                                          │         analyzes missingness
                                          │                   │
                                          │                   v
                                          │            _select_backend()
                                          │                   │
                                          │                   v
                                          │            backend.solve()
                                          │            (EM or direct)
                                          │                   │
                                          v                   v
                                      Shared optimization Result[MVNParams]
                                      (convergence, etc.)      │
                                                               v
                                                        MVNSolution
                                                               │
                                                               v
                                                        User accesses:
                                                        .muhat
                                                        .sigmahat
                                                        .loglik
                                                        .converged
```

---

## API Reference

### Top-Level Package

```python
import pystatistics

pystatistics.__version__     # "0.1.0"
pystatistics.DataSource      # Universal data container
pystatistics.regression      # Regression submodule
pystatistics.mvnmle          # MVN MLE submodule
```

### DataSource API

```python
from pystatistics import DataSource
from pystatistics.core.capabilities import (
    CAPABILITY_MATERIALIZED, 
    CAPABILITY_STREAMING,
    CAPABILITY_REPEATABLE,
)

# Factory methods
ds = DataSource.from_arrays(X=X, y=y)
ds = DataSource.from_arrays(data=data, columns=['a', 'b', 'c'])
ds = DataSource.from_file("data.csv")
ds = DataSource.from_file("data.npy")
ds = DataSource.from_dataframe(df)
ds = DataSource.from_tensors(X=X_gpu, y=y_gpu)

# Properties
ds.n_observations  # int
ds.metadata        # dict
ds.keys()          # frozenset[str] - available array names

# Capability checking (use constants, not raw strings)
ds.supports(CAPABILITY_MATERIALIZED)   # True
ds.supports(CAPABILITY_STREAMING)      # False
ds.supports(CAPABILITY_REPEATABLE)     # True

# Constrained dict-like access
ds['X']  # Access named array (raises KeyError with helpful message if missing)
ds['y']  # Access named array
ds['Z']  # KeyError: "DataSource has no array 'Z'. Available: {'X', 'y'}"
```

### Regression API

```python
from pystatistics.regression import Design, fit

# Design construction (the "buyer" for regression)
# Design knows regression needs X (design matrix) and y (response)
design = Design.from_datasource(ds, y='target')
design = Design.from_datasource(ds, x=['a', 'b'], y='c')
design = Design.from_arrays(X, y)

# Properties
design.X           # np.ndarray (design matrix)
design.y           # np.ndarray (response vector)
design.n           # int (observations)
design.p           # int (parameters)
design.source      # DataSource (the lumber yard it came from)

# Fitting
result = fit(design)
result = fit(design, backend='cpu_qr')
result = fit(design, backend='gpu')

# Convenience (skips explicit DataSource)
result = fit(X, y)

# Result access - always available
result.coefficients      # np.ndarray
result.standard_errors   # np.ndarray
result.t_statistics      # np.ndarray
result.p_values          # np.ndarray
result.r_squared         # float
result.adjusted_r_squared # float
result.residual_std_error # float
result.df_residual       # int
result.timing            # dict | None
result.provenance        # dict (versions, device, algorithm)
result.summary()         # Print R-style table

# Result access - may be None for non-repeatable data
result.residuals         # np.ndarray | None
result.fitted_values     # np.ndarray | None
result.info['residuals_available']  # True or False
```

### MVN MLE API

```python
from pystatistics.mvnmle import mlest, MVNData, datasets

# Basic usage (MVNData created internally)
result = mlest(data)
result = mlest(data, backend='gpu')
result = mlest(data, max_iter=500, tol=1e-8)

# Explicit MVNData construction (the "buyer" for MVN MLE)
# MVNData knows MVN MLE needs the data matrix and missingness patterns
mvn_data = MVNData.from_datasource(ds)
mvn_data = MVNData.from_arrays(data)

# MVNData properties (heavier than Design due to missingness structure)
mvn_data.data              # np.ndarray (raw data with NaNs)
mvn_data.n                 # int (observations)
mvn_data.p                 # int (variables)
mvn_data.n_patterns        # int (number of unique missingness patterns)
mvn_data.pattern_ids       # np.ndarray (which pattern each row belongs to)
mvn_data.pattern_counts    # dict[int, int] (rows per pattern)
mvn_data.source            # DataSource (the lumber yard it came from)

result = mlest(mvn_data)

# Reference datasets
result = mlest(datasets.apple)
result = mlest(datasets.missvals)

# Result access
result.muhat       # np.ndarray (mean estimates)
result.sigmahat    # np.ndarray (covariance estimates)
result.loglik      # float (log-likelihood)
result.converged   # bool
result.n_iter      # int
result.backend     # str
result.timing      # dict | None
result.provenance  # dict (versions, device, algorithm)
```

---

## Implementation Guidelines

### Stability Rule

> **Any change that requires renaming a top-level concept (DataSource, Design/MVNData, Backend, Result, compute, kernel) is prohibited until after v1.0 ships.**

This rule exists to prevent refactor churn. The architecture is now frozen. Build on it, don't redesign it.

### Capability String Policy

All capability strings are defined in `core/capabilities.py`. When adding new capabilities:

1. **Use consistent naming** - All capabilities are adjectives/past participles describing state (`materialized`, `streaming`, `repeatable`, `gpu_native`)
2. **Never use shorthand** - Use `sufficient_statistics`, not `sufficient_stats` or `suffstats`
3. **Add to capabilities.py ONLY** - Never define capability strings inline elsewhere
4. **Never remove or rename** - Only add new capabilities; old ones are permanent

### Kernel vs Backend: The Bright Line

| Term | Definition | Location | Example |
|------|------------|----------|---------|
| **Kernel** | Numeric primitive, stateless, domain-agnostic | `core/compute/linalg/` | `qr_cpu()`, `cholesky_gpu()` |
| **Backend** | Domain-specific implementation of Backend protocol | `{domain}/backends/` | `CPUQRBackend`, `GPUEMBackend` |

**The rule:** If it implements the `Backend` protocol and returns a `Result`, it's a backend. If it's a pure numeric operation that backends *call*, it's a kernel.

A backend like `regression/backends/cpu.py` may internally call kernels like `compute/linalg/qr.py`, but the kernel doesn't know about regression, Design objects, or Result envelopes.

### Adding a New Domain

To add a new statistical domain (e.g., `survival`):

1. **Create directory structure:**
```
pystatistics/survival/
├── __init__.py         # Public API
├── data.py             # SurvivalData (wrapper around DataSource)
├── solution.py         # SurvivalSolution, SurvivalParams
├── solvers.py          # fit() or equivalent
└── backends/
    ├── __init__.py
    ├── cpu.py          # Reference implementation
    └── gpu.py          # GPU implementation
```

2. **Define domain-specific wrapper (the "buyer"):**
```python
@dataclass(frozen=True)
class SurvivalData:
    """
    Wrapper around DataSource for survival analysis.
    
    This is the "buyer" that knows survival analysis needs:
    - Time-to-event or censoring
    - Event indicator (0/1)
    - Optional covariates
    
    It visits the DataSource "lumber yard" and extracts
    exactly what survival backends need.
    """
    _time: NDArray[np.floating]
    _event: NDArray[np.integer]
    _X: NDArray[np.floating] | None
    _source: DataSource  # Always keep reference to source
    
    @classmethod
    def from_datasource(cls, source: DataSource, *, 
                        time: str, event: str, 
                        covariates: list[str] | None = None) -> SurvivalData:
        """Extract survival data from DataSource."""
        ...
```

3. **Define domain-specific params:**
```python
@dataclass
class SurvivalParams:
    hazard_ratios: np.ndarray
    baseline_hazard: np.ndarray
    log_likelihood: float
    concordance: float
```

4. **Implement at least one backend:**
```python
class CPUCoxBackend:
    @property
    def name(self) -> str:
        return 'cpu_cox'
    
    def solve(self, data: SurvivalData) -> Result[SurvivalParams]:
        ...
```

5. **Register in umbrella `__init__.py`:**
```python
# pystatistics/__init__.py
from pystatistics import survival  # Add import
```

**Naming Convention:**
- Use `Design` ONLY for regression (statistical term "design matrix")
- Use `{Domain}Data` for all other domains (MVNData, SurvivalData, etc.)

### Code Style Requirements

1. **No default parameters except at top-level API**
```python
# GOOD: Top-level API has defaults
def fit(design, *, backend='auto', mode='stable'): ...

# GOOD: Internal function requires all params  
def _solve_qr(X, y, check_rank): ...  # No defaults

# BAD: Internal function with defaults
def _solve_qr(X, y, check_rank=True): ...  # Violates principle
```

2. **Each function does ONE thing**
```python
# GOOD: Separate concerns
def validate_design_matrix(X, name): ...
def compute_qr(X): ...
def solve_from_qr(Q, R, y): ...

# BAD: Multi-purpose function
def validate_and_solve(X, y, validate=True, method='auto'): ...
```

3. **Fail fast, fail loud**
```python
# GOOD: Explicit error
if X.shape[0] < X.shape[1]:
    raise DimensionError(f"X has more columns ({X.shape[1]}) than rows ({X.shape[0]})")

# BAD: Silent handling
if X.shape[0] < X.shape[1]:
    X = X.T  # "Fix" the problem silently
```

### Backend Implementation Checklist

- [ ] Implements `name` property with `{device}_{algorithm}` format
- [ ] Implements `solve(design) -> Result[P]`
- [ ] Returns complete `Result` envelope with timing
- [ ] NEVER validates inputs (trusts solver)
- [ ] Populates `info` dict with method/algorithm details
- [ ] Raises `NumericalError` or subclass for computation failures
- [ ] Has corresponding unit tests
- [ ] Passes R validation fixtures (for CPU reference backends)

---

## Testing Strategy

### Test Structure

```
tests/
├── unit/
│   ├── core/
│   │   ├── test_datasource.py
│   │   ├── test_result.py
│   │   ├── test_validation.py
│   │   └── test_exceptions.py
│   ├── regression/
│   │   ├── test_design.py
│   │   ├── test_solution.py
│   │   └── test_backends.py
│   └── mvnmle/
│       └── ...
│
├── integration/
│   ├── test_vs_r_regression.py    # Compare to R lm()
│   ├── test_vs_r_mvnmle.py        # Compare to R mvnmle
│   ├── test_gpu_cpu_equivalence.py
│   └── test_streaming.py
│
├── benchmarks/
│   ├── bench_regression.py
│   ├── bench_mvnmle.py
│   └── bench_gpu_speedup.py
│
└── fixtures/
    ├── r_generated/               # R reference outputs
    │   ├── regression/
    │   └── mvnmle/
    └── synthetic/                 # Synthetic test cases
```

### Validation Requirements

**Every CPU reference backend must pass:**

1. **R equivalence tests** - Match R output to 1e-12 for direct methods, 1e-7 for iterative
2. **Edge case tests** - Ill-conditioned matrices, near-singular, high/low noise
3. **Error handling tests** - Proper exceptions for invalid inputs

**Every GPU backend must pass:**

1. **CPU equivalence tests** - Match CPU backend output
2. **Memory management tests** - Handle large datasets gracefully
3. **Performance benchmarks** - Documented speedup factors

### R Fixture Generation

```r
# tests/fixtures/generate_regression_fixtures.R
set.seed(42)

generate_fixture <- function(name, n, p, ...) {
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)
  fit <- lm(y ~ X - 1)  # No intercept for clean comparison
  
  list(
    X = X,
    y = y,
    coefficients = coef(fit),
    residuals = residuals(fit),
    fitted_values = fitted(fit),
    sigma = summary(fit)$sigma,
    std_errors = summary(fit)$coefficients[, "Std. Error"],
    r_squared = summary(fit)$r.squared
  )
}

# Generate fixtures
fixtures <- list(
  basic_100x5 = generate_fixture("basic", 100, 5),
  tall_1000x5 = generate_fixture("tall", 1000, 5),
  near_square_50x45 = generate_fixture("near_square", 50, 45)
)

jsonlite::write_json(fixtures, "regression_fixtures.json")
```

---

## Performance Considerations

### Backend Selection Logic

```python
def select_optimal_backend(design, preference='auto'):
    """
    Select backend based on problem characteristics.
    
    Decision factors:
    1. User preference (explicit beats auto)
    2. Hardware availability
    3. Problem size (GPU overhead vs benefit)
    4. Memory constraints
    """
    if preference != 'auto':
        return get_backend(preference)
    
    n, p = design.n, design.p
    
    # Small problems: CPU (avoid GPU overhead)
    if n * p < 10_000:
        return CPUQRBackend()
    
    # Check GPU availability
    gpu = detect_gpu()
    if gpu is None:
        return CPUQRBackend()
    
    # Large problems: GPU preferred
    if n * p > 100_000:
        return GPUQRBackend()
    
    # Medium problems: CPU usually wins due to transfer overhead
    return CPUQRBackend()
```

### Memory Management

**CPU:**
- Pre-allocate output arrays
- Use in-place operations where safe
- Release intermediate matrices early

**GPU:**
- Minimize CPU↔GPU transfers
- Accept GPU tensors directly when possible
- Chunked processing for datasets larger than GPU memory
- Graceful fallback to CPU if OOM

### Streaming for Large Datasets

For datasets too large to fit in memory:

```python
# DataSource with streaming capability
ds = DataSource.from_hdf5("large_data.h5")
assert ds.supports('stream')

# Design uses sufficient statistics internally
design = Design.from_datasource(ds, y='target')

# Solver detects streaming and accumulates X'X, X'y
result = fit(design)  # Still exact OLS, just computed differently
```

**Key insight:** Streaming computation uses sufficient statistics accumulation. The mathematical result is IDENTICAL to dense computation—only the data access pattern changes.

---

## Future Extensions

### Planned Domains

| Domain | Target Version | Key Features |
|--------|----------------|--------------|
| `survival` | v0.3 | Cox PH, Kaplan-Meier, log-rank test |
| `mixed` | v0.4 | LMM, GLMM, REML estimation |
| `timeseries` | v0.5 | ARIMA, state space, Kalman filter |
| `hypothesis` | v0.6 | t-tests, ANOVA, chi-square |

### Planned Features

1. **Formula interface** (v0.4+)
   - Optional layer on top of explicit matrices
   - Patsy-based parsing
   - Only after core is battle-tested

2. **Distributed computing** (v0.5+)
   - Sufficient statistics accumulation across nodes
   - Dask/Ray integration

3. **Serialization/reproducibility** (v0.3+)
   - Save/load Result objects
   - Seed management for randomized algorithms

4. **Robust standard errors** (v0.3)
   - HC0, HC1, HC2, HC3, HC4 heteroskedasticity-robust
   - Clustered standard errors

---

## Appendix: Design Decisions

### Historical Context: Why Traditional Stats Packages Entangled Layers

Traditional statistical software (S, R, SAS, Stata) entangled data, model specification, and fitting because:

1. **Interactive environments** - S and R grew up as interactive REPLs where "model spec + data + fitting" was one gesture (`lm(y ~ x, data=df)`)
2. **RAM assumptions** - Most datasets fit in memory, so I/O wasn't a first-class concern
3. **Audience** - Statisticians first, programmers second. DSLs (formula interfaces) were a feature, not a smell.

**The PyStatistics Inversion:**

PyStatistics architecture inverts this:
- **I/O is first-class** - DataSource handles all data loading concerns
- **Compute is first-class** - Backends are explicit, swappable, hardware-aware
- **Model spec is just metadata** - Domain wrappers carry configuration, not behavior

This is the modern pattern for large-scale, hardware-accelerated scientific computing.

### Why DataSource + Domain Wrapper Instead of One Layer?

**Decision:** Two-layer abstraction with domain-agnostic DataSource and domain-specific wrappers (Design, MVNData, etc.).

**Rationale (The Lumber Yard Analogy):**
1. **DataSource is the lumber yard** - It has data (logs). It doesn't know or care what you're building.
2. **Domain wrappers are the buyers** - Each domain sends someone who knows exactly what's needed:
   - Regression sends someone who knows they need logs for a "design matrix" (hence `Design`)
   - MVN MLE sends someone who knows they need logs with "missingness patterns" (hence `MVNData`)
   - Survival sends someone who knows they need "time-to-event" and "censoring" (hence `SurvivalData`)
3. **Same DataSource can serve multiple domains** - Load once, use for regression AND MVN analysis
4. **Clear separation of concerns** - DataSource handles I/O; wrappers handle domain semantics

**Why this matters:**
- Domains don't reinvent CSV parsing, DataFrame conversion, GPU tensor handling
- Users learn DataSource once, apply it everywhere
- Each domain gets the exact interface it needs without contortions

### Why Protocol Instead of ABC?

**Decision:** Use `Protocol` (structural typing) for Backend interface.

**Rationale:**
1. Flexibility—backends don't need explicit inheritance
2. Easier testing—mock objects just need to have the right methods
3. Forward-compatible with potential non-Python backends

**Trade-off:** Less explicit about interface requirements. Mitigated by comprehensive type hints and documentation.

### Why Generic Result Instead of Dict?

**Decision:** `Result[P]` with typed parameter payload.

**Rationale:**
1. Type safety through generics
2. IDE autocomplete works
3. Immutability prevents accidental modification
4. Common envelope enables shared tooling

### Why Separate CPU and GPU Backends?

**Decision:** Distinct backend classes rather than unified backend with device parameter.

**Rationale:**
1. Algorithms may differ (GPU-friendly algorithms aren't always CPU-optimal)
2. Clearer code organization
3. Easier to test in isolation
4. No runtime device-switching complexity

---

## Glossary

| Term | Definition |
|------|------------|
| **DataSource** | Domain-agnostic data container (the "lumber yard") |
| **Domain Wrapper** | Domain-specific data extractor that wraps DataSource (the "buyer") |
| **Design** | Regression-specific wrapper; named after "design matrix" |
| **MVNData** | MVN MLE-specific wrapper; caches missingness patterns |
| **SurvivalData** | Survival analysis-specific wrapper |
| **Backend** | Domain-specific implementation of the Backend protocol; lives in `{domain}/backends/` |
| **Compute** | Shared numeric infrastructure in `core/compute/` (NOT backends) |
| **Kernel** | A stateless numeric primitive (QR, Cholesky, SVD) in `core/compute/linalg/`; called BY backends, never domain-aware |
| **Result** | Generic envelope containing computation results |
| **Params** | Domain-specific parameter payload (coefficients, estimates) |
| **Solver** | Dispatcher coordinating Wrapper → Backend → Result |
| **Capability** | Feature a DataSource may or may not support (defined in `core/capabilities.py`) |
| **Provenance** | Metadata for reproducibility (versions, device, algorithm) |
| **Sufficient Statistics** | Aggregated data (X'X, X'y) sufficient for estimation |
| **Repeatable** | Capability indicating data can be iterated multiple times |
| **Immutability** | Guarantee that objects don't change after construction |

---

## References

1. R Core Team. (2024). R: A Language and Environment for Statistical Computing
2. Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.)
3. Little, R. J. A., & Rubin, D. B. (2019). Statistical Analysis with Missing Data (3rd ed.)
4. McCullagh, P., & Nelder, J. A. (1989). Generalized Linear Models (2nd ed.)
5. scikit-learn API Design Principles. scikit-learn.org

---

**Document Status:** Final Design Specification  
**Next Steps:** Complete regression CPU backend, retrofit mvnmle integration  
**Review Cycle:** Quarterly architecture review