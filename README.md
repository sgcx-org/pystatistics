# PyStatistics

GPU-accelerated statistical computing for Python.

## Overview

PyStatistics is a unified ecosystem for regulatory-grade statistical analysis with optional GPU acceleration. It provides:

- **Correctness**: Numerical fidelity to R reference implementations (machine precision)
- **Performance**: GPU acceleration for 10-100x speedups on large datasets
- **Regulatory Grade**: Suitable for FDA submissions and clinical research

## Installation

```bash
pip install pystatistics

# With GPU support
pip install pystatistics[gpu]

# Development
pip install pystatistics[dev]
```

## Quick Start

```python
from pystatistics.regression import fit
import numpy as np

# Generate data
X = np.random.randn(1000, 5)
y = X @ [1, 2, 3, -1, 0.5] + np.random.randn(1000) * 0.1

# Fit model
result = fit(X, y)

print(result.coefficients)
print(result.r_squared)
print(result.summary())
```

## Modules

| Module | Status | Description |
|--------|--------|-------------|
| `regression` | 🔶 In Progress | Linear and generalized linear models |
| `mvnmle` | ⬜ Planned | Multivariate normal MLE with missing data |
| `survival` | ⬜ Planned | Survival analysis (Cox, Kaplan-Meier) |
| `longitudinal` | ⬜ Planned | Mixed effects models |

## Philosophy

1. **Correctness > Fidelity > Performance > Convenience**
2. **Fail fast, fail loud**: No silent fallbacks or "helpful" defaults
3. **Explicit over implicit**: Require parameters, don't assume intent
4. **Dual-path architecture**: CPU reference + GPU acceleration

## License

MIT

## Author

Hai-Shuo (contact@sgcx.org)
