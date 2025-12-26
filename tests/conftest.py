"""
pytest configuration and shared fixtures.
"""

import pytest
import numpy as np


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_regression_data(rng):
    """Simple regression dataset for basic tests."""
    n, p = 100, 3
    X = rng.standard_normal((n, p))
    beta_true = np.array([1.0, -2.0, 0.5])
    y = X @ beta_true + rng.standard_normal(n) * 0.1
    return X, y, beta_true


@pytest.fixture
def collinear_data(rng):
    """Dataset with perfect collinearity (should fail)."""
    n = 100
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = x1 + x2  # Perfect collinearity
    X = np.column_stack([x1, x2, x3])
    y = rng.standard_normal(n)
    return X, y
