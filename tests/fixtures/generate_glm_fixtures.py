#!/usr/bin/env python3
"""
Generate test fixtures for GLM validation.

Creates CSV files with known properties for testing against R's glm().

Test cases:
1. glm_gaussian_basic     - Gaussian identity, n=100, p=3 (must match LM exactly)
2. glm_binomial_basic     - Binomial logit, n=200, p=3 (standard logistic)
3. glm_binomial_balanced  - Binomial logit, n=500, p=5 (balanced classes)
4. glm_binomial_separated - Binomial logit, n=100, p=2 (near-separation)
5. glm_poisson_basic      - Poisson log, n=200, p=3 (standard count data)
6. glm_poisson_zeros      - Poisson log, n=200, p=3 (many zeros)
7. glm_poisson_large_counts - Poisson log, n=300, p=3 (large lambda values)
8. glm_gaussian_large     - Gaussian identity, n=5000, p=10 (scale test)
9. glm_binomial_large     - Binomial logit, n=5000, p=10 (scale test)

Run from /path/to/pystatistics:
    python tests/fixtures/generate_glm_fixtures.py
"""

import numpy as np
import json
from pathlib import Path

# Reproducibility
RNG = np.random.default_rng(20250214)

FIXTURES_DIR = Path(__file__).parent


def save_fixture(name: str, X: np.ndarray, y: np.ndarray, metadata: dict):
    """Save fixture as CSV with metadata JSON."""
    data = np.column_stack([X, y])

    p = X.shape[1]
    col_names = [f'x{i}' for i in range(p)] + ['y']

    # Save CSV with high precision
    csv_path = FIXTURES_DIR / f"{name}.csv"
    header = ','.join(col_names)
    np.savetxt(csv_path, data, delimiter=',', header=header, comments='', fmt='%.18e')

    # Save metadata
    meta_path = FIXTURES_DIR / f"{name}_meta.json"
    meta = {
        'name': name,
        'n': int(X.shape[0]),
        'p': int(X.shape[1]),
        **metadata
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  {name}: n={meta['n']}, p={meta['p']}, family={meta.get('family', '?')}")


# =====================================================================
# Gaussian fixtures
# =====================================================================

def generate_glm_gaussian_basic():
    """Gaussian with identity link — must match LM exactly."""
    n, p = 100, 3
    X = np.column_stack([
        np.ones(n),
        RNG.standard_normal(n),
        RNG.standard_normal(n)
    ])
    beta_true = np.array([1.0, 2.0, -0.5])
    sigma = 0.5
    y = X @ beta_true + RNG.standard_normal(n) * sigma

    save_fixture('glm_gaussian_basic', X, y, {
        'family': 'gaussian',
        'link': 'identity',
        'beta_true': beta_true.tolist(),
        'sigma': sigma,
        'description': 'Gaussian GLM with identity link (must match OLS exactly)'
    })


def generate_glm_gaussian_large():
    """Gaussian large-scale test."""
    n, p = 5000, 10
    X = np.column_stack([
        np.ones(n),
        RNG.standard_normal((n, p - 1))
    ])
    beta_true = RNG.standard_normal(p) * 2.0
    sigma = 1.0
    y = X @ beta_true + RNG.standard_normal(n) * sigma

    save_fixture('glm_gaussian_large', X, y, {
        'family': 'gaussian',
        'link': 'identity',
        'beta_true': beta_true.tolist(),
        'sigma': sigma,
        'description': 'Large Gaussian GLM for scale testing'
    })


# =====================================================================
# Binomial fixtures
# =====================================================================

def generate_glm_binomial_basic():
    """Standard logistic regression."""
    n, p = 200, 3
    X = np.column_stack([
        np.ones(n),
        RNG.standard_normal(n),
        RNG.standard_normal(n)
    ])
    beta_true = np.array([0.5, 1.0, -0.8])
    eta = X @ beta_true
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = RNG.binomial(1, prob).astype(np.float64)

    save_fixture('glm_binomial_basic', X, y, {
        'family': 'binomial',
        'link': 'logit',
        'beta_true': beta_true.tolist(),
        'prevalence': float(y.mean()),
        'description': 'Standard logistic regression'
    })


def generate_glm_binomial_balanced():
    """Balanced classes, more predictors."""
    n, p = 500, 5
    X = np.column_stack([
        np.ones(n),
        RNG.standard_normal((n, p - 1))
    ])
    # Choose coefficients to give roughly balanced classes
    beta_true = np.array([0.0, 0.8, -0.6, 0.4, -0.3])
    eta = X @ beta_true
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = RNG.binomial(1, prob).astype(np.float64)

    save_fixture('glm_binomial_balanced', X, y, {
        'family': 'binomial',
        'link': 'logit',
        'beta_true': beta_true.tolist(),
        'prevalence': float(y.mean()),
        'description': 'Balanced logistic regression with 5 predictors'
    })


def generate_glm_binomial_separated():
    """Near-separated data — IRLS edge case.

    One predictor nearly perfectly separates the classes.
    IRLS should converge but with large coefficients.
    """
    n = 100
    X = np.column_stack([
        np.ones(n),
        RNG.standard_normal(n)
    ])
    # Strong signal: coefficient = 3.0 makes near-separation
    beta_true = np.array([0.0, 3.0])
    eta = X @ beta_true
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = RNG.binomial(1, prob).astype(np.float64)

    save_fixture('glm_binomial_separated', X, y, {
        'family': 'binomial',
        'link': 'logit',
        'beta_true': beta_true.tolist(),
        'prevalence': float(y.mean()),
        'description': 'Near-separated logistic regression (large coefficients expected)'
    })


def generate_glm_binomial_large():
    """Large-scale logistic regression."""
    n, p = 5000, 10
    X = np.column_stack([
        np.ones(n),
        RNG.standard_normal((n, p - 1))
    ])
    beta_true = np.array([0.2, 0.5, -0.3, 0.4, -0.2, 0.6, -0.4, 0.3, -0.5, 0.1])
    eta = X @ beta_true
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = RNG.binomial(1, prob).astype(np.float64)

    save_fixture('glm_binomial_large', X, y, {
        'family': 'binomial',
        'link': 'logit',
        'beta_true': beta_true.tolist(),
        'prevalence': float(y.mean()),
        'description': 'Large logistic regression for scale testing'
    })


# =====================================================================
# Poisson fixtures
# =====================================================================

def generate_glm_poisson_basic():
    """Standard Poisson regression with moderate counts."""
    n, p = 200, 3
    X = np.column_stack([
        np.ones(n),
        RNG.standard_normal(n),
        RNG.standard_normal(n)
    ])
    beta_true = np.array([1.0, 0.5, -0.3])
    eta = X @ beta_true
    mu = np.exp(eta)
    y = RNG.poisson(mu).astype(np.float64)

    save_fixture('glm_poisson_basic', X, y, {
        'family': 'poisson',
        'link': 'log',
        'beta_true': beta_true.tolist(),
        'mean_count': float(y.mean()),
        'description': 'Standard Poisson regression with moderate counts'
    })


def generate_glm_poisson_zeros():
    """Poisson with many zeros (low lambda)."""
    n, p = 200, 3
    X = np.column_stack([
        np.ones(n),
        RNG.standard_normal(n),
        RNG.standard_normal(n)
    ])
    # Small intercept → small lambda → many zeros
    beta_true = np.array([-0.5, 0.3, -0.2])
    eta = X @ beta_true
    mu = np.exp(eta)
    y = RNG.poisson(mu).astype(np.float64)

    zero_frac = float((y == 0).mean())
    save_fixture('glm_poisson_zeros', X, y, {
        'family': 'poisson',
        'link': 'log',
        'beta_true': beta_true.tolist(),
        'mean_count': float(y.mean()),
        'zero_fraction': zero_frac,
        'description': 'Poisson regression with many zeros (low lambda)'
    })


def generate_glm_poisson_large_counts():
    """Poisson with large counts (high lambda)."""
    n, p = 300, 3
    X = np.column_stack([
        np.ones(n),
        RNG.standard_normal(n) * 0.5,  # smaller spread to avoid overflow
        RNG.standard_normal(n) * 0.5
    ])
    # Large intercept → large lambda
    beta_true = np.array([3.0, 0.4, -0.3])
    eta = X @ beta_true
    mu = np.exp(eta)
    y = RNG.poisson(mu).astype(np.float64)

    save_fixture('glm_poisson_large_counts', X, y, {
        'family': 'poisson',
        'link': 'log',
        'beta_true': beta_true.tolist(),
        'mean_count': float(y.mean()),
        'max_count': float(y.max()),
        'description': 'Poisson regression with large counts'
    })


# =====================================================================
# Main
# =====================================================================

def main():
    print("Generating GLM test fixtures...\n")

    generate_glm_gaussian_basic()
    generate_glm_gaussian_large()
    generate_glm_binomial_basic()
    generate_glm_binomial_balanced()
    generate_glm_binomial_separated()
    generate_glm_binomial_large()
    generate_glm_poisson_basic()
    generate_glm_poisson_zeros()
    generate_glm_poisson_large_counts()

    print(f"\nGenerated 9 GLM fixtures in {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
