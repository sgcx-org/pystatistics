#!/usr/bin/env python3
"""
Generate test fixtures for descriptive statistics R validation.

Creates CSV files for testing against R's descriptive statistics functions.

Test cases:
1. desc_basic_100x5        - Normal data, no missing, well-conditioned
2. desc_large_1000x10      - Larger scale test
3. desc_nan_scattered       - Random 10% NaN scattered throughout
4. desc_nan_columnwise      - Structured missingness (heavier in some cols)
5. desc_perfect_correlation - Two perfectly correlated columns
6. desc_ties                - Many tied values (Spearman/Kendall stress)
7. desc_single_column       - 1D edge case
8. desc_constant_column     - One column has var=0
9. desc_extreme_values      - Very large/small magnitudes
10. desc_negative_correlation - Strong negative correlations

Run from /path/to/pystatistics:
    python tests/fixtures/generate_descriptive_fixtures.py
"""

import numpy as np
import json
from pathlib import Path

# Reproducibility
RNG = np.random.default_rng(20250301)

FIXTURES_DIR = Path(__file__).parent


def save_fixture(name: str, data: np.ndarray, metadata: dict):
    """Save fixture as CSV with metadata JSON."""
    p = data.shape[1]
    col_names = [f'x{i}' for i in range(p)]

    # Save CSV with high precision
    csv_path = FIXTURES_DIR / f"{name}.csv"
    header = ','.join(col_names)
    np.savetxt(csv_path, data, delimiter=',', header=header, comments='', fmt='%.18e')

    # Save metadata
    meta_path = FIXTURES_DIR / f"{name}_meta.json"
    meta = {
        'name': name,
        'n': int(data.shape[0]),
        'p': int(data.shape[1]),
        **metadata
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    n_nan = int(np.sum(np.isnan(data)))
    print(f"  {name}: n={meta['n']}, p={meta['p']}, NaN={n_nan}")


# =====================================================================
# Fixture 1: Basic well-conditioned normal data
# =====================================================================

def make_basic():
    n, p = 100, 5
    # Multivariate normal with moderate correlation
    mean = np.array([0.0, 5.0, -2.0, 10.0, 1.0])
    L = np.array([
        [1.0, 0, 0, 0, 0],
        [0.3, 0.95, 0, 0, 0],
        [-0.2, 0.1, 0.97, 0, 0],
        [0.5, -0.1, 0.2, 0.84, 0],
        [0.1, 0.4, -0.3, 0.1, 0.85],
    ])
    cov = L @ L.T
    data = RNG.multivariate_normal(mean, cov, size=n)
    save_fixture('desc_basic_100x5', data, {
        'description': 'Basic 100x5 normal data, no NaN, moderate correlation',
    })


# =====================================================================
# Fixture 2: Larger scale
# =====================================================================

def make_large():
    n, p = 1000, 10
    data = RNG.standard_normal((n, p))
    # Add some correlation structure
    data[:, 1] += 0.5 * data[:, 0]
    data[:, 3] -= 0.7 * data[:, 2]
    data[:, 5] += 0.3 * data[:, 4] + 0.2 * data[:, 0]
    save_fixture('desc_large_1000x10', data, {
        'description': 'Large 1000x10 with mild correlation structure',
    })


# =====================================================================
# Fixture 3: Scattered NaN (10%)
# =====================================================================

def make_nan_scattered():
    n, p = 50, 4
    data = RNG.standard_normal((n, p))
    # Scatter NaN randomly (roughly 10%)
    nan_mask = RNG.random((n, p)) < 0.10
    data[nan_mask] = np.nan
    save_fixture('desc_nan_scattered', data, {
        'description': 'Scattered NaN (~10%), 50x4',
    })


# =====================================================================
# Fixture 4: Structured NaN (heavier in some columns)
# =====================================================================

def make_nan_columnwise():
    n, p = 60, 5
    data = RNG.standard_normal((n, p))
    # Col 0: no NaN
    # Col 1: 5% NaN
    # Col 2: 20% NaN
    # Col 3: 40% NaN
    # Col 4: 5% NaN
    for col, frac in [(1, 0.05), (2, 0.20), (3, 0.40), (4, 0.05)]:
        mask = RNG.random(n) < frac
        data[mask, col] = np.nan
    save_fixture('desc_nan_columnwise', data, {
        'description': 'Structured NaN with varying rates per column, 60x5',
    })


# =====================================================================
# Fixture 5: Perfect correlation
# =====================================================================

def make_perfect_correlation():
    n = 50
    x0 = RNG.standard_normal(n)
    x1 = 2.0 * x0 + 3.0  # perfectly correlated
    x2 = RNG.standard_normal(n)  # independent
    data = np.column_stack([x0, x1, x2])
    save_fixture('desc_perfect_correlation', data, {
        'description': 'Cols 0 and 1 perfectly correlated (r=1), col 2 independent',
    })


# =====================================================================
# Fixture 6: Many ties
# =====================================================================

def make_ties():
    n = 80
    # Integer data with many ties
    x0 = RNG.integers(1, 4, size=n).astype(np.float64)  # values 1,2,3
    x1 = RNG.integers(1, 6, size=n).astype(np.float64)  # values 1-5
    x2 = np.round(RNG.standard_normal(n), 1)  # some ties from rounding
    data = np.column_stack([x0, x1, x2])
    save_fixture('desc_ties', data, {
        'description': 'Integer and rounded data with many ties, 80x3',
    })


# =====================================================================
# Fixture 7: Single column
# =====================================================================

def make_single_column():
    n = 30
    data = (RNG.standard_normal(n) * 10 + 50).reshape(-1, 1)
    save_fixture('desc_single_column', data, {
        'description': 'Single column (p=1), 30 observations',
    })


# =====================================================================
# Fixture 8: Constant column
# =====================================================================

def make_constant_column():
    n = 40
    x0 = RNG.standard_normal(n)
    x1 = np.full(n, 7.0)  # constant
    x2 = RNG.standard_normal(n) * 3 + 1
    data = np.column_stack([x0, x1, x2])
    save_fixture('desc_constant_column', data, {
        'description': 'Col 1 is constant (var=0), cols 0 and 2 normal, 40x3',
    })


# =====================================================================
# Fixture 9: Extreme values
# =====================================================================

def make_extreme_values():
    n = 50
    x0 = RNG.standard_normal(n) * 1e6
    x1 = RNG.standard_normal(n) * 1e-6
    x2 = RNG.standard_normal(n)
    data = np.column_stack([x0, x1, x2])
    save_fixture('desc_extreme_values', data, {
        'description': 'Extreme magnitudes: col0 ~1e6, col1 ~1e-6, col2 ~1',
    })


# =====================================================================
# Fixture 10: Negative correlations
# =====================================================================

def make_negative_correlation():
    n = 100
    x0 = RNG.standard_normal(n)
    x1 = -0.9 * x0 + 0.44 * RNG.standard_normal(n)  # strong negative
    x2 = 0.8 * x0 + 0.6 * RNG.standard_normal(n)     # strong positive
    data = np.column_stack([x0, x1, x2])
    save_fixture('desc_negative_correlation', data, {
        'description': 'Strong negative (0-1) and positive (0-2) correlations',
    })


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    print("Generating descriptive statistics fixtures...")
    make_basic()
    make_large()
    make_nan_scattered()
    make_nan_columnwise()
    make_perfect_correlation()
    make_ties()
    make_single_column()
    make_constant_column()
    make_extreme_values()
    make_negative_correlation()
    print("\nDone! Now run the R validation script:")
    print("  Rscript tests/fixtures/run_r_descriptive_validation.R")
