#!/usr/bin/env python3
"""Diagnostic script for near_square fixture failure."""

import numpy as np
import json
from pathlib import Path
from pystatistics import DataSource
from pystatistics.regression import Design, fit

# Load the fixture
fixture_dir = Path('tests/fixtures')
fixture_name = 'near_square'

csv_path = fixture_dir / f"{fixture_name}.csv"
meta_path = fixture_dir / f"{fixture_name}_meta.json"
r_results_path = fixture_dir / f"{fixture_name}_r_results.json"

with open(meta_path) as f:
    meta = json.load(f)

with open(r_results_path) as f:
    r_results = json.load(f)

# Get R's values
r_fitted = np.array(r_results['fitted_all'])
r_residuals = np.array(r_results['residuals_all'])
r_rss = r_results['rss']
r_coeffs = np.array(r_results['coefficients'])

# Load data and fit with PyStatistics (same as validation script)
ds = DataSource.from_file(csv_path)
x_cols = sorted(k for k in ds.keys() if k != 'y')
design = Design.from_datasource(ds, x=x_cols, y='y')
result = fit(design)

print("=" * 70)
print("DIAGNOSTIC: near_square fixture")
print("=" * 70)

print("\n=== Result object attributes ===")
print([attr for attr in dir(result) if not attr.startswith('_')])

print("\n=== Dataset Info ===")
print(f"n = {meta['n']}, p = {meta['p']}")
print(f"condition_number = {meta['condition_number']:.2f}")
print(f"df_residual from R: {r_results.get('df_residual', '?')}")
print(f"R rank: {r_results.get('rank', '?')}")
print(f"Number of coefficients: Python={len(result.coefficients)}, R={len(r_coeffs)}")

print("\n=== Prediction-Space Metrics ===")
fitted_diff = np.max(np.abs(result.fitted_values - r_fitted))
resid_diff = np.max(np.abs(result.residuals - r_residuals))

print(f"Fitted values max abs diff: {fitted_diff:.2e}")
print(f"Residuals max abs diff:     {resid_diff:.2e}")

# Try to find RSS - might be named differently
if hasattr(result, 'rss'):
    rss_diff = abs(result.rss - r_rss)
    print(f"RSS abs diff:               {rss_diff:.2e}")
elif hasattr(result, 'ssr'):
    rss_diff = abs(result.ssr - r_rss)
    print(f"RSS (as ssr) abs diff:      {rss_diff:.2e}")
else:
    # Compute from residuals
    our_rss = np.sum(result.residuals ** 2)
    rss_diff = abs(our_rss - r_rss)
    print(f"RSS (computed) abs diff:    {rss_diff:.2e}")

print("\n=== Coefficient-Space Metrics ===")
coef_diff = np.max(np.abs(result.coefficients - r_coeffs))
print(f"Coefficients max abs diff:  {coef_diff:.2e}")

# Find where the biggest coefficient difference is
coef_abs_diff = np.abs(result.coefficients - r_coeffs)
worst_idx = np.argmax(coef_abs_diff)
print(f"\nWorst coefficient mismatch at index {worst_idx}:")
print(f"  Python: {result.coefficients[worst_idx]:.6f}")
print(f"  R:      {r_coeffs[worst_idx]:.6f}")

print("\n=== Diagnosis ===")
if fitted_diff < 1e-8 and resid_diff < 1e-8:
    print("✓ Fitted values and residuals MATCH R closely")
    print("  -> Model predictions are correct")
    if coef_diff > 0.01:
        print("✗ But coefficients differ significantly")
        print("  -> This suggests PIVOT ORDERING mismatch")
        print("  -> Coefficients are likely in different column order")
else:
    print("✗ Fitted values or residuals do NOT match R")
    print("  -> Deeper algorithmic difference (not just pivot order)")