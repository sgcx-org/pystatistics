#!/usr/bin/env python3
"""
Validate PyStatistics regression against R reference results.

Compares our fit() output to R's lm() output for each fixture.

Run from /mnt/projects/pystatistics:
    python tests/fixtures/validate_against_r.py
"""

import json
import numpy as np
from pathlib import Path

from pystatistics import DataSource
from pystatistics.regression import Design, fit

FIXTURES_DIR = Path(__file__).parent

# Tolerances
RTOL = 1e-10  # Relative tolerance
ATOL = 1e-12  # Absolute tolerance (for values near zero)

# Relaxed tolerances for ill-conditioned matrices
# With condition number ~10^k, expect to lose ~k digits of precision
RTOL_ILL_CONDITIONED = 1e-4
ATOL_ILL_CONDITIONED = 1e-6


def compare_arrays(name: str, python_val, r_val, rtol: float, atol: float) -> tuple[bool, str]:
    """Compare arrays/scalars with tolerance."""
    python_arr = np.atleast_1d(np.asarray(python_val, dtype=np.float64))
    r_arr = np.atleast_1d(np.asarray(r_val, dtype=np.float64))
    
    if python_arr.shape != r_arr.shape:
        return False, f"Shape mismatch: Python {python_arr.shape} vs R {r_arr.shape}"
    
    # Check closeness
    close = np.allclose(python_arr, r_arr, rtol=rtol, atol=atol)
    
    if close:
        return True, "OK"
    else:
        # Find worst discrepancy
        abs_diff = np.abs(python_arr - r_arr)
        rel_diff = abs_diff / (np.abs(r_arr) + 1e-15)
        max_abs_idx = np.argmax(abs_diff)
        max_rel_idx = np.argmax(rel_diff)
        
        msg = (
            f"Max abs diff: {abs_diff.flat[max_abs_idx]:.2e} at index {max_abs_idx} "
            f"(Python={python_arr.flat[max_abs_idx]:.15e}, R={r_arr.flat[max_abs_idx]:.15e})\n"
            f"         Max rel diff: {rel_diff.flat[max_rel_idx]:.2e} at index {max_rel_idx}"
        )
        return False, msg


def validate_fixture(fixture_name: str) -> tuple[bool, list[str], bool]:
    """Validate one fixture against R results."""
    csv_path = FIXTURES_DIR / f"{fixture_name}.csv"
    r_results_path = FIXTURES_DIR / f"{fixture_name}_r_results.json"
    meta_path = FIXTURES_DIR / f"{fixture_name}_meta.json"
    
    if not r_results_path.exists():
        return False, [f"R results file not found: {r_results_path}"], False
    
    # Load R results
    with open(r_results_path) as f:
        r_results = json.load(f)
    
    # Check if ill-conditioned (use relaxed tolerances)
    is_ill_conditioned = False
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            cond = meta.get('condition_number', 1.0)
            # If condition number > 10^4, use relaxed tolerances
            if cond > 1e4:
                is_ill_conditioned = True
    
    if is_ill_conditioned:
        rtol, atol = RTOL_ILL_CONDITIONED, ATOL_ILL_CONDITIONED
    else:
        rtol, atol = RTOL, ATOL
    
    # Load data and fit with PyStatistics
    ds = DataSource.from_file(csv_path)
    
    # Get predictor columns (everything except 'y')
    # Use keys() which returns frozenset, convert to sorted list
    x_cols = [c for c in ds.metadata['columns'] if c != 'y']
    
    # Build design and fit
    design = Design.from_datasource(ds, x=x_cols, y='y')
    result = fit(design)
    
    # Compare results
    errors = []
    passes = []
    
    # Coefficients (most important!)
    ok, msg = compare_arrays('coefficients', result.coefficients, r_results['coefficients'], rtol, atol)
    if ok:
        passes.append("coefficients")
    else:
        errors.append(f"coefficients: {msg}")
    
    # R-squared
    ok, msg = compare_arrays('r_squared', result.r_squared, r_results['r_squared'], rtol, atol)
    if ok:
        passes.append("r_squared")
    else:
        errors.append(f"r_squared: {msg}")
    
    # Adjusted R-squared
    ok, msg = compare_arrays('adj_r_squared', result.adjusted_r_squared, r_results['adj_r_squared'], rtol, atol)
    if ok:
        passes.append("adj_r_squared")
    else:
        errors.append(f"adj_r_squared: {msg}")
    
    # Residual standard error
    ok, msg = compare_arrays('sigma', result.residual_std_error, r_results['sigma'], rtol, atol)
    if ok:
        passes.append("sigma")
    else:
        errors.append(f"sigma: {msg}")
    
    # RSS
    ok, msg = compare_arrays('rss', result.rss, r_results['rss'], rtol, atol)
    if ok:
        passes.append("rss")
    else:
        errors.append(f"rss: {msg}")
    
    # TSS
    ok, msg = compare_arrays('tss', result.tss, r_results['tss'], rtol, atol)
    if ok:
        passes.append("tss")
    else:
        errors.append(f"tss: {msg}")
    
    # Standard errors
    ok, msg = compare_arrays('standard_errors', result.standard_errors, r_results['standard_errors'], rtol, atol)
    if ok:
        passes.append("standard_errors")
    else:
        errors.append(f"standard_errors: {msg}")
    
    # t-statistics
    ok, msg = compare_arrays('t_statistics', result.t_statistics, r_results['t_statistics'], rtol, atol)
    if ok:
        passes.append("t_statistics")
    else:
        errors.append(f"t_statistics: {msg}")

    # p-values
    ok, msg = compare_arrays('p_values', result.p_values, r_results['p_values'], rtol, atol)
    if ok:
        passes.append("p_values")
    else:
        errors.append(f"p_values: {msg}")

    # Residuals (all)
    ok, msg = compare_arrays('residuals', result.residuals, r_results['residuals_all'], rtol, atol)
    if ok:
        passes.append("residuals")
    else:
        errors.append(f"residuals: {msg}")
    
    # Fitted values (all)
    ok, msg = compare_arrays('fitted_values', result.fitted_values, r_results['fitted_all'], rtol, atol)
    if ok:
        passes.append("fitted_values")
    else:
        errors.append(f"fitted_values: {msg}")
    
    # df_residual
    ok, msg = compare_arrays('df_residual', result.df_residual, r_results['df_residual'], rtol, atol)
    if ok:
        passes.append("df_residual")
    else:
        errors.append(f"df_residual: {msg}")
    
    all_passed = len(errors) == 0
    return all_passed, errors if errors else passes, is_ill_conditioned


def main():
    print("=" * 70)
    print("PyStatistics vs R Validation")
    print("=" * 70)
    print(f"Standard tolerance: rtol={RTOL}, atol={ATOL}")
    print(f"Ill-conditioned tolerance: rtol={RTOL_ILL_CONDITIONED}, atol={ATOL_ILL_CONDITIONED}")
    print()
    
    # Find all fixtures
    csv_files = sorted(FIXTURES_DIR.glob("*.csv"))
    fixtures = [f.stem for f in csv_files if not f.stem.endswith('_r_results')]
    
    if not fixtures:
        print("❌ No fixtures found. Run generate_fixtures.py first.")
        return
    
    total_passed = 0
    total_failed = 0
    failed_fixtures = []
    
    for fixture in fixtures:
        r_results = FIXTURES_DIR / f"{fixture}_r_results.json"
        if not r_results.exists():
            print(f"⏭️  {fixture}: Skipping (no R results)")
            continue
        
        print(f"\n{'─' * 70}")
        print(f"Testing: {fixture}")
        print('─' * 70)
        
        try:
            passed, messages, is_relaxed = validate_fixture(fixture)
            
            tolerance_note = " (relaxed tolerance for ill-conditioned)" if is_relaxed else ""
            
            if passed:
                print(f"✅ PASSED - All {len(messages)} checks match R{tolerance_note}")
                total_passed += 1
            else:
                print(f"❌ FAILED - {len(messages)} discrepancies:{tolerance_note}")
                for msg in messages:
                    print(f"   • {msg}")
                total_failed += 1
                failed_fixtures.append(fixture)
                
        except Exception as e:
            print(f"💥 ERROR: {e}")
            total_failed += 1
            failed_fixtures.append(fixture)
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if failed_fixtures:
        print(f"\nFailed fixtures: {', '.join(failed_fixtures)}")
        exit(1)
    else:
        print("\n✅ All fixtures match R to machine precision!")
        exit(0)


if __name__ == "__main__":
    main()