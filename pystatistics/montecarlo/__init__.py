"""
PyStatistics Monte Carlo methods.

Provides bootstrap resampling (matching R's boot package) and
permutation testing with CPU and GPU backends.

Usage:
    from pystatistics.montecarlo import boot, boot_ci, permutation_test

    # Bootstrap
    result = boot(data, statistic, n_resamples=999, seed=42)
    ci_result = boot_ci(result, ci_type="percentile")

    # Permutation test
    result = permutation_test(x, y, statistic, n_resamples=9999)
"""

from pystatistics.montecarlo.solvers import boot, boot_ci, permutation_test

__all__ = [
    "boot",
    "boot_ci",
    "permutation_test",
]
