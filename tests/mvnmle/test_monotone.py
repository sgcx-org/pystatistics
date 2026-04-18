"""Monotone missingness — detection and closed-form MLE.

These tests pin the following invariants:

  1. ``is_monotone`` / ``monotone_permutation`` correctly classify
     example datasets (true-positive on nested-missingness,
     true-negative on random MCAR).
  2. The closed-form MLE matches iterative EM to a tight tolerance
     on monotone data.
  3. ``algorithm='monotone'`` on non-monotone data raises loudly
     (Rule 1 — no silent dispatch).
  4. Correct handling of edge cases: complete data (trivially
     monotone); single-column datasets; two-pattern nested cases.
"""

import numpy as np
import pytest

from pystatistics.core.exceptions import ValidationError
from pystatistics.mvnmle import (
    datasets,
    is_monotone,
    mlest,
    mlest_monotone_closed_form,
    monotone_permutation,
)


# =====================================================================
# Detection
# =====================================================================


def test_complete_data_is_monotone():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    assert is_monotone(X) is True
    order = monotone_permutation(X)
    assert order is not None
    assert sorted(order.tolist()) == list(range(4))


def test_simple_attrition_is_monotone():
    """Classic longitudinal attrition: later waves missing when
    earlier waves missing never happens; only the later variable can
    be missing alone. Monotone with identity order."""
    X = np.array([
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [1.2, 2.2, np.nan],
        [1.3, 2.3, np.nan],
        [1.4, np.nan, np.nan],
        [1.5, np.nan, np.nan],
    ])
    assert is_monotone(X) is True
    order = monotone_permutation(X)
    # First column least-missing (never), last column most-missing.
    assert order.tolist() == [0, 1, 2]


def test_permuted_attrition_is_monotone_with_permutation():
    """Same attrition structure but with shuffled columns; detection
    must find the permutation."""
    X = np.array([
        [3.0, 1.0, 2.0],
        [3.1, 1.1, 2.1],
        [np.nan, 1.2, 2.2],
        [np.nan, 1.3, 2.3],
        [np.nan, 1.4, np.nan],
        [np.nan, 1.5, np.nan],
    ])
    assert is_monotone(X) is True
    order = monotone_permutation(X)
    # Column 1 (ref col 1) has 0 missing; col 2 has 2 missing; col 0 has 4 missing.
    assert order.tolist() == [1, 2, 0]


def test_random_mcar_is_not_monotone():
    """Random MCAR missingness on many columns essentially never
    produces a monotone pattern."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 5))
    X[rng.random(X.shape) < 0.15] = np.nan
    # Drop all-NaN rows if any
    X = X[~np.all(np.isnan(X), axis=1)]
    assert is_monotone(X) is False
    assert monotone_permutation(X) is None


def test_apple_dataset_is_monotone():
    """The classic apple dataset (v=2) has missingness only in the
    second variable — trivially monotone."""
    assert is_monotone(datasets.apple) is True


def test_missvals_dataset_is_monotone():
    """The missvals dataset has five columns with missing-row sets
    {}, {}, {9,10,11,12}, {9,10,11,12}, {6,..,12} — nested under the
    column order (2, 4, 0, 1, 3), i.e. monotone. Good corner-case
    coverage: the regression chain has to survive two zero-missing
    columns followed by ties."""
    assert is_monotone(datasets.missvals) is True
    order = monotone_permutation(datasets.missvals)
    # The ordering is stable-sort by missing count; cols 2 and 4 both
    # have zero missing. Stable sort keeps original order for ties.
    assert order.tolist() == [2, 4, 0, 1, 3]


# =====================================================================
# Closed-form MLE correctness
# =====================================================================


def _monotone_synthetic(rng, n=200, v=5, attrition_frac=0.3):
    """Generate monotone missing-data: rows 0..n1 complete, next batch
    missing last var, next batch missing last two, etc."""
    X = rng.standard_normal((n, v))
    # Drop last k columns for an increasing suffix of rows.
    for k in range(1, v):
        start = int(n * (1 - attrition_frac * k / (v - 1)))
        X[start:, v - k:] = np.nan
    assert is_monotone(X)
    return X


def test_closed_form_matches_em_on_attrition_data():
    rng = np.random.default_rng(0)
    X = _monotone_synthetic(rng, n=500, v=6, attrition_frac=0.25)

    em_result = mlest(X, algorithm='em', tol=1e-8, max_iter=5000)
    mon_result = mlest(X, algorithm='monotone')

    assert mon_result.converged
    assert em_result.converged

    # Closed-form is the *exact* MLE (up to numerical noise); EM reaches
    # it iteratively within its tolerance. Expect 6-digit agreement.
    np.testing.assert_allclose(
        mon_result.muhat, em_result.muhat, rtol=1e-5, atol=1e-7,
    )
    np.testing.assert_allclose(
        mon_result.sigmahat, em_result.sigmahat, rtol=1e-5, atol=1e-7,
    )
    assert abs(mon_result.loglik - em_result.loglik) < 1e-4


def test_closed_form_matches_em_on_apple():
    """Apple dataset: v=2, second var missing on bottom rows — a
    canonical monotone case."""
    em_result = mlest(datasets.apple, algorithm='em', tol=1e-8, max_iter=10000)
    mon_result = mlest(datasets.apple, algorithm='monotone')

    np.testing.assert_allclose(mon_result.muhat, em_result.muhat, rtol=1e-5)
    np.testing.assert_allclose(mon_result.sigmahat, em_result.sigmahat, rtol=1e-5)


def test_closed_form_matches_em_after_column_permutation():
    """Closed-form result should be invariant under column permutation
    of the input, since we undo the internal sort."""
    rng = np.random.default_rng(1)
    X = _monotone_synthetic(rng, n=200, v=4)
    perm = np.array([2, 0, 3, 1])
    X_perm = X[:, perm]

    mon_original = mlest(X, algorithm='monotone')
    mon_permuted = mlest(X_perm, algorithm='monotone')

    # The permuted fit's mu should match the original mu reordered by perm.
    np.testing.assert_allclose(
        mon_permuted.muhat, mon_original.muhat[perm], rtol=1e-10,
    )
    np.testing.assert_allclose(
        mon_permuted.sigmahat,
        mon_original.sigmahat[np.ix_(perm, perm)],
        rtol=1e-10,
    )


# =====================================================================
# Rule 1 — no silent fallback on non-monotone data
# =====================================================================


def test_monotone_algorithm_on_non_monotone_data_raises():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    X[rng.random(X.shape) < 0.2] = np.nan
    X = X[~np.all(np.isnan(X), axis=1)]
    assert not is_monotone(X)

    with pytest.raises(ValidationError, match="not monotone"):
        mlest(X, algorithm='monotone')


def test_closed_form_direct_call_on_non_monotone_raises():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 4))
    X[rng.random(X.shape) < 0.2] = np.nan
    X = X[~np.all(np.isnan(X), axis=1)]

    with pytest.raises(ValidationError, match="not monotone"):
        mlest_monotone_closed_form(X)


# =====================================================================
# Performance floor — closed-form must be faster than EM
# =====================================================================


def test_closed_form_faster_than_em_on_larger_v():
    """Closed-form should beat EM on workloads with enough variables
    that EM's iteration count grows meaningfully. Requires at least
    2× speedup on a 1500×20 monotone dataset; typical measured
    ratios are 5-20× but we keep the guard modest because EM now
    ships with SQUAREM and batched operations.

    Smaller-v cases (v ≤ 8) can have the iteration count low enough
    that EM beats closed-form in wall-clock even though closed-form
    does no iteration — the closed-form's v OLS solves have fixed
    overhead that amortises only at larger v.
    """
    import time
    rng = np.random.default_rng(0)
    X = _monotone_synthetic(rng, n=1500, v=20, attrition_frac=0.3)

    # Warmup
    _ = mlest(X, algorithm='em', max_iter=500)
    _ = mlest(X, algorithm='monotone')

    t = time.perf_counter()
    _ = mlest(X, algorithm='em', max_iter=500)
    em_time = time.perf_counter() - t

    t = time.perf_counter()
    _ = mlest(X, algorithm='monotone')
    mon_time = time.perf_counter() - t

    assert mon_time * 2 < em_time, (
        f"Closed-form {mon_time*1000:.1f} ms vs EM {em_time*1000:.1f} ms — "
        f"expected ≥ 2× speedup on a 20-variable monotone workload."
    )
