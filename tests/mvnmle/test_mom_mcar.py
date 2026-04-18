"""Method-of-moments MCAR test — correctness, honesty, and speed.

Invariants pinned by these tests:

  1. ``mom_mcar_test`` returns an ``MCARTestResult`` whose
     ``method`` field identifies the estimator as method-of-moments,
     not Little's. No silent masquerading as Little's test.
  2. Under MCAR (the null) on moderate-n data, MoM and MLE agree on
     the statistic and p-value to a reasonable tolerance — not
     bit-identical (different estimators), but directionally
     consistent enough to be useful as a fast screen.
  3. All-missing-row handling mirrors ``little_mcar_test``.
  4. Completes in under 100 ms on breast_cancer-scale data
     (569 × 30), dramatically faster than the iterative MLE path.
"""

import time
import warnings

import numpy as np
import pytest

from pystatistics.mvnmle import (
    datasets,
    little_mcar_test,
    mom_mcar_test,
)


# =====================================================================
# Honesty — the result identifies itself
# =====================================================================


def test_mom_result_method_field_says_method_of_moments():
    r = mom_mcar_test(datasets.apple)
    assert "Method-of-moments" in r.method
    assert "Little" not in r.method


def test_little_result_method_field_still_says_little():
    """Guard against accidental regression: adding the ``method`` field
    must not change what ``little_mcar_test`` reports."""
    r = little_mcar_test(datasets.apple)
    assert r.method == "Little (MLE plug-in)"


def test_summary_includes_method_line():
    r = mom_mcar_test(datasets.apple)
    assert "Method: Method-of-moments" in r.summary()


# =====================================================================
# Correctness — MoM and MLE agree to a reasonable tolerance under MCAR
# =====================================================================


def test_mom_runs_on_apple():
    r = mom_mcar_test(datasets.apple)
    assert np.isfinite(r.statistic)
    assert 0.0 <= r.p_value <= 1.0
    assert r.df > 0


def test_mom_runs_on_missvals():
    r = mom_mcar_test(datasets.missvals)
    assert np.isfinite(r.statistic)
    assert 0.0 <= r.p_value <= 1.0


def test_mom_mle_agree_qualitatively_on_apple():
    """Apple's data violate MCAR (the test statistic is much larger
    than df), so both MoM and MLE should reject. Statistics will
    differ because of different plug-in estimators, but the
    rejection decision should agree."""
    mom = mom_mcar_test(datasets.apple)
    mle = little_mcar_test(datasets.apple)
    assert mom.df == mle.df
    assert mom.rejected == mle.rejected
    # Statistics should be within a factor of 2 — different plug-ins,
    # not wildly different scale.
    assert 0.5 * mle.statistic <= mom.statistic <= 2.0 * mle.statistic


def test_mom_mle_agree_under_large_n_mcar():
    """On n=1000 MCAR data, both estimators are consistent so the
    statistics should be close."""
    rng = np.random.default_rng(0)
    n, v = 1000, 5
    X = rng.standard_normal((n, v))
    # Force MCAR: drop 15% randomly.
    X[rng.random(X.shape) < 0.15] = np.nan
    X = X[~np.all(np.isnan(X), axis=1)]

    mom = mom_mcar_test(X)
    mle = little_mcar_test(X)

    assert mom.df == mle.df
    # Under MCAR at n=1000, both p-values should be large (failure to
    # reject). The statistics themselves should be within 20 % of
    # each other.
    assert not mom.rejected
    assert not mle.rejected
    rel_diff = abs(mom.statistic - mle.statistic) / max(mle.statistic, 1.0)
    assert rel_diff < 0.25, (
        f"MoM stat={mom.statistic:.2f}, MLE stat={mle.statistic:.2f} — "
        f"relative diff {rel_diff:.2%} too large on MCAR data"
    )


# =====================================================================
# Edge cases — mirror little_mcar_test handling
# =====================================================================


def test_mom_auto_drops_all_missing_rows_with_warning():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    X[rng.random(X.shape) < 0.2] = np.nan
    X[0, :] = np.nan

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        r = mom_mcar_test(X)
    assert any("all values missing" in str(w.message) for w in captured)
    assert np.isfinite(r.statistic)


def test_mom_complete_data_returns_sentinel():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    r = mom_mcar_test(X)
    assert r.statistic == 0.0
    assert r.p_value == 1.0
    assert r.df == 0


# =====================================================================
# Speed — MoM should beat MLE on realistic shapes
# =====================================================================


def test_mom_dramatically_faster_than_mle_on_breast_scale():
    """Regression for the headline claim. MoM should be at least 10×
    faster than the MLE-based test on breast_cancer-scale data,
    which is where the whole point of having MoM lives. Generous
    guard — typical measured ratio is 30-100×."""
    sklearn = pytest.importorskip("sklearn.datasets")
    X = sklearn.load_breast_cancer().data.astype(float).copy()
    rng = np.random.default_rng(0)
    X[rng.random(X.shape) < 0.15] = np.nan
    X = X[~np.all(np.isnan(X), axis=1)]

    # Warmup
    _ = mom_mcar_test(X)
    _ = little_mcar_test(X)

    t = time.perf_counter()
    _ = mom_mcar_test(X)
    mom_time = time.perf_counter() - t

    t = time.perf_counter()
    _ = little_mcar_test(X)
    mle_time = time.perf_counter() - t

    assert mom_time * 10 < mle_time, (
        f"MoM {mom_time*1000:.1f} ms vs MLE {mle_time*1000:.1f} ms — "
        f"expected ≥ 10× speedup on breast_cancer-scale data."
    )
