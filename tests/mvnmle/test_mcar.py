"""Regression coverage for little_mcar_test real-data ergonomics.

Landed after Project Lacuna surfaced three blockers on canonical
sklearn demo datasets (iris, wine, breast_cancer):

  1. Ill-conditioned per-pattern covariance (cond > 1e12) is the
     rule, not the exception, on real tabular data. The test must
     degrade to a pseudo-inverse with a warning by default, and
     only raise in explicit strict mode.
  2. Rows with no observed values appear routinely at realistic
     missingness rates on low-dimensional data. They contribute
     nothing; auto-drop with a warning is the user-ergonomic move.
  3. BFGS-direct scales poorly with the number of missingness
     patterns. On 200x13 with 15% missing (~100 patterns) it hangs
     past 60 s; EM — which Little's statistic can use interchangeably
     since the test depends only on the final ML estimates — returns
     in under a second. EM is now the little_mcar_test default.
"""

import time
import warnings

import numpy as np
import pytest

from pystatistics.mvnmle import little_mcar_test


def test_auto_drops_all_missing_rows_with_warning():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    X[rng.random(X.shape) < 0.2] = np.nan
    X[0, :] = np.nan  # guaranteed all-missing row

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = little_mcar_test(X)

    assert any("all values missing" in str(w.message) for w in captured)
    assert np.isfinite(result.statistic)
    assert result.df > 0


def test_strict_all_missing_rows_still_rejected():
    from pystatistics.core.exceptions import PyStatisticsError
    X = np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]])
    # PyStatisticsError covers ValidationError (the actual type raised
    # by MVNDesign.from_array). Keep ValueError / RuntimeError in the
    # allowlist for forward-compat if the implementation ever routes
    # through one of those instead.
    with pytest.raises((PyStatisticsError, ValueError, RuntimeError)):
        little_mcar_test(X, drop_all_missing_rows=False)


@pytest.mark.parametrize("loader_name", ["iris", "wine", "breast_cancer"])
def test_sklearn_demo_datasets_complete(loader_name):
    """The three canonical sklearn datasets that tripped Project Lacuna.

    Contract: at MCAR-ish 15 % missingness, each completes and
    returns a finite statistic. Whether a given run hits the
    pseudo-inverse fallback depends on the RNG draw for the mask —
    we don't assert on the warning here because the unit-level
    tests in ``test_no_silent_fallback.TestRegularizedInverse``
    already pin the regularize=True / regularize=False contracts
    directly on the inverse function.
    """
    sklearn = pytest.importorskip("sklearn.datasets")
    loader = getattr(sklearn, f"load_{loader_name}")
    X = loader().data.astype(float).copy()

    rng = np.random.default_rng(0)
    X[rng.random(X.shape) < 0.15] = np.nan
    X = X[~np.all(np.isnan(X), axis=1)]  # pre-drop to keep the test quiet

    result = little_mcar_test(X)
    assert np.isfinite(result.statistic)
    assert result.df > 0
    assert 0.0 <= result.p_value <= 1.0


def test_unconverged_mle_raises_not_silently_returned():
    """Rule 1: never return a chi-square built on non-MLE estimates.

    Forcing algorithm='direct' with a miserly max_iter on
    many-pattern data reliably fails to converge; the function must
    refuse to fabricate a statistic from whatever the optimizer's
    last iterate happened to be.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 13))
    X[rng.random(X.shape) < 0.15] = np.nan
    # This combination previously returned a noise statistic silently.
    # mlest's max_iter lives on mlest, not on little_mcar_test — so
    # we drive the convergence failure by asking for BFGS with a
    # token iteration budget via the underlying mlest flag, which
    # little_mcar_test does not currently forward. Instead, we
    # exercise the convergence check by invoking mlest directly
    # with a clipped budget and then walking through the same
    # wrapping RuntimeError the mcar_test would emit. Easiest path:
    # pass algorithm='direct' (which on this shape will exhaust the
    # default max_iter=100 with our Python-loop pattern objective).
    from pystatistics.mvnmle import little_mcar_test
    with pytest.raises(RuntimeError, match="did not converge"):
        little_mcar_test(X, algorithm="direct")


def test_many_patterns_does_not_hang():
    """EM default must keep wide multi-pattern data responsive.

    200 × 13 with 15 % missing produces ~100 unique patterns. BFGS
    on that scale ran past 60 s before timing out; EM finishes well
    under a second. The generous ceiling here is intentional — a
    regression that reinstates BFGS-direct as the default would
    blow past it even on a fast machine.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 13))
    X[rng.random(X.shape) < 0.15] = np.nan

    t = time.perf_counter()
    result = little_mcar_test(X)
    elapsed = time.perf_counter() - t

    assert elapsed < 10.0, (
        f"little_mcar_test on 200x13 with many patterns took "
        f"{elapsed:.1f}s — regression in algorithm default?"
    )
    assert np.isfinite(result.statistic)
