"""SQUAREM acceleration correctness tests.

SQUAREM extrapolates the EM sequence for speed while preserving the
MLE. These tests pin the invariants we care about:

  - Accelerated EM converges to the same MLE as plain EM (within
    numerical tolerance appropriate for a parameter-space
    convergence criterion).
  - The accelerated path uses strictly fewer EM-equivalent steps
    on problems where SQUAREM is supposed to help.
  - Monotonicity of the observed-data log-likelihood is preserved
    by the safeguarded line search.

Correctness at the bit level against R mvnmle is covered by the
existing ``test_em.py`` / ``test_mlest.py`` R-reference tests,
which all use the default (accelerated) path — so the fact that
they pass already certifies SQUAREM doesn't break MLE identification
on apple and missvals.
"""

import numpy as np
import pytest

from pystatistics.mvnmle import datasets, mlest


def test_squarem_matches_plain_em_to_convergence_tolerance():
    """Accelerated EM and plain EM land on the same MLE on apple."""
    accelerated = mlest(datasets.apple, algorithm="em")
    # Work around the fact that ``accelerate`` is a backend-level flag.
    # We reach into the backend by constructing one directly.
    from pystatistics.mvnmle.backends.em import EMBackend
    from pystatistics.mvnmle.design import MVNDesign
    design = MVNDesign.from_array(datasets.apple)
    plain = EMBackend(device="cpu").solve(
        design, tol=1e-8, max_iter=10000, accelerate=False,
    )

    np.testing.assert_allclose(
        accelerated.muhat, plain.params.muhat, rtol=1e-6, atol=1e-8,
    )
    np.testing.assert_allclose(
        accelerated.sigmahat, plain.params.sigmahat, rtol=1e-6, atol=1e-8,
    )
    assert abs(accelerated.loglik - plain.params.loglik) < 1e-6


def test_squarem_reduces_iteration_count_on_missvals():
    """SQUAREM should cut EM-equivalent steps on missvals (where plain
    EM historically takes ~240 iterations)."""
    from pystatistics.mvnmle.backends.em import EMBackend
    from pystatistics.mvnmle.design import MVNDesign
    design = MVNDesign.from_array(datasets.missvals)

    plain = EMBackend(device="cpu").solve(
        design, tol=1e-4, max_iter=10000, accelerate=False,
    )
    accel = EMBackend(device="cpu").solve(
        design, tol=1e-4, max_iter=10000, accelerate=True,
    )

    assert plain.params.converged and accel.params.converged
    # Plain EM converges in ~240 steps; accelerated consumes ~100
    # EM-equivalent calls. Require at least 1.5× fewer — generous
    # guardrail; typical ratio is 2-3×.
    assert accel.params.n_iter < plain.params.n_iter / 1.5, (
        f"SQUAREM used {accel.params.n_iter} EM-equivalent steps vs "
        f"plain EM's {plain.params.n_iter}; expected substantial reduction."
    )


def test_squarem_monotonicity_is_respected():
    """The safeguarded SQUAREM step must not decrease log-likelihood.

    We recover the loglik history by calling mlest with progressively
    larger max_iter caps and checking the loglik at each cap.
    """
    caps = [5, 10, 20, 50, 100, 500]
    logliks = []
    for cap in caps:
        r = mlest(datasets.missvals, algorithm="em", max_iter=cap)
        logliks.append(r.loglik)

    # Log-likelihood should be monotonically non-decreasing in
    # iteration count. Small numerical noise allowed (1e-10).
    for a, b in zip(logliks[:-1], logliks[1:]):
        assert b >= a - 1e-10, (
            f"SQUAREM violated monotonicity: loglik {a:.10f} → {b:.10f}"
        )


def test_squarem_on_sklearn_wine_matches_plain_em():
    """Real-data regression: wine (13 variables, ~100 patterns) was
    where SQUAREM's wall-clock win was most visible; both paths must
    still agree on the MLE."""
    sklearn = pytest.importorskip("sklearn.datasets")
    X = sklearn.load_wine().data.astype(float).copy()
    rng = np.random.default_rng(0)
    X[rng.random(X.shape) < 0.15] = np.nan
    X = X[~np.all(np.isnan(X), axis=1)]

    from pystatistics.mvnmle.backends.em import EMBackend
    from pystatistics.mvnmle.design import MVNDesign
    design = MVNDesign.from_array(X)

    accel = EMBackend(device="cpu").solve(
        design, tol=1e-4, max_iter=5000, accelerate=True,
    )
    plain = EMBackend(device="cpu").solve(
        design, tol=1e-4, max_iter=5000, accelerate=False,
    )

    np.testing.assert_allclose(
        accel.params.muhat, plain.params.muhat, rtol=1e-4, atol=1e-6,
    )
    np.testing.assert_allclose(
        accel.params.sigmahat, plain.params.sigmahat, rtol=1e-4, atol=1e-6,
    )
    # Both converged to the same MLE up to the tol parameter.
    assert abs(accel.params.loglik - plain.params.loglik) < 1e-2
