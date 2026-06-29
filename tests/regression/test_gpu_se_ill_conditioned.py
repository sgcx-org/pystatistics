"""
GPU OLS standard errors on ill-conditioned designs (4.3.2 regression guard).

A 4.3.0 refactor switched the GPU standard-error path to invert the backend's
*float32* device Gram (``info['gram']``). On an ill-conditioned design the
float32 Gram has lost its smallest eigenvalue, so inverting it understates the
variance — the GPU reported standard errors several times too small (a wrong
coefficient that looks precise), while CPU/R reported the correct large SEs.

The fix recomputes a float64-accurate Gram on the host (``X'X``, or ``X'WX`` for
a weighted fit) and never uses the float32 device Gram for the covariance. These
tests assert the GPU OLS standard errors match the CPU (QR/float64) reference on
a near-collinear design — i.e. the SE must blow up with conditioning, not
understate — for both unweighted and weighted fits.
"""

import numpy as np
import pytest

from pystatistics.regression import fit


def _gpu_available():
    try:
        import torch
        if torch.cuda.is_available():
            return True
        mps = getattr(torch.backends, "mps", None)
        return bool(mps and mps.is_available())
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _gpu_available(), reason="GPU (CUDA or MPS) required"
)


def _ill_conditioned_design(n=4000, seed=0):
    """Near-collinear OLS design (scale-invariant condition ~2e4).

    Column 3 is column 2 plus a tiny perturbation, so X'X is ill-conditioned but
    still invertible in float64. The error term is independent of the perturbation
    (so the residual is genuine and the SEs are well-defined and large).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    xb = rng.standard_normal(n)
    xc = rng.standard_normal(n)
    err = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x1, x1 + 1e-4 * noise, xb, xc])
    y = X @ np.array([1.0, 0.5, -0.3, 0.2, 0.1]) + 0.3 * err
    return X, y


class TestGPUStandardErrorsIllConditioned:
    """GPU OLS SEs must match CPU on an ill-conditioned design, not understate."""

    def test_design_is_ill_conditioned(self):
        X, _ = _ill_conditioned_design()
        # Sanity: the design really is ill-conditioned (the regime that exposed
        # the float32-Gram bug).
        assert np.linalg.cond(X) > 1e3

    def test_unweighted_se_matches_cpu(self):
        X, y = _ill_conditioned_design()
        se_cpu = fit(X, y, backend='cpu').standard_errors
        se_gpu = fit(X, y, backend='gpu', force=True).standard_errors

        # The near-collinear coefficients have large SEs (tens), not understated.
        assert se_cpu.max() > 10.0
        np.testing.assert_allclose(
            se_gpu, se_cpu, rtol=1e-2,
            err_msg="GPU OLS standard errors diverge from CPU on an "
                    "ill-conditioned design (float32-Gram understatement?)",
        )

    def test_unweighted_se_not_understated(self):
        """The specific failure mode: GPU se_max must not be << CPU se_max."""
        X, y = _ill_conditioned_design()
        se_cpu_max = fit(X, y, backend='cpu').standard_errors.max()
        se_gpu_max = fit(X, y, backend='gpu', force=True).standard_errors.max()
        # Pre-fix this ratio was ~0.3 (or NaN); post-fix it is ~1.0.
        assert se_gpu_max > 0.5 * se_cpu_max
        assert np.isfinite(se_gpu_max)

    def test_weighted_se_matches_cpu(self):
        X, y = _ill_conditioned_design()
        rng = np.random.default_rng(7)
        w = rng.uniform(0.5, 2.0, X.shape[0])
        se_cpu = fit(X, y, backend='cpu', weights=w).standard_errors
        se_gpu = fit(X, y, backend='gpu', force=True, weights=w).standard_errors

        assert se_cpu.max() > 10.0
        np.testing.assert_allclose(
            se_gpu, se_cpu, rtol=1e-2,
            err_msg="Weighted GPU OLS standard errors diverge from CPU on an "
                    "ill-conditioned design (must recompute X'WX in float64)",
        )

    def test_well_conditioned_still_matches(self):
        """Guard against over-correcting: a well-conditioned fit must still agree."""
        rng = np.random.default_rng(1)
        n = 1000
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
        y = X @ np.array([1.0, 0.5, -0.3, 0.2]) + rng.standard_normal(n)
        se_cpu = fit(X, y, backend='cpu').standard_errors
        se_gpu = fit(X, y, backend='gpu').standard_errors
        np.testing.assert_allclose(se_gpu, se_cpu, rtol=1e-3)
