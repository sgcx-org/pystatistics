"""GPU (fp32) parity for prior weights / offset, against the CPU reference.

The CPU path is validated against R to round-off in
``test_weights_offset_r_validation.py``; here we only assert that the float32 GPU
backends (OLS Cholesky and GLM IRLS) reproduce the CPU fit to float32 tolerance
with ``weights=`` and ``offset=`` supplied. Skipped when no GPU is available.
"""

import numpy as np
import pytest


def _gpu_available():
    try:
        import torch
        if torch.cuda.is_available():
            return True
        mps = getattr(torch.backends, "mps", None)
        return bool(mps and mps.is_available())
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _gpu_available(), reason="GPU (CUDA or MPS) required")

from pystatistics.regression import fit  # noqa: E402

# float32 relative tolerance for coefficients vs the float64 CPU fit.
FP32_RTOL = 2e-3
FP32_ATOL = 2e-4


@pytest.fixture
def wls_data():
    rng = np.random.default_rng(10)
    n = 200
    X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    y = X @ np.array([1.0, 2.0, -0.5]) + rng.standard_normal(n) * 0.5
    w = rng.uniform(0.3, 3.0, n)
    off = rng.standard_normal(n) * 0.3
    return X, y, w, off


def test_gpu_wls_offset_matches_cpu(wls_data):
    X, y, w, off = wls_data
    cpu = fit(X, y, weights=w, offset=off, backend="cpu")
    gpu = fit(X, y, weights=w, offset=off, backend="gpu")
    np.testing.assert_allclose(gpu.coefficients, cpu.coefficients, rtol=FP32_RTOL, atol=FP32_ATOL)
    np.testing.assert_allclose(gpu.standard_errors, cpu.standard_errors, rtol=5e-3, atol=1e-4)


@pytest.mark.parametrize("family", ["binomial", "poisson"])
def test_gpu_glm_weights_offset_matches_cpu(family):
    rng = np.random.default_rng(11)
    n = 300
    X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    eta = X @ np.array([0.3, 0.8, -0.5])
    if family == "binomial":
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    else:
        y = rng.poisson(np.exp(eta)).astype(float)
    w = rng.uniform(0.5, 2.0, n)
    off = rng.standard_normal(n) * 0.2

    cpu = fit(X, y, family=family, weights=w, offset=off, backend="cpu")
    gpu = fit(X, y, family=family, weights=w, offset=off, backend="gpu", tol=1e-6, max_iter=200)
    np.testing.assert_allclose(gpu.coefficients, cpu.coefficients, rtol=FP32_RTOL, atol=FP32_ATOL)
    np.testing.assert_allclose(gpu.deviance, cpu.deviance, rtol=FP32_RTOL)
    np.testing.assert_allclose(gpu.null_deviance, cpu.null_deviance, rtol=FP32_RTOL)
