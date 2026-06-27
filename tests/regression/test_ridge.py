"""Tests for ridge (L2-penalized) regression: fit(l2=) and the ridge() wrapper.

Correctness of the ridge *convention* (matching MASS::lm.ridge to ~1e-15) is
checked in the pystatistics-validation drivers against R; here we test the math
properties and the API contract that don't need R:

- the augmented solve equals the direct penalized normal equations (the solver is
  correct for its stated objective),
- ridge shrinks coefficients toward zero as lambda grows (LM and GLM),
- penalized fits report NaN standard errors (no misleading inference),
- the ridge() wrapper equals fit(l2=...), and l2=0 is the unpenalized fit,
- backend='gpu_fp64' requires CUDA.
"""

import numpy as np
import pytest

from pystatistics.regression import fit, ridge, Design
from pystatistics.regression._penalty import (
    standardize, back_transform, augmented_ridge_solve,
)


def _lm_data(n=400, p=5, seed=11):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    beta = rng.standard_normal(p)
    y = X @ beta + rng.standard_normal(n)
    return X, y


def _poisson_data(n=600, p=5, seed=12):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1)) * 0.4])
    eta = X @ (rng.standard_normal(p) * 0.3)
    y = rng.poisson(np.exp(eta)).astype(float)
    return X, y


class TestRidgeSolveMath:
    def test_augmented_equals_direct_normal_equations(self):
        """The augmented-QR ridge solve must equal (Z'Z + l2 I)^-1 Z'y_c exactly."""
        X, y = _lm_data()
        Z, y_c, _ = standardize(X, y)
        l2 = 7.5
        beta_aug = augmented_ridge_solve(Z, y_c, l2)
        beta_dir = np.linalg.solve(Z.T @ Z + l2 * np.eye(Z.shape[1]), Z.T @ y_c)
        np.testing.assert_allclose(beta_aug, beta_dir, rtol=1e-10, atol=1e-12)

    def test_l2_zero_equals_unpenalized(self):
        X, y = _lm_data()
        a = fit(X, y).coefficients
        b = fit(X, y, l2=0.0).coefficients
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)

    def test_negative_lam_raises(self):
        X, y = _lm_data()
        with pytest.raises(ValueError):
            ridge(X, y, lam=-1.0)


class TestRidgeShrinkage:
    def test_lm_coefficients_shrink(self):
        X, y = _lm_data()
        norms = [np.linalg.norm(ridge(X, y, lam=lam).coefficients[1:])
                 for lam in (0.0001, 10.0, 100.0, 1000.0)]
        assert all(norms[i] >= norms[i + 1] for i in range(len(norms) - 1))

    def test_glm_coefficients_shrink(self):
        X, y = _poisson_data()
        norms = [np.linalg.norm(
            ridge(X, y, lam=lam, family="poisson").coefficients[1:])
            for lam in (0.0001, 10.0, 100.0, 1000.0)]
        assert all(norms[i] >= norms[i + 1] for i in range(len(norms) - 1))


class TestRidgeNoMisleadingInference:
    def test_lm_penalized_se_are_nan(self):
        X, y = _lm_data()
        res = ridge(X, y, lam=10.0)
        assert np.all(np.isnan(res.standard_errors))
        assert np.all(np.isnan(res.p_values))

    def test_glm_penalized_se_are_nan(self):
        X, y = _poisson_data()
        res = ridge(X, y, lam=10.0, family="poisson")
        assert np.all(np.isnan(res.standard_errors))


class TestRidgeWrapper:
    def test_ridge_equals_fit_l2_lm(self):
        X, y = _lm_data()
        np.testing.assert_allclose(
            ridge(X, y, lam=12.0).coefficients,
            fit(X, y, l2=12.0).coefficients, rtol=1e-12, atol=1e-12)

    def test_ridge_equals_fit_l2_glm(self):
        X, y = _poisson_data()
        np.testing.assert_allclose(
            ridge(X, y, lam=12.0, family="poisson").coefficients,
            fit(X, y, family="poisson", l2=12.0).coefficients, rtol=1e-10, atol=1e-12)


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class TestGpuFp64Backend:
    @pytest.mark.skipif(_cuda_available(),
                        reason="On a CUDA box gpu_fp64 succeeds; this tests the no-CUDA error")
    def test_gpu_fp64_requires_cuda(self):
        """Without CUDA, backend='gpu_fp64' must fail loudly (no float64 GPU)."""
        X, y = _poisson_data()
        with pytest.raises(RuntimeError):
            fit(X, y, family="poisson", backend="gpu_fp64")
