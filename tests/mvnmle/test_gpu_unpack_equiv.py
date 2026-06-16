"""GPU ``_unpack_gpu`` Sigma == canonical ``CholeskyParameterization.unpack``.

The GPU FP64 objective historically reconstructed the Cholesky factor's
off-diagonals with a hand-rolled *column-major* loop, while the canonical
NumPy parameterization (and the FP32 GPU path) place them *row-major* via
``tril_indices``. For ``n_vars >= 3`` the two orderings differ, so the FP64
path optimised one covariance while ``extract_parameters`` reported another.

Both GPU paths now share :func:`unpack_cholesky`, so these tests assert that
every standard-Cholesky reconstruction — the shared helper directly, and each
GPU objective's ``_unpack_gpu`` — matches the canonical reference to tight
tolerance, across ``n_vars in {3, 4, 8}`` and both float precisions, on the CPU
torch device (the helper is device-agnostic and MPS forbids FP64).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pystatistics.mvnmle._objectives.gpu_fp32 import GPUObjectiveFP32
from pystatistics.mvnmle._objectives.gpu_fp64 import GPUObjectiveFP64
from pystatistics.mvnmle._objectives.parameterizations import (
    CholeskyParameterization,
)
from pystatistics.mvnmle._objectives._batched_cholesky import unpack_cholesky

DTYPES = {torch.float32: 1e-5, torch.float64: 1e-11}


def _canonical_theta(n_vars: int, seed: int):
    """A valid standard-Cholesky parameter vector and its canonical (mu, Sigma).

    Built by packing a well-conditioned PD covariance, so the off-diagonal
    Cholesky entries are non-trivial (the bug only manifests when they are).
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_vars, n_vars))
    sigma = a @ a.T + n_vars * np.eye(n_vars)  # symmetric PD
    mu = rng.standard_normal(n_vars)

    param = CholeskyParameterization(n_vars)
    theta = param.pack(mu, sigma)
    mu_ref, sigma_ref = param.unpack(theta)
    return theta, mu_ref, sigma_ref


def _complete_data(n_vars: int, seed: int) -> np.ndarray:
    """Complete (no-missing) data so a GPU objective can be constructed."""
    rng = np.random.default_rng(seed + 1000)
    return rng.standard_normal((200, n_vars))


# --------------------------------------------------------------------------
# The shared helper directly
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_vars", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("dtype,atol", list(DTYPES.items()))
def test_unpack_cholesky_matches_canonical(n_vars, dtype, atol):
    theta, mu_ref, sigma_ref = _canonical_theta(n_vars, seed=n_vars)

    theta_t = torch.tensor(theta, dtype=dtype)
    mu_t, sigma_t = unpack_cholesky(torch, theta_t, n_vars)

    assert mu_t.shape == (n_vars,)
    assert sigma_t.shape == (n_vars, n_vars)
    np.testing.assert_allclose(mu_t.numpy(), mu_ref, atol=atol, rtol=0)
    np.testing.assert_allclose(sigma_t.numpy(), sigma_ref, atol=atol, rtol=0)
    # Reconstructed Sigma must be exactly symmetric.
    np.testing.assert_array_equal(sigma_t.numpy(), sigma_t.numpy().T)


def test_unpack_cholesky_is_differentiable():
    """The helper must preserve the autograd graph (gradients flow to theta)."""
    theta, _, _ = _canonical_theta(4, seed=4)
    theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)

    _, sigma_t = unpack_cholesky(torch, theta_t, 4)
    sigma_t.sum().backward()

    assert theta_t.grad is not None
    assert torch.isfinite(theta_t.grad).all()


# --------------------------------------------------------------------------
# Through each GPU objective's _unpack_gpu
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [GPUObjectiveFP32, GPUObjectiveFP64])
@pytest.mark.parametrize("n_vars", [3, 4, 8])
@pytest.mark.parametrize("dtype,atol", list(DTYPES.items()))
def test_objective_unpack_matches_canonical(cls, n_vars, dtype, atol):
    theta, mu_ref, sigma_ref = _canonical_theta(n_vars, seed=n_vars)
    data = _complete_data(n_vars, seed=n_vars)

    with warnings.catch_warnings():
        # FP64 on the CPU device warns about performance; irrelevant here.
        warnings.simplefilter("ignore")
        obj = cls(data, device="cpu")

    theta_t = torch.tensor(theta, dtype=dtype)
    mu_t, sigma_t = obj._unpack_gpu(theta_t)

    np.testing.assert_allclose(mu_t.detach().numpy(), mu_ref, atol=atol, rtol=0)
    np.testing.assert_allclose(
        sigma_t.detach().numpy(), sigma_ref, atol=atol, rtol=0
    )


def test_fp32_and_fp64_unpack_agree():
    """Both GPU paths must now produce the same Sigma from the same theta."""
    n_vars = 5
    theta, _, sigma_ref = _canonical_theta(n_vars, seed=99)
    data = _complete_data(n_vars, seed=99)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o32 = GPUObjectiveFP32(data, device="cpu")
        o64 = GPUObjectiveFP64(data, device="cpu")

    theta32 = torch.tensor(theta, dtype=torch.float32)
    theta64 = torch.tensor(theta, dtype=torch.float64)
    _, s32 = o32._unpack_gpu(theta32)
    _, s64 = o64._unpack_gpu(theta64)

    # Both track the canonical reference; they agree to FP32 precision.
    np.testing.assert_allclose(s32.numpy(), sigma_ref, atol=1e-5, rtol=0)
    np.testing.assert_allclose(s64.numpy(), sigma_ref, atol=1e-11, rtol=0)
    np.testing.assert_allclose(s32.numpy(), s64.numpy(), atol=1e-5, rtol=0)
