"""Batched GPU objective kernel == per-pattern looped reference.

The GPU forward-Cholesky objective was changed from a Python loop over
missingness patterns to a single batched Cholesky (pystatistics 3.3.0). These
tests guard that refactor: the batched kernel ``batched_neg2_loglik`` and its
autodiff gradient must match an independent, deliberately naive NumPy loop over
patterns implementing the exact formula the looped objective used,

    f = sum_k [ n_k * log|Sigma_k + eps I| + tr((Sigma_k + eps I)^{-1} M_k) ],
    M_k = sum_i (y_i - mu_k)(y_i - mu_k)^T,  Sigma_k = observed sub-block.

The kernel is fed explicit ``(mu, Sigma)`` so the test does not depend on either
parameter unpacking path (the FP64 ``_unpack_gpu`` has a separate, pre-existing
reconstruction discrepancy tracked outside this refactor). Both precisions are
exercised on the CPU torch device — the kernel is device-agnostic, and MPS
forbids FP64.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pystatistics.mvnmle._objectives.gpu_fp64 import GPUObjectiveFP64
from pystatistics.mvnmle._objectives._batched_cholesky import (
    build_batched_constants,
    to_torch,
    batched_neg2_loglik,
)

EPS = 1e-6


def _numpy_loop(patterns, mu, sigma, eps=EPS) -> float:
    """Naive per-pattern loop in NumPy — the oracle the batch must match."""
    total = 0.0
    for pat in patterns:
        oi = np.asarray(pat.observed_indices)
        if oi.size == 0:
            continue
        n = float(pat.n_obs)
        mu_k = mu[oi]
        sig_k = sigma[np.ix_(oi, oi)] + eps * np.eye(oi.size)
        d = np.asarray(pat.data, dtype=np.float64) - mu_k
        M_k = d.T @ d
        sign, logdet = np.linalg.slogdet(sig_k)
        trace = np.trace(np.linalg.solve(sig_k, M_k))
        total += n * logdet + trace
    return total


def _make_data(seed=0, n=240, p=6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, p))
    cov = A @ A.T + p * np.eye(p)
    X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    X[0:60, 0] = np.nan
    X[40:120, 3] = np.nan
    X[100:160, 1] = np.nan
    X[150:200, 4:6] = np.nan
    return X.astype(np.float64)


def _spd(p, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, p))
    return rng.standard_normal(p), A @ A.T + p * np.eye(p)


@pytest.mark.parametrize("torch_dtype,tol", [
    (torch.float32, 1e-3),
    (torch.float64, 1e-9),
])
def test_kernel_value_matches_loop(torch_dtype, tol):
    X = _make_data()
    obj = GPUObjectiveFP64(X, device="cpu")
    ct = to_torch(build_batched_constants(obj.patterns, obj.n_vars),
                  torch, "cpu", torch_dtype)
    for seed in (1, 2, 3):
        mu, sigma = _spd(obj.n_vars, seed)
        got = batched_neg2_loglik(
            torch, torch.tensor(mu, dtype=torch_dtype),
            torch.tensor(sigma, dtype=torch_dtype), ct, EPS).item()
        want = _numpy_loop(obj.patterns, mu, sigma)
        assert np.isclose(got, want, rtol=tol), f"batched {got} vs loop {want}"


def test_kernel_value_all_observed():
    """Edge case: no missingness — a single full-dimensional pattern."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((150, 4)).astype(np.float64)
    obj = GPUObjectiveFP64(X, device="cpu")
    ct = to_torch(build_batched_constants(obj.patterns, obj.n_vars),
                  torch, "cpu", torch.float64)
    mu, sigma = _spd(4, 5)
    got = batched_neg2_loglik(torch, torch.tensor(mu), torch.tensor(sigma),
                              ct, EPS).item()
    assert np.isclose(got, _numpy_loop(obj.patterns, mu, sigma), rtol=1e-9)


_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


@pytest.mark.skipif(not _MPS, reason="requires Apple Metal (MPS)")
def test_batched_kernel_runs_on_mps():
    """Guard MPS op-coverage: every op in the batched kernel (gather,
    diag_embed, batched cholesky, batched triangular solve) must run on Metal.
    The CUDA-gated GPU suite skips on Apple Silicon, so this is the only check
    that the FP32 path — the one that actually ships to MPS users — executes."""
    X = _make_data()
    obj = GPUObjectiveFP64(X, device="cpu")  # used only to build patterns
    ct = to_torch(build_batched_constants(obj.patterns, obj.n_vars),
                  torch, "mps", torch.float32)
    mu, sigma = _spd(obj.n_vars, 1)
    got = batched_neg2_loglik(
        torch, torch.tensor(mu, dtype=torch.float32, device="mps"),
        torch.tensor(sigma, dtype=torch.float32, device="mps"), ct, EPS).item()
    assert np.isclose(got, _numpy_loop(obj.patterns, mu, sigma), rtol=1e-2)


def test_kernel_gradient_matches_numerical():
    """Autodiff gradient of the batched kernel w.r.t. (mu, L) ≈ finite
    differences of the NumPy loop, where Sigma = L L^T."""
    X = _make_data(seed=3, n=180, p=5)
    obj = GPUObjectiveFP64(X, device="cpu")
    p = obj.n_vars
    ct = to_torch(build_batched_constants(obj.patterns, obj.n_vars),
                  torch, "cpu", torch.float64)

    rng = np.random.default_rng(11)
    mu0 = rng.standard_normal(p)
    L0 = np.tril(rng.standard_normal((p, p))) + p * np.eye(p)

    mu_t = torch.tensor(mu0, requires_grad=True)
    L_t = torch.tensor(L0, requires_grad=True)
    sigma_t = L_t @ L_t.T
    loss = batched_neg2_loglik(torch, mu_t, sigma_t, ct, EPS)
    loss.backward()

    def f(mu, L):
        return _numpy_loop(obj.patterns, mu, L @ L.T)

    h = 1e-6
    g_mu = np.zeros(p)
    for i in range(p):
        a, b = mu0.copy(), mu0.copy()
        a[i] += h
        b[i] -= h
        g_mu[i] = (f(a, L0) - f(b, L0)) / (2 * h)
    np.testing.assert_allclose(mu_t.grad.numpy(), g_mu, rtol=1e-4, atol=1e-3)

    # check a few L entries
    for (i, j) in [(0, 0), (3, 1), (4, 4)]:
        a, b = L0.copy(), L0.copy()
        a[i, j] += h
        b[i, j] -= h
        g = (f(mu0, a) - f(mu0, b)) / (2 * h)
        assert np.isclose(L_t.grad.numpy()[i, j], g, rtol=1e-4, atol=1e-3)
