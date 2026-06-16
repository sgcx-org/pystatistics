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
    _tri_inv_blocked,
    chunk_bounds,
    auto_chunk_size,
    accumulate_gradient,
    analytic_gradient,
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


@pytest.mark.parametrize("v", [1, 2, 3, 7, 16, 50])
@pytest.mark.parametrize("rho", [0.0, 0.9, 0.99])
def test_blocked_triangular_inverse_exact(v, rho):
    """The matmul-only blocked inverse equals the true inverse, including for
    highly-correlated (ill-conditioned) Cholesky factors where a Neumann series
    would diverge."""
    idx = torch.arange(v)
    corr = rho ** (idx[:, None] - idx[None, :]).abs().to(torch.float64)
    Sig = corr.expand(64, v, v) + 1e-2 * torch.eye(v, dtype=torch.float64)
    L = torch.linalg.cholesky(Sig)
    W = _tri_inv_blocked(torch, L)
    eye = torch.eye(v, dtype=torch.float64).expand(64, v, v)
    # W must be the true inverse: W @ L = I
    assert torch.allclose(W @ L, eye, atol=1e-8), (W @ L - eye).abs().max().item()


def test_blocked_inverse_autodiff_matches_solve():
    """tr(Sigma^-1 M) via the blocked inverse has the same gradient as via
    triangular solve (both on CPU; the blocked path is forced here)."""
    P, v = 1500, 12
    rng = np.random.default_rng(5)
    A = rng.standard_normal((P, v, v))
    Sig = torch.tensor(A @ A.transpose(0, 2, 1) + v * np.eye(v))
    M = torch.tensor((lambda b: b @ b.transpose(0, 2, 1))(rng.standard_normal((P, v, v))))
    L0 = torch.linalg.cholesky(Sig)

    def via_solve(L):
        Y = torch.linalg.solve_triangular(L, M, upper=False)
        X = torch.linalg.solve_triangular(L.transpose(-1, -2), Y, upper=True)
        return torch.diagonal(X, dim1=-2, dim2=-1).sum(-1)

    def via_blocked(L):
        W = _tri_inv_blocked(torch, L)
        return ((W.transpose(-1, -2) @ W) * M).sum((-2, -1))

    g = {}
    for nm, fn in [("solve", via_solve), ("blocked", via_blocked)]:
        L = L0.clone().requires_grad_(True)
        fn(L).sum().backward()
        g[nm] = L.grad
    rel = (g["blocked"] - g["solve"]).abs().max() / g["solve"].abs().max()
    assert rel < 1e-6, rel.item()


def test_chunk_bounds_and_auto_size():
    assert chunk_bounds(10, None) == [(0, 10)]
    assert chunk_bounds(10, 0) == [(0, 10)]
    assert chunk_bounds(10, 100) == [(0, 10)]
    assert chunk_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)]
    # auto size shrinks with v and is always positive
    assert auto_chunk_size(100, 4) >= 1
    assert auto_chunk_size(25, 4) > auto_chunk_size(200, 4)


def test_chunked_matches_unchunked_value_and_gradient():
    """Pattern-chunking must not change the objective or its gradient."""
    X = _make_data(seed=4, n=220, p=8)
    big = GPUObjectiveFP64(X, device="cpu", chunk_size=10 ** 9)   # one chunk
    small = GPUObjectiveFP64(X, device="cpu", chunk_size=2)       # many chunks
    assert small.chunk_size == 2 and big.chunk_size >= big.n_patterns
    for scale in (1.0, 0.9):
        theta = big.get_initial_parameters() * scale
        assert np.isclose(big.compute_objective(theta),
                          small.compute_objective(theta), rtol=1e-10), scale
        np.testing.assert_allclose(big.compute_gradient(theta),
                                   small.compute_gradient(theta),
                                   rtol=1e-8, atol=1e-10)


def test_analytic_gradient_matches_autodiff():
    """The closed-form matrix gradient equals reverse-mode autodiff (which it
    replaces to avoid Metal's slow Cholesky backward)."""
    X = _make_data(seed=6, n=240, p=7)
    obj = GPUObjectiveFP64(X, device="cpu")
    theta = obj.get_initial_parameters()
    for scale in (1.0, 0.9, 1.15):
        th = theta * scale
        t1 = torch.tensor(th, dtype=torch.float64, requires_grad=True)
        g_auto = accumulate_gradient(torch, t1, obj._unpack_gpu, obj._consts,
                                     obj.eps, obj.chunk_size)[0].detach().numpy()
        t2 = torch.tensor(th, dtype=torch.float64, requires_grad=True)
        g_ana = analytic_gradient(torch, t2, obj._unpack_gpu, obj._consts,
                                  obj.eps, obj.chunk_size).detach().numpy()
        np.testing.assert_allclose(g_ana, g_auto, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("method", ["auto", "solve", "blocked"])
def test_method_toggle_is_result_invariant(method):
    """The trace/inverse ``method`` toggle ('auto'|'solve'|'blocked') selects a
    computational path only — it must not change the result. Each forced method
    yields the same objective value and the same gradient as the default path
    and as reverse-mode autodiff. Exercised on CPU, where both the blocked
    inverse and the triangular-solve path are valid (the solve path uses
    ``solve_triangular`` against the identity, since ``cholesky_inverse`` is
    unavailable on MPS)."""
    X = _make_data(seed=8, n=240, p=7)
    obj = GPUObjectiveFP64(X, device="cpu")
    theta = obj.get_initial_parameters()

    # Objective value is identical across methods.
    th = torch.tensor(theta, dtype=torch.float64)
    mu, sigma = obj._unpack_gpu(th)
    val_default = float(batched_neg2_loglik(torch, mu, sigma, obj._consts, obj.eps))
    val_method = float(batched_neg2_loglik(torch, mu, sigma, obj._consts,
                                           obj.eps, method))
    assert np.isclose(val_default, val_method, rtol=1e-10), (method, val_default,
                                                             val_method)

    # Analytic gradient under the forced method equals reverse-mode autodiff.
    t1 = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    g_auto = accumulate_gradient(torch, t1, obj._unpack_gpu, obj._consts,
                                 obj.eps, obj.chunk_size)[0].detach().numpy()
    t2 = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    g_ana = analytic_gradient(torch, t2, obj._unpack_gpu, obj._consts,
                              obj.eps, obj.chunk_size, method=method).detach().numpy()
    np.testing.assert_allclose(g_ana, g_auto, rtol=1e-8, atol=1e-9)


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
