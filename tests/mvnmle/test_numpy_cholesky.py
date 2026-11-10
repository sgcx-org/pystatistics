"""
Tests for the NumPy forward-Cholesky direct MVN MLE path.

Covers the kernel (``_objectives/_batched_cholesky_np.py``), the objective
(``_objectives/numpy_fp64.py``) and the backend
(``backends/numpy_cholesky.py``):

- objective and analytic gradient agree with the PyTorch FP64 forward-Cholesky
  objective to near machine precision (skipped without PyTorch),
- the analytic gradient agrees with central finite differences,
- end-to-end fits agree with the PyTorch FP64 CPU fit, the R references and
  the closed-form complete-data MLE,
- a PyTorch-less install routes the default direct path here,
- invalid input and non-positive-definite blocks fail loudly.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pystatistics.core.exceptions import ValidationError
from pystatistics.mvnmle import datasets, mlest
from pystatistics.mvnmle._objectives._batched_cholesky_np import (
    _batched_cholesky_with_ridge,
    _batched_tril_inverse,
    unpack_cholesky_np,
)
from pystatistics.mvnmle._objectives.numpy_fp64 import NumpyObjectiveFP64
from pystatistics.mvnmle._objectives.parameterizations import (
    CholeskyParameterization,
)
from pystatistics.mvnmle.backends.numpy_cholesky import NumpyCholeskyMLEBackend
from pystatistics.mvnmle.design import MVNDesign

REFERENCES = Path(__file__).parent / "references"
NUMPY_BACKEND_NAME = 'cpu_cholesky_numpy_fp64'


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


needs_torch = pytest.mark.skipif(not _torch_available(), reason="needs PyTorch")


def make_problem(n, p, missing_rate, seed):
    """Seeded MVN sample with MCAR missingness (no all-missing rows)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, p))
    sigma = A @ A.T + p * np.eye(p)
    X = rng.multivariate_normal(rng.standard_normal(p), sigma, size=n)
    X[rng.random((n, p)) < missing_rate] = np.nan
    return X[~np.isnan(X).all(axis=1)]


def perturbed_theta(objective, seed):
    """A point away from the start, so the comparison is not at a special
    (e.g. near-stationary) parameter value."""
    rng = np.random.default_rng(seed)
    theta0 = objective.get_initial_parameters()
    return theta0 + 0.05 * rng.standard_normal(theta0.shape)


def fit_numpy(data, **kwargs):
    return NumpyCholeskyMLEBackend().solve(MVNDesign.from_array(data), **kwargs)


# (n, p, missing_rate, seed): several missingness patterns each.
PROBLEMS = [
    (60, 2, 0.25, 1),
    (200, 5, 0.20, 2),
    (500, 8, 0.30, 3),
    (2000, 12, 0.15, 4),
]


class TestKernelPieces:
    def test_unpack_matches_canonical_parameterization(self):
        p = 5
        rng = np.random.default_rng(0)
        param = CholeskyParameterization(p)
        theta = rng.standard_normal(param.n_params)
        mu_ref, sigma_ref = param.unpack(theta)
        mu, L = unpack_cholesky_np(theta, p)
        np.testing.assert_array_equal(mu, mu_ref)
        assert np.all(np.triu(L, k=1) == 0.0)
        np.testing.assert_allclose(L @ L.T, sigma_ref, rtol=1e-14, atol=1e-14)

    def test_batched_tril_inverse(self):
        rng = np.random.default_rng(1)
        L = np.tril(rng.standard_normal((7, 6, 6)))
        L[:, np.arange(6), np.arange(6)] = 1.0 + rng.random((7, 6))
        W = _batched_tril_inverse(L)
        assert np.all(np.triu(W, k=1) == 0.0)
        np.testing.assert_allclose(
            W @ L, np.broadcast_to(np.eye(6), (7, 6, 6)), atol=1e-12)

    def test_batched_tril_inverse_single_variable(self):
        W = _batched_tril_inverse(np.array([[[2.0]], [[4.0]]]))
        np.testing.assert_array_equal(W, [[[0.5]], [[0.25]]])

    def test_ridge_rescues_marginally_indefinite_batch(self):
        # One block is singular to working precision; a ridge of order eps
        # makes it factorisable, and the well-conditioned block is unharmed.
        sig = np.stack([np.eye(2), np.array([[1.0, 1.0], [1.0, 1.0]])])
        L = _batched_cholesky_with_ridge(sig, eps=1e-12)
        assert np.all(np.isfinite(L))
        np.testing.assert_allclose(L[0], np.eye(2), atol=1e-9)

    def test_indefinite_batch_raises(self):
        sig = np.array([[[1.0, 0.0], [0.0, -1.0]]])
        with pytest.raises(np.linalg.LinAlgError, match="ridge escalation"):
            _batched_cholesky_with_ridge(sig, eps=1e-12)


class TestAgreementWithTorch:
    """Same function as the PyTorch FP64 forward-Cholesky objective."""

    @needs_torch
    @pytest.mark.parametrize('n,p,missing_rate,seed', PROBLEMS)
    def test_value_and_gradient(self, n, p, missing_rate, seed):
        from pystatistics.mvnmle._objectives.gpu_fp64 import GPUObjectiveFP64

        X = make_problem(n, p, missing_rate, seed)
        obj_np = NumpyObjectiveFP64(X)
        obj_t = GPUObjectiveFP64(X, device='cpu')
        assert obj_np.n_patterns > 2

        np.testing.assert_array_equal(
            obj_np.get_initial_parameters(), obj_t.get_initial_parameters())

        theta = perturbed_theta(obj_np, seed)
        f_np, g_np = obj_np.compute_value_and_gradient(theta)
        f_t, g_t = obj_t.compute_value_and_gradient(theta)

        assert abs(f_np - f_t) <= 1e-12 * abs(f_t)
        assert abs(obj_np.compute_objective(theta) - f_t) <= 1e-12 * abs(f_t)
        assert np.max(np.abs(g_np - g_t)) <= 1e-11 * np.max(np.abs(g_t))

    @needs_torch
    @pytest.mark.parametrize('n,p,missing_rate,seed', PROBLEMS)
    def test_end_to_end_fit(self, n, p, missing_rate, seed):
        from pystatistics.mvnmle.backends.gpu import DirectMLEBackend

        X = make_problem(n, p, missing_rate, seed)
        design = MVNDesign.from_array(X)
        res_np = NumpyCholeskyMLEBackend().solve(design)
        res_t = DirectMLEBackend(device='cpu', use_fp64=True).solve(design)

        assert res_np.params.converged and res_t.params.converged
        assert abs(res_np.params.loglik - res_t.params.loglik) <= (
            1e-9 * abs(res_t.params.loglik))
        np.testing.assert_allclose(
            res_np.params.muhat, res_t.params.muhat, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            res_np.params.sigmahat, res_t.params.sigmahat,
            rtol=1e-6, atol=1e-6)

    @needs_torch
    def test_result_metadata_matches_torch_path(self):
        from pystatistics.mvnmle.backends.gpu import DirectMLEBackend

        design = MVNDesign.from_array(datasets.apple)
        res_np = NumpyCholeskyMLEBackend().solve(design)
        res_t = DirectMLEBackend(device='cpu', use_fp64=True).solve(design)

        assert set(res_np.info) == set(res_t.info)
        for key in ('method', 'parameterization', 'device', 'precision'):
            assert res_np.info[key] == res_t.info[key]
        assert set(res_np.timing) == set(res_t.timing)


class TestGradient:
    @pytest.mark.parametrize('n,p,missing_rate,seed', PROBLEMS[:3])
    def test_matches_central_finite_differences(self, n, p, missing_rate, seed):
        obj = NumpyObjectiveFP64(make_problem(n, p, missing_rate, seed))
        theta = perturbed_theta(obj, seed)
        grad = obj.compute_gradient(theta)

        h = 1e-6
        fd = np.empty_like(theta)
        for i in range(theta.size):
            step = np.zeros_like(theta)
            step[i] = h
            fd[i] = (obj.compute_objective(theta + step)
                     - obj.compute_objective(theta - step)) / (2 * h)

        assert np.max(np.abs(grad - fd)) <= 1e-6 * np.max(np.abs(grad))

    def test_chunking_does_not_change_the_result(self):
        X = make_problem(500, 8, 0.30, 3)
        whole = NumpyObjectiveFP64(X)
        chunked = NumpyObjectiveFP64(X, chunk_size=7)
        assert whole.n_patterns > 7
        theta = perturbed_theta(whole, 0)

        f_w, g_w = whole.compute_value_and_gradient(theta)
        f_c, g_c = chunked.compute_value_and_gradient(theta)
        assert abs(f_w - f_c) <= 1e-12 * abs(f_w)
        np.testing.assert_allclose(g_c, g_w, rtol=1e-10, atol=1e-10)
        assert abs(chunked.compute_objective(theta) - f_w) <= 1e-12 * abs(f_w)

    def test_gradient_vanishes_at_complete_data_mle(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((300, 4)) @ rng.standard_normal((4, 4))
        obj = NumpyObjectiveFP64(X)
        mu = X.mean(axis=0)
        sigma = np.cov(X, rowvar=False, bias=True)
        theta = obj.parameterization.pack(mu, sigma)
        grad = obj.compute_gradient(theta)
        assert np.max(np.abs(grad)) / obj.n_observed_scalars < 1e-9


class TestFit:
    def test_complete_data_recovers_closed_form_mle(self):
        rng = np.random.default_rng(6)
        X = rng.standard_normal((400, 4)) @ rng.standard_normal((4, 4)) + 3.0
        res = fit_numpy(X)
        assert res.params.converged
        np.testing.assert_allclose(res.params.muhat, X.mean(axis=0), atol=1e-4)
        np.testing.assert_allclose(
            res.params.sigmahat, np.cov(X, rowvar=False, bias=True),
            rtol=1e-3, atol=1e-4)

    def test_single_variable(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal((50, 1)) * 2.0 + 1.0
        res = fit_numpy(x)
        assert res.params.converged
        np.testing.assert_allclose(res.params.muhat, x.mean(axis=0), atol=1e-5)
        np.testing.assert_allclose(
            res.params.sigmahat, [[x.var()]], rtol=1e-4)

    def test_apple_matches_r_reference(self):
        with open(REFERENCES / "apple_reference.json") as f:
            ref = json.load(f)
        res = fit_numpy(datasets.apple)
        assert res.params.converged
        assert abs(res.params.loglik - ref['loglik']) < 1e-7
        np.testing.assert_allclose(res.params.muhat, ref['muhat'], rtol=1e-3)
        np.testing.assert_allclose(
            res.params.sigmahat, ref['sigmahat'], rtol=1e-3)

    def test_missvals_matches_r_reference(self):
        # missvals (n=13, p=5) is a hard, tiny-n problem: the estimates reach
        # R's optimum but the convergence flag is platform-dependent, so only
        # the estimates are asserted (as for the other direct paths).
        with open(REFERENCES / "missvals_reference.json") as f:
            ref = json.load(f)
        res = fit_numpy(datasets.missvals, max_iter=2000)
        assert abs(res.params.loglik - ref['loglik']) < 1e-5
        np.testing.assert_allclose(
            res.params.muhat, ref['muhat'], rtol=1e-3, atol=1e-3)

    @needs_torch
    @pytest.mark.parametrize('name,max_iter', [('apple', 100),
                                               ('missvals', 2000)])
    def test_bundled_datasets_match_torch_path(self, name, max_iter):
        data = getattr(datasets, name)
        res = fit_numpy(data, max_iter=max_iter)
        fast = mlest(data, backend='cpu', max_iter=max_iter)
        assert fast.backend_name == 'cpu_cholesky_fp64'
        assert abs(res.params.loglik - fast.loglik) < 1e-6

    def test_nonconvergence_is_reported(self):
        res = fit_numpy(make_problem(200, 5, 0.20, 2), max_iter=1)
        assert not res.params.converged
        assert any('did not converge' in w for w in res.warnings)

    def test_scipy_method_is_forwarded(self):
        res = fit_numpy(datasets.apple, method='L-BFGS-B')
        assert res.info['method'] == 'L-BFGS-B'


class TestTorchFreeRouting:
    """Without PyTorch the default direct path is the NumPy forward-Cholesky
    backend, and the switch is announced."""

    @pytest.mark.parametrize('backend', [None, 'cpu', 'auto'])
    def test_default_routes_to_numpy_backend(self, monkeypatch, backend):
        monkeypatch.setitem(sys.modules, 'torch', None)  # import -> ImportError
        with pytest.warns(UserWarning, match="PyTorch is not installed"):
            result = mlest(datasets.apple, backend=backend)
        assert result.backend_name == NUMPY_BACKEND_NAME
        info = result._result.info
        assert info['parameterization'] == 'cholesky'
        assert info['precision'] == 'fp64'
        assert info['device'] == 'cpu'
        assert result.converged

    def test_estimates_match_r_without_torch(self, monkeypatch):
        with open(REFERENCES / "apple_reference.json") as f:
            ref = json.load(f)
        monkeypatch.setitem(sys.modules, 'torch', None)
        with pytest.warns(UserWarning, match="PyTorch is not installed"):
            result = mlest(datasets.apple)
        assert abs(result.loglik - ref['loglik']) < 1e-7

    def test_objective_module_does_not_import_torch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, 'torch', None)
        obj = NumpyObjectiveFP64(datasets.apple)
        value, grad = obj.compute_value_and_gradient(
            obj.get_initial_parameters())
        assert np.isfinite(value) and np.all(np.isfinite(grad))


class TestInvalidInput:
    def test_all_missing_column_raises(self):
        X = make_problem(50, 3, 0.1, 8)
        X[:, 1] = np.nan
        with pytest.raises(ValidationError, match="columns with all missing"):
            NumpyObjectiveFP64(X)

    def test_one_dimensional_input_raises(self):
        with pytest.raises(ValidationError, match="must be 2D"):
            NumpyObjectiveFP64(np.arange(5.0))
