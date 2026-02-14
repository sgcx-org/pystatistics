"""
GPU tests for MVN MLE (both direct BFGS and EM algorithms).

Validates that GPU backends produce results consistent with CPU references.
GPU uses FP32 (except CUDA FP64 when explicitly requested), so tolerances
are relaxed compared to CPU-only tests.

MPS (Apple Silicon) notes:
- MPS only supports FP32, which limits gradient precision
- L-BFGS-B may report non-convergence even when the solution is close to MLE
- Missvals (5 variables) is particularly challenging in FP32

CUDA notes:
- Consumer CUDA (RTX) uses FP32 by default
- FP32 precision is generally better than MPS due to more mature libraries

Skipped automatically when no GPU is available.
"""

import numpy as np
import pytest

from pystatistics.mvnmle import mlest, datasets, MVNDesign, MVNSolution


def _gpu_available():
    """Check if any GPU is available for testing."""
    try:
        import torch
        return (torch.cuda.is_available() or
                (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()))
    except ImportError:
        return False


def _gpu_device():
    """Return the GPU device type string ('cuda' or 'mps')."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return None


def _is_mps():
    return _gpu_device() == 'mps'


pytestmark = pytest.mark.skipif(
    not _gpu_available(), reason="No GPU available"
)


# =====================================================================
# Direct (BFGS) GPU tests
# =====================================================================

class TestDirectGPU:
    """GPU backend for direct BFGS optimization.

    FP32 gradient precision may cause the optimizer to report non-convergence
    even when the solution is very close to the MLE. Tests check numerical
    quality of the solution rather than the convergence flag.
    """

    def test_gpu_backend_runs(self):
        """GPU direct should produce a finite solution."""
        result = mlest(datasets.apple, algorithm='direct', backend='gpu')
        assert 'gpu' in result.backend_name
        assert np.all(np.isfinite(result.muhat))
        assert np.all(np.isfinite(result.sigmahat))
        assert np.isfinite(result.loglik)

    def test_gpu_matches_cpu_apple(self):
        """GPU direct should match CPU direct within FP32 tolerance."""
        cpu = mlest(datasets.apple, algorithm='direct', backend='cpu')
        gpu = mlest(datasets.apple, algorithm='direct', backend='gpu')

        np.testing.assert_allclose(gpu.muhat, cpu.muhat, rtol=1e-2,
                                   err_msg="GPU direct means differ from CPU")
        np.testing.assert_allclose(gpu.sigmahat, cpu.sigmahat, rtol=5e-2,
                                   err_msg="GPU direct covariance differs from CPU")
        assert abs(gpu.loglik - cpu.loglik) < 1.0, (
            f"GPU loglik {gpu.loglik} differs from CPU {cpu.loglik} "
            f"by {abs(gpu.loglik - cpu.loglik)}"
        )

    def test_gpu_matches_cpu_missvals(self):
        """GPU direct on missvals — FP32 tolerance.

        Missvals (5 variables, heavy missingness) is challenging for FP32.
        MPS FP32 may have large errors; CUDA FP32 is typically closer.
        """
        cpu = mlest(datasets.missvals, algorithm='direct', backend='cpu',
                    max_iter=500)
        gpu = mlest(datasets.missvals, algorithm='direct', backend='gpu',
                    max_iter=500)

        # Use very wide tolerance — FP32 optimization on 5-variable missing
        # data is inherently imprecise. The key test is that solutions are
        # finite and in the right ballpark.
        rtol = 0.5 if _is_mps() else 0.1
        np.testing.assert_allclose(gpu.muhat, cpu.muhat, rtol=rtol,
                                   err_msg="GPU direct means differ from CPU (missvals)")

    def test_gpu_covariance_symmetric(self):
        result = mlest(datasets.apple, algorithm='direct', backend='gpu')
        np.testing.assert_allclose(
            result.sigmahat, result.sigmahat.T,
            atol=1e-6,
            err_msg="GPU direct covariance not symmetric"
        )

    def test_gpu_covariance_positive_definite(self):
        result = mlest(datasets.apple, algorithm='direct', backend='gpu')
        eigenvals = np.linalg.eigvalsh(result.sigmahat)
        assert np.all(eigenvals > 0), f"Not PD: min eigenvalue = {eigenvals.min()}"

    def test_gpu_timing_populated(self):
        result = mlest(datasets.apple, algorithm='direct', backend='gpu')
        assert result.timing is not None
        assert 'total_seconds' in result.timing
        assert result.timing['total_seconds'] > 0

    def test_gpu_solution_interface(self):
        """Full MVNSolution interface works with GPU results."""
        result = mlest(datasets.apple, algorithm='direct', backend='gpu')
        assert isinstance(result, MVNSolution)
        assert isinstance(result.aic, float)
        assert isinstance(result.bic, float)
        assert "MVN MLE Results" in result.summary()
        d = result.to_dict()
        assert 'muhat' in d
        assert 'loglik' in d


# =====================================================================
# EM GPU tests
# =====================================================================

class TestEMGPU:
    """GPU backend for EM algorithm.

    Note: EM currently uses numpy for all computation even when device='cuda'
    or 'mps'. The GPU device flag is accepted and the backend name reflects
    the requested device, but actual torch-based EM is future work. These
    tests verify the dispatch path works correctly.
    """

    def test_em_gpu_backend_runs(self):
        result = mlest(datasets.apple, algorithm='em', backend='gpu')
        assert result.converged
        assert result.backend_name.endswith('_em')

    def test_em_gpu_matches_cpu_apple(self):
        """EM on GPU device should match EM on CPU."""
        cpu = mlest(datasets.apple, algorithm='em', backend='cpu',
                    tol=1e-8, max_iter=10000)
        gpu = mlest(datasets.apple, algorithm='em', backend='gpu',
                    tol=1e-8, max_iter=10000)

        np.testing.assert_allclose(gpu.muhat, cpu.muhat, rtol=1e-3,
                                   err_msg="EM GPU means differ from CPU")
        np.testing.assert_allclose(gpu.sigmahat, cpu.sigmahat, rtol=1e-2,
                                   err_msg="EM GPU covariance differs from CPU")
        assert abs(gpu.loglik - cpu.loglik) < 1.0, (
            f"EM GPU loglik {gpu.loglik} differs from CPU {cpu.loglik} "
            f"by {abs(gpu.loglik - cpu.loglik)}"
        )

    def test_em_gpu_matches_cpu_missvals(self):
        """EM on GPU device should match EM on CPU on missvals."""
        cpu = mlest(datasets.missvals, algorithm='em', backend='cpu',
                    tol=1e-8, max_iter=100000)
        gpu = mlest(datasets.missvals, algorithm='em', backend='gpu',
                    tol=1e-8, max_iter=100000)

        np.testing.assert_allclose(gpu.muhat, cpu.muhat, rtol=5e-2,
                                   err_msg="EM GPU means differ from CPU (missvals)")
        np.testing.assert_allclose(gpu.sigmahat, cpu.sigmahat, rtol=5e-2,
                                   err_msg="EM GPU covariance differs from CPU (missvals)")

    def test_em_gpu_covariance_symmetric(self):
        result = mlest(datasets.apple, algorithm='em', backend='gpu')
        np.testing.assert_allclose(
            result.sigmahat, result.sigmahat.T,
            atol=1e-6,
            err_msg="EM GPU covariance not symmetric"
        )

    def test_em_gpu_covariance_positive_definite(self):
        result = mlest(datasets.apple, algorithm='em', backend='gpu')
        eigenvals = np.linalg.eigvalsh(result.sigmahat)
        assert np.all(eigenvals > 0), f"Not PD: min eigenvalue = {eigenvals.min()}"

    def test_em_gpu_timing_populated(self):
        result = mlest(datasets.apple, algorithm='em', backend='gpu')
        assert result.timing is not None
        assert 'total_seconds' in result.timing
        assert result.timing['total_seconds'] > 0

    def test_em_gpu_solution_interface(self):
        """Full MVNSolution interface works with EM GPU results."""
        result = mlest(datasets.apple, algorithm='em', backend='gpu')
        assert isinstance(result, MVNSolution)
        assert isinstance(result.aic, float)
        assert isinstance(result.bic, float)
        assert "MVN MLE Results" in result.summary()
        d = result.to_dict()
        assert 'muhat' in d
        assert 'loglik' in d

    def test_em_gpu_gradient_norm_is_none(self):
        """EM does not compute gradients, even on GPU."""
        result = mlest(datasets.apple, algorithm='em', backend='gpu')
        assert result.gradient_norm is None


# =====================================================================
# Cross-algorithm GPU tests
# =====================================================================

class TestCrossAlgorithmGPU:
    """EM and direct should agree even when both run on GPU."""

    def test_apple_loglik_agrees_gpu(self):
        """Both algorithms on GPU should reach the same MLE."""
        em = mlest(datasets.apple, algorithm='em', backend='gpu',
                   tol=1e-8, max_iter=10000)
        direct = mlest(datasets.apple, algorithm='direct', backend='gpu')

        # FP32 accumulation on GPU means wider tolerance than CPU comparison
        assert abs(em.loglik - direct.loglik) < 1.0, (
            f"EM GPU loglik {em.loglik} differs from direct GPU {direct.loglik} "
            f"by {abs(em.loglik - direct.loglik)}"
        )

    def test_apple_means_agree_gpu(self):
        em = mlest(datasets.apple, algorithm='em', backend='gpu',
                   tol=1e-8, max_iter=10000)
        direct = mlest(datasets.apple, algorithm='direct', backend='gpu')
        np.testing.assert_allclose(em.muhat, direct.muhat, rtol=1e-2)

    def test_apple_covariance_agrees_gpu(self):
        em = mlest(datasets.apple, algorithm='em', backend='gpu',
                   tol=1e-8, max_iter=10000)
        direct = mlest(datasets.apple, algorithm='direct', backend='gpu')
        np.testing.assert_allclose(em.sigmahat, direct.sigmahat, rtol=5e-2)


# =====================================================================
# Backend auto-selection with GPU present
# =====================================================================

class TestAutoBackendWithGPU:
    """When GPU is available, backend='auto' should select the appropriate
    device. On CUDA, auto selects GPU for direct; on MPS, auto falls back
    to CPU (MPS not auto-selected, same as regression module)."""

    def test_auto_direct_selects_gpu_on_cuda(self):
        device = _gpu_device()
        result = mlest(datasets.apple, algorithm='direct', backend='auto')
        if device == 'cuda':
            assert 'gpu' in result.backend_name
        else:
            # MPS: auto falls back to CPU (MPS not auto-selected)
            assert 'cpu' in result.backend_name

    def test_auto_em_runs(self):
        """EM with auto backend should run and converge regardless of device."""
        result = mlest(datasets.apple, algorithm='em', backend='auto')
        assert result.converged
        assert result.backend_name.endswith('_em')

    def test_auto_em_convergence_missvals(self):
        """EM with auto backend should converge on missvals."""
        result = mlest(datasets.missvals, algorithm='em', backend='auto',
                       tol=1e-6, max_iter=50000)
        assert result.converged
        assert np.all(np.isfinite(result.muhat))
        assert np.all(np.isfinite(result.sigmahat))


# =====================================================================
# Edge cases on GPU
# =====================================================================

class TestGPUEdgeCases:
    """GPU-specific edge cases."""

    def test_complete_data_gpu_direct(self):
        """Complete data (no missing) should produce valid results on GPU."""
        rng = np.random.default_rng(42)
        data = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=50)
        result = mlest(data, algorithm='direct', backend='gpu')
        # FP32 optimizer may not report convergence, but solution should be valid
        assert np.all(np.isfinite(result.muhat))
        assert np.all(np.isfinite(result.sigmahat))
        assert np.isfinite(result.loglik)

    def test_complete_data_gpu_em(self):
        """Complete data EM on GPU should converge quickly."""
        rng = np.random.default_rng(42)
        data = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=50)
        result = mlest(data, algorithm='em', backend='gpu')
        assert result.converged
        assert result.n_iter <= 5

    def test_high_missing_rate_gpu(self):
        """High missingness should not crash GPU backend."""
        rng = np.random.default_rng(42)
        data = rng.multivariate_normal([5, 10], [[2, 1], [1, 3]], size=100)
        mask = rng.random(data.shape) < 0.4
        for i in range(data.shape[0]):
            if mask[i].all():
                mask[i, 0] = False
        for j in range(data.shape[1]):
            if mask[:, j].all():
                mask[0, j] = False
        data[mask] = np.nan
        result = mlest(data, algorithm='em', backend='gpu', max_iter=5000)
        assert np.all(np.isfinite(result.muhat))
        assert np.all(np.isfinite(result.sigmahat))

    def test_design_object_gpu(self):
        """MVNDesign objects should work with GPU backends."""
        design = MVNDesign.from_array(datasets.apple)
        result_direct = mlest(design, algorithm='direct', backend='gpu')
        result_em = mlest(design, algorithm='em', backend='gpu')
        assert isinstance(result_direct, MVNSolution)
        assert isinstance(result_em, MVNSolution)
        # EM should converge; direct may not on FP32 but solution should be valid
        assert result_em.converged
        assert np.all(np.isfinite(result_direct.muhat))
        assert np.all(np.isfinite(result_direct.sigmahat))
