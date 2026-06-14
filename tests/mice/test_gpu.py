"""
GPU backend tests for MICE.

The GPU path is validated against the trusted CPU implementation (which is in
turn validated against R), not against R directly — the two-tier strategy used
across pystatistics. GPU results are stochastic and run in FP32, so we check
distributional agreement with CPU at the GPU/FP32 tolerance, plus the structural
invariants (no NaN, observed values preserved, PMM donor property).

All tests skip when no CUDA GPU is present.
"""

import numpy as np
import pytest

from pystatistics.mice import datasets, mice


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _cuda_available(), reason="No CUDA GPU available")


@pytest.fixture(scope="module")
def miss():
    complete = datasets.make_gaussian_complete(200, seed=1)
    return datasets.make_mcar(complete, 0.25, seed=2)


class TestGpuBasics:
    def test_runs_on_gpu(self, miss):
        sol = mice(miss, m=4, maxit=5, seed=0, backend="gpu")
        assert "gpu" in sol.backend_name
        assert sol.info["device"] == "cuda"
        assert sol.info["precision"] == "fp32"

    def test_completed_no_nan(self, miss):
        sol = mice(miss, m=4, maxit=5, seed=0, backend="gpu")
        for d in sol.completed_datasets():
            assert not np.isnan(d).any()
            assert d.shape == miss.shape

    def test_observed_preserved(self, miss):
        sol = mice(miss, m=3, maxit=4, seed=0, backend="gpu")
        observed = ~np.isnan(miss)
        for d in sol.completed_datasets():
            # FP32 round-trip: observed values preserved to single precision.
            np.testing.assert_allclose(d[observed], miss[observed], rtol=1e-5, atol=1e-5)

    def test_auto_selects_gpu_on_cuda(self, miss):
        sol = mice(miss, m=3, maxit=3, seed=0, backend="auto")
        assert "gpu" in sol.backend_name


class TestGpuReproducibility:
    def test_same_seed_reproducible(self, miss):
        a = mice(miss, m=4, maxit=5, method="pmm", seed=11, backend="gpu")
        b = mice(miss, m=4, maxit=5, method="pmm", seed=11, backend="gpu")
        for da, db in zip(a.completed_datasets(), b.completed_datasets()):
            # Same seed + same device: identical up to FP32 kernel determinism.
            np.testing.assert_allclose(da, db, rtol=1e-5, atol=1e-6)

    def test_different_seed_differs(self, miss):
        a = mice(miss, m=4, maxit=5, method="norm", seed=1, backend="gpu")
        b = mice(miss, m=4, maxit=5, method="norm", seed=2, backend="gpu")
        assert any(
            not np.allclose(da, db)
            for da, db in zip(a.completed_datasets(), b.completed_datasets())
        )


class TestGpuPmmDonorProperty:
    def test_fp32_imputes_near_observed(self, miss):
        sol = mice(miss, m=5, maxit=5, method="pmm", seed=0, backend="gpu")
        for j in sol.incomplete_columns:
            observed = miss[~np.isnan(miss[:, j]), j]
            imp = sol.imputations(j).ravel()
            dist = np.abs(imp[:, None] - observed[None, :]).min(axis=1)
            # Every imputed value is a donor copy, exact up to FP32 rounding.
            assert dist.max() < 1e-4

    def test_fp64_imputes_exact_observed(self, miss):
        sol = mice(miss, m=5, maxit=5, method="pmm", seed=0, backend="gpu", use_fp64=True)
        for j in sol.incomplete_columns:
            observed = set(np.round(miss[~np.isnan(miss[:, j]), j], 10))
            for v in sol.imputations(j).ravel():
                assert round(float(v), 10) in observed


@pytest.mark.parametrize("method", ["pmm", "norm"])
class TestGpuMatchesCpu:
    def test_imputed_distribution_matches_cpu(self, miss, method):
        # Independent stochastic runs (different RNG, FP32 vs FP64) — agreement
        # is distributional, at the GPU/FP32 Monte-Carlo scale.
        m = 30
        cpu = mice(miss, m=m, maxit=10, method=method, seed=100, backend="cpu")
        gpu = mice(miss, m=m, maxit=10, method=method, seed=100, backend="gpu")
        for j in gpu.incomplete_columns:
            cc = cpu.imputations(j).ravel()
            cg = gpu.imputations(j).ravel()
            assert abs(cg.mean() - cc.mean()) < 0.15, f"col {j} mean drift"
            assert abs(cg.std() - cc.std()) < 0.15, f"col {j} sd drift"

    def test_pooled_regression_matches_cpu(self, miss, method):
        from pystatistics.mice import pool

        def pooled(sol):
            est, var = [], []
            for d in sol.completed_datasets():
                X = np.column_stack([np.ones(len(d)), d[:, 1], d[:, 2]])
                y = d[:, 0]
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                resid = y - X @ beta
                s2 = resid @ resid / (len(d) - 3)
                est.append(beta)
                var.append(np.diag(s2 * np.linalg.inv(X.T @ X)))
            return pool(np.array(est), np.array(var), dfcom=len(miss) - 3)

        m = 30
        cpu = pooled(mice(miss, m=m, maxit=10, method=method, seed=7, backend="cpu"))
        gpu = pooled(mice(miss, m=m, maxit=10, method=method, seed=7, backend="gpu"))
        np.testing.assert_allclose(
            np.asarray(gpu.estimate), np.asarray(cpu.estimate), atol=0.1
        )


class TestGpuFp64:
    def test_fp64_runs(self, miss):
        sol = mice(miss, m=3, maxit=4, seed=0, backend="gpu", use_fp64=True)
        assert sol.info["precision"] == "fp64"
        for d in sol.completed_datasets():
            assert not np.isnan(d).any()

    def test_fp64_closer_to_cpu_observed(self, miss):
        # FP64 preserves observed values to (near) machine precision.
        sol = mice(miss, m=2, maxit=3, seed=0, backend="gpu", use_fp64=True)
        observed = ~np.isnan(miss)
        for d in sol.completed_datasets():
            np.testing.assert_allclose(d[observed], miss[observed], rtol=1e-12, atol=1e-12)
