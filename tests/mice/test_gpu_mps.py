"""GPU backend tests for MICE on Apple Silicon (MPS).

The MPS path shares the batched sweep with the CUDA path, diverging only in the
PMM donor search's insertion-rank op (``_gpu_methods._insertion_rank``: a
combined-sort merge-rank on MPS, ``searchsorted`` on CUDA). It runs FP32 only —
MPS has no float64 — so ``use_fp64=True`` is rejected at the dispatch boundary.

As with the CUDA suite, the MPS path is validated against the trusted CPU
implementation (itself validated against R), distributionally at the MPS/FP32
tolerance, plus the structural invariants. All tests skip when MPS is absent.
"""

import numpy as np
import pytest

from pystatistics.mice import datasets, mice, pool


def _mps_available() -> bool:
    try:
        import torch

        return torch.backends.mps.is_available()
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _mps_available(), reason="No MPS device available")


@pytest.fixture(scope="module")
def miss():
    complete = datasets.make_gaussian_complete(200, seed=1)
    return datasets.make_mcar(complete, 0.25, seed=2)


class TestMpsBasics:
    def test_runs_on_mps(self, miss):
        sol = mice(miss, m=4, maxit=5, seed=0, backend="gpu")
        assert "gpu" in sol.backend_name
        assert sol.info["device"] == "mps"
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
            np.testing.assert_allclose(d[observed], miss[observed], rtol=1e-5, atol=1e-5)


class TestMpsFp64Rejected:
    def test_use_fp64_rejected_on_mps(self, miss):
        # MPS has no float64 — the request must fail loud, not silently downgrade.
        with pytest.raises(ValueError, match="float64|MPS|use_fp64"):
            mice(miss, m=2, maxit=3, seed=0, backend="gpu", use_fp64=True)


class TestMpsReproducibility:
    def test_same_seed_reproducible(self, miss):
        a = mice(miss, m=4, maxit=5, method="pmm", seed=11, backend="gpu")
        b = mice(miss, m=4, maxit=5, method="pmm", seed=11, backend="gpu")
        for da, db in zip(a.completed_datasets(), b.completed_datasets()):
            np.testing.assert_allclose(da, db, rtol=1e-5, atol=1e-6)

    def test_different_seed_differs(self, miss):
        a = mice(miss, m=4, maxit=5, method="norm", seed=1, backend="gpu")
        b = mice(miss, m=4, maxit=5, method="norm", seed=2, backend="gpu")
        assert any(
            not np.allclose(da, db)
            for da, db in zip(a.completed_datasets(), b.completed_datasets())
        )


class TestMpsPmmDonorProperty:
    def test_fp32_imputes_near_observed(self, miss):
        # Every PMM imputed value is a copy of an observed value (the donor
        # search is exact), to FP32 rounding — the hard correctness invariant.
        sol = mice(miss, m=5, maxit=5, method="pmm", seed=0, backend="gpu")
        for j in sol.incomplete_columns:
            observed = miss[~np.isnan(miss[:, j]), j]
            imp = sol.imputations(j).ravel()
            dist = np.abs(imp[:, None] - observed[None, :]).min(axis=1)
            assert dist.max() < 1e-4


@pytest.mark.parametrize("method", ["pmm", "norm"])
class TestMpsMatchesCpu:
    def test_imputed_distribution_matches_cpu(self, miss, method):
        m = 30
        cpu = mice(miss, m=m, maxit=10, method=method, seed=100, backend="cpu")
        gpu = mice(miss, m=m, maxit=10, method=method, seed=100, backend="gpu")
        for j in gpu.incomplete_columns:
            cc = cpu.imputations(j).ravel()
            cg = gpu.imputations(j).ravel()
            assert abs(cg.mean() - cc.mean()) < 0.15, f"col {j} mean drift"
            assert abs(cg.std() - cc.std()) < 0.15, f"col {j} sd drift"

    def test_pooled_regression_matches_cpu(self, miss, method):
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


class TestMpsDegenerate:
    def test_collinear_predictors_never_silent_nan(self):
        # The batched draw runs sync-free and defers fail-loud to the backend's
        # end-of-sweep non-finite guard. With heavily collinear predictors the
        # ridge keeps the Gram solvable; either way the result must be finite or
        # the run must fail loud — never a silently-returned NaN.
        from pystatistics.core.exceptions import ValidationError

        rng = np.random.default_rng(0)
        n = 300
        x = rng.standard_normal((n, 1))
        X = np.column_stack([
            x,                                    # fully observed anchor
            x + 1e-9 * rng.standard_normal((n, 1)),  # near-duplicate of col 0
            rng.standard_normal((n, 3)),
        ])
        mask = rng.random(X.shape) < 0.2
        mask[:, 0] = False
        X[mask] = np.nan
        try:
            sol = mice(X, m=4, maxit=5, method="pmm", seed=0, backend="gpu")
        except ValidationError:
            return  # acceptable: failed loud rather than returning garbage
        for d in sol.completed_datasets():
            assert np.isfinite(d).all()


class TestMpsScales:
    def test_large_n_matches_cpu_distribution(self):
        complete = datasets.make_gaussian_complete(8000, seed=2)
        miss = datasets.make_mcar(complete, 0.2, seed=3)
        cpu = mice(miss, m=20, maxit=8, method="pmm", seed=5, backend="cpu")
        gpu = mice(miss, m=20, maxit=8, method="pmm", seed=5, backend="gpu")
        for j in gpu.incomplete_columns:
            assert abs(gpu.imputations(j).mean() - cpu.imputations(j).mean()) < 0.1
