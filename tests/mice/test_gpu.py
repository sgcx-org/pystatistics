"""
GPU backend tests for MICE.

The GPU path is validated against the trusted CPU implementation (which is in
turn validated against R), not against R directly — the two-tier strategy used
across pystatistics. GPU results are stochastic and run in FP32, so we check
distributional agreement with CPU at the GPU/FP32 tolerance, plus the structural
invariants (no NaN, observed values preserved, PMM donor property).

All tests skip when no CUDA GPU is present.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pystatistics.mice import datasets, mice
from pystatistics.mice.design import MICEDesign

_REF_DIR = Path(__file__).parent / "references"
_REF_JSON = _REF_DIR / "mice_categorical_reference.json"
_REF_CSV = _REF_DIR / "mice_categorical_data.csv"


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
        sol = mice(miss, n_imputations=4, max_iter=5, seed=0, backend="gpu")
        assert "gpu" in sol.backend_name
        assert sol.info["device"] == "cuda"
        assert sol.info["precision"] == "fp32"

    def test_completed_no_nan(self, miss):
        sol = mice(miss, n_imputations=4, max_iter=5, seed=0, backend="gpu")
        for d in sol.completed_datasets():
            assert not np.isnan(d).any()
            assert d.shape == miss.shape

    def test_observed_preserved(self, miss):
        sol = mice(miss, n_imputations=3, max_iter=4, seed=0, backend="gpu")
        observed = ~np.isnan(miss)
        for d in sol.completed_datasets():
            # FP32 round-trip: observed values preserved to single precision.
            np.testing.assert_allclose(d[observed], miss[observed], rtol=1e-5, atol=1e-5)

    def test_auto_selects_gpu_on_cuda(self, miss):
        sol = mice(miss, n_imputations=3, max_iter=3, seed=0, backend="auto")
        assert "gpu" in sol.backend_name


class TestGpuReproducibility:
    def test_same_seed_reproducible(self, miss):
        a = mice(miss, n_imputations=4, max_iter=5, method="pmm", seed=11, backend="gpu")
        b = mice(miss, n_imputations=4, max_iter=5, method="pmm", seed=11, backend="gpu")
        for da, db in zip(a.completed_datasets(), b.completed_datasets()):
            # Same seed + same device: identical up to FP32 kernel determinism.
            np.testing.assert_allclose(da, db, rtol=1e-5, atol=1e-6)

    def test_different_seed_differs(self, miss):
        a = mice(miss, n_imputations=4, max_iter=5, method="norm", seed=1, backend="gpu")
        b = mice(miss, n_imputations=4, max_iter=5, method="norm", seed=2, backend="gpu")
        assert any(
            not np.allclose(da, db)
            for da, db in zip(a.completed_datasets(), b.completed_datasets())
        )


class TestGpuPmmDonorProperty:
    def test_fp32_imputes_near_observed(self, miss):
        sol = mice(miss, n_imputations=5, max_iter=5, method="pmm", seed=0, backend="gpu")
        for j in sol.incomplete_columns:
            observed = miss[~np.isnan(miss[:, j]), j]
            imp = sol.imputations(j).ravel()
            dist = np.abs(imp[:, None] - observed[None, :]).min(axis=1)
            # Every imputed value is a donor copy, exact up to FP32 rounding.
            assert dist.max() < 1e-4

    def test_fp64_imputes_exact_observed(self, miss):
        sol = mice(miss, n_imputations=5, max_iter=5, method="pmm", seed=0, backend="gpu_fp64")
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
        cpu = mice(miss, n_imputations=m, max_iter=10, method=method, seed=100, backend="cpu")
        gpu = mice(miss, n_imputations=m, max_iter=10, method=method, seed=100, backend="gpu")
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
            return pool(np.array(est), np.array(var), df_complete=len(miss) - 3)

        m = 30
        cpu = pooled(mice(miss, n_imputations=m, max_iter=10, method=method, seed=7, backend="cpu"))
        gpu = pooled(mice(miss, n_imputations=m, max_iter=10, method=method, seed=7, backend="gpu"))
        np.testing.assert_allclose(
            np.asarray(gpu.estimate), np.asarray(cpu.estimate), atol=0.1
        )


class TestGpuScales:
    def test_large_n_donor_search_is_memory_frugal(self):
        # The windowed donor matcher must not rebuild a dense (m, n_mis, n_obs)
        # distance tensor. At this size the dense version needed ~1+ GB (and
        # OOM'd at larger n); the windowed path should stay well under that.
        import torch

        complete = datasets.make_gaussian_complete(12000, seed=0)
        miss = datasets.make_mcar(complete, 0.2, seed=1)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        sol = mice(miss, n_imputations=20, max_iter=5, method="pmm", seed=0, backend="gpu")
        peak_gb = torch.cuda.max_memory_allocated() / 1e9

        assert not np.isnan(sol.completed(0)).any()
        assert peak_gb < 1.0, (
            f"peak GPU memory {peak_gb:.2f} GB suggests a dense donor matrix"
        )

    def test_large_n_matches_cpu_distribution(self):
        # Correctness preserved at scale: windowed GPU donors agree with CPU.
        complete = datasets.make_gaussian_complete(8000, seed=2)
        miss = datasets.make_mcar(complete, 0.2, seed=3)
        cpu = mice(miss, n_imputations=20, max_iter=8, method="pmm", seed=5, backend="cpu")
        gpu = mice(miss, n_imputations=20, max_iter=8, method="pmm", seed=5, backend="gpu")
        for j in gpu.incomplete_columns:
            assert abs(gpu.imputations(j).mean() - cpu.imputations(j).mean()) < 0.1


class TestGpuCategoricalTargets:
    """Categorical (incomplete) targets imputed on CUDA: binary (logreg),
    unordered (polyreg), ordered (polr). Validated against the trusted CPU methods
    distributionally at the GPU/FP32 tolerance, plus structural invariants. The
    device-specific numerics (batched IRLS/Newton, ``solve_triangular`` branch,
    autograd Hessian for polr) are additionally validated directly on CUDA in
    ``scratch/mice_cat/cuda_validation.py`` (machine-precision vs CPU)."""

    @staticmethod
    def _binary_design(seed=0):
        rng = np.random.default_rng(seed)
        n = 600
        x1, x2 = rng.standard_normal(n), rng.standard_normal(n)
        p = 1.0 / (1.0 + np.exp(-(0.4 + 1.1 * x1 - 0.7 * x2)))
        b = (rng.random(n) < p).astype(float)
        X = np.column_stack([b, x1, x2])
        X[rng.random(n) < 0.25, 0] = np.nan
        return MICEDesign.from_array(X, column_kinds=["binary", "numeric", "numeric"])

    @staticmethod
    def _multi_design(K=4, ordered=False, seed=0):
        rng = np.random.default_rng(seed)
        n, q = 700, 3
        X = rng.standard_normal((n, q))
        if ordered:
            eta = X @ (rng.standard_normal(q) * 0.9)
            cuts = np.linspace(-1.2, 1.2, K - 1)
            u = rng.uniform(size=n)
            cum = 1.0 / (1.0 + np.exp(-(cuts[None, :] - eta[:, None])))
            y = (u[:, None] > cum).sum(axis=1).astype(int)
        else:
            B = rng.standard_normal((K - 1, q + 1)) * 1.1
            Xa = np.column_stack([np.ones(n), X])
            e = np.column_stack([Xa @ B.T, np.zeros(n)])
            e -= e.max(1, keepdims=True)
            pp = np.exp(e); pp /= pp.sum(1, keepdims=True)
            y = np.array([rng.choice(K, p=pp[i]) for i in range(n)], dtype=int)
        for lv in range(K):
            if not np.any(y == lv):
                y[lv] = lv
        data = np.column_stack([y.astype(float), X])
        data[rng.random(n) < 0.25, 0] = np.nan
        kind = "ordered" if ordered else "categorical"
        return MICEDesign.from_array(data, column_kinds=[kind] + ["numeric"] * q)

    @pytest.mark.parametrize("kind,builder", [
        ("logreg", lambda s: TestGpuCategoricalTargets._binary_design(s)),
        ("polyreg", lambda s: TestGpuCategoricalTargets._multi_design(seed=s)),
        ("polr", lambda s: TestGpuCategoricalTargets._multi_design(ordered=True, seed=s)),
    ])
    def test_proportions_match_cpu(self, kind, builder):
        d = builder(7)
        assert d.method_for(0) == kind
        m = 30
        cpu = mice(d, n_imputations=m, max_iter=10, seed=7, backend="cpu")
        gpu = mice(d, n_imputations=m, max_iter=10, seed=7, backend="gpu")
        assert gpu.info["device"] == "cuda"
        levels = d.levels_for(0)
        ci, gi = cpu.imputations(0).ravel(), gpu.imputations(0).ravel()
        cp = np.array([np.mean(ci == lv) for lv in levels])
        gp = np.array([np.mean(gi == lv) for lv in levels])
        assert np.max(np.abs(cp - gp)) < 0.06, f"{kind}: cpu={np.round(cp,3)} gpu={np.round(gp,3)}"
        assert set(np.unique(gi).tolist()).issubset(set(levels.tolist()))
        for dset in gpu.completed_datasets():
            assert not np.isnan(dset).any()

    def test_categorical_targets_reproducible(self):
        d = self._multi_design(ordered=True, seed=2)
        a = mice(d, n_imputations=5, max_iter=5, seed=11, backend="gpu")
        b = mice(d, n_imputations=5, max_iter=5, seed=11, backend="gpu")
        for da, db in zip(a.completed_datasets(), b.completed_datasets()):
            np.testing.assert_array_equal(da, db)


@pytest.mark.skipif(
    not (_REF_JSON.exists() and _REF_CSV.exists()),
    reason="R categorical fixtures absent (run generate_categorical_fixtures.R)",
)
class TestGpuCategoricalMatchesR:
    """Direct CUDA-vs-R validation on the mixed bin/nom/ord fixture (all categorical
    methods on GPU). Distributional agreement with R ``mice`` 3.19.0."""

    _COL_BY_NAME = {"bin": 1, "nom": 2, "ord": 3}

    @pytest.fixture(scope="class")
    def r_and_gpu(self):
        ref = json.load(open(_REF_JSON))
        matrix = np.genfromtxt(_REF_CSV, delimiter=",", skip_header=1)
        design = MICEDesign.from_array(
            matrix, column_kinds=["numeric", "binary", "categorical", "ordered"]
        )
        meta = ref["meta"]
        sol = mice(design, n_imputations=meta["m"], max_iter=meta["maxit"], seed=20260614, backend="gpu")
        return ref, sol

    @pytest.mark.parametrize("name", ["bin", "nom", "ord"])
    def test_proportions_match_r(self, r_and_gpu, name):
        ref, sol = r_and_gpu
        assert sol.info["device"] == "cuda"
        col = self._COL_BY_NAME[name]
        levels = [int(lv) for lv in ref[name]["levels"]]
        imp = sol.imputations(col).ravel()
        counts = np.array([np.sum(imp == float(lv)) for lv in levels], dtype=float)
        ours = counts / counts.sum()
        np.testing.assert_allclose(
            ours, np.asarray(ref[name]["proportions"], dtype=float), atol=0.06
        )


class TestGpuFp64:
    def test_fp64_runs(self, miss):
        sol = mice(miss, n_imputations=3, max_iter=4, seed=0, backend="gpu_fp64")
        assert sol.info["precision"] == "fp64"
        for d in sol.completed_datasets():
            assert not np.isnan(d).any()

    def test_fp64_closer_to_cpu_observed(self, miss):
        # FP64 preserves observed values to (near) machine precision.
        sol = mice(miss, n_imputations=2, max_iter=3, seed=0, backend="gpu_fp64")
        observed = ~np.isnan(miss)
        for d in sol.completed_datasets():
            np.testing.assert_allclose(d[observed], miss[observed], rtol=1e-12, atol=1e-12)
