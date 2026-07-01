"""Tests for the low-rank / GRM mixed model (grm_lmm).

CPU is the float64 reference (validated against an independent n×n V-space
deviance). The GPU backend is validated against the CPU reference at the
GPU_FP32 statistical-equivalence tier, and its CF-1 conditioning gate is
exercised (refuse ill-conditioned float32 loudly; never silently wrong).
"""

import numpy as np
import pytest

from pystatistics.mixed import grm_lmm, GRMSolution
from pystatistics.core.exceptions import ValidationError, NumericalError
from pystatistics.core.compute.device import detect_gpu


def _gen(n, M, p=2, h2=0.5, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, M))
    W -= W.mean(0)
    W /= (W.std(0) + 1e-8)
    X = np.column_stack([np.ones(n)] + [rng.standard_normal(n) for _ in range(p - 1)])
    g = (W @ (rng.standard_normal(M) * np.sqrt(h2))) / np.sqrt(M)
    y = X @ np.arange(1, p + 1) + g + rng.standard_normal(n) * np.sqrt(1 - h2)
    return y, X, W


def _dev_vspace(theta, W, X, y):
    n, M = W.shape
    p = X.shape[1]
    V = (theta ** 2 / M) * (W @ W.T) + np.eye(n)
    Lv = np.linalg.cholesky(V)
    Vi = np.linalg.inv(V)
    XtViX = X.T @ Vi @ X
    beta = np.linalg.solve(XtViX, X.T @ Vi @ y)
    r = y - X @ beta
    logdetV = 2.0 * np.sum(np.log(np.diag(Lv)))
    _, logdetXVX = np.linalg.slogdet(XtViX)
    df = n - p
    return logdetV + logdetXVX + df * (1 + np.log(2 * np.pi * float(r @ Vi @ r) / df))


_HAS_GPU = detect_gpu() is not None
_GPU_TYPE = detect_gpu().device_type if _HAS_GPU else None


class TestGRMCpu:
    def test_returns_solution(self):
        y, X, W = _gen(800, 200, seed=1)
        r = grm_lmm(y, X, W)
        assert isinstance(r, GRMSolution)
        assert r.converged
        assert r.backend_name == "grm_cpu"

    def test_deviance_matches_vspace(self):
        """M-space profiled deviance == independent n×n V-space deviance."""
        from pystatistics.mixed._grm_cpu import grm_deviance_cpu
        y, X, W = _gen(400, 120, seed=3)
        for th in (0.3, 1.0, 1.7, 3.0):
            dm = grm_deviance_cpu(th, W, X, y, reml=True)
            dv = _dev_vspace(th, W, X, y)
            assert abs(dm - dv) < 1e-7

    def test_heritability_recovery(self):
        for h2 in (0.2, 0.5, 0.8):
            ests = [grm_lmm(*_gen(1500, 600, h2=h2, seed=100 + s)).heritability
                    for s in range(4)]
            assert abs(np.mean(ests) - h2) < 0.05

    def test_se_form_a_matches_dense(self):
        y, X, W = _gen(800, 200, seed=9)
        r = grm_lmm(y, X, W)
        n, M = W.shape
        V = (r.params.theta ** 2 / M) * (W @ W.T) + np.eye(n)
        Vi = np.linalg.inv(V)
        C = np.linalg.inv(X.T @ Vi @ X)
        se_dense = np.sqrt(np.diag(r.var_residual * C))
        np.testing.assert_allclose(r.standard_errors, se_dense, atol=1e-9)

    def test_variance_components_consistent(self):
        y, X, W = _gen(1000, 300, h2=0.6, seed=4)
        r = grm_lmm(y, X, W)
        assert r.var_genetic > 0 and r.var_residual > 0
        np.testing.assert_allclose(
            r.heritability, r.var_genetic / (r.var_genetic + r.var_residual),
            atol=1e-12)
        np.testing.assert_allclose(
            r.variance_ratio, r.var_genetic / r.var_residual, rtol=1e-9)

    def test_reml_vs_ml_differ(self):
        y, X, W = _gen(600, 150, seed=5)
        assert (grm_lmm(y, X, W, reml=True).var_residual
                != grm_lmm(y, X, W, reml=False).var_residual)

    def test_accessors_and_summary(self):
        y, X, W = _gen(500, 100, seed=6)
        r = grm_lmm(y, X, W, names=("intercept", "age"))
        assert set(r.coef) == {"intercept", "age"}
        assert r.conf_int.shape == (2, 2)
        assert r.genetic_values.shape == (500,)
        assert np.all(np.isfinite(r.p_values))
        assert "Heritability" in r.summary()


class TestGRMValidation:
    def test_row_mismatch_raises(self):
        y, X, W = _gen(200, 50)
        with pytest.raises(ValidationError):
            grm_lmm(y[:100], X, W)

    def test_non_finite_raises(self):
        y, X, W = _gen(200, 50)
        y2 = y.copy(); y2[0] = np.nan
        with pytest.raises(ValidationError):
            grm_lmm(y2, X, W)

    def test_bad_conf_level_raises(self):
        y, X, W = _gen(200, 50)
        with pytest.raises(ValidationError):
            grm_lmm(y, X, W, conf_level=1.5)

    def test_unknown_backend_raises(self):
        y, X, W = _gen(200, 50)
        with pytest.raises(ValidationError):
            grm_lmm(y, X, W, backend="banana")

    def test_names_length_mismatch_raises(self):
        y, X, W = _gen(200, 50)
        with pytest.raises(ValidationError):
            grm_lmm(y, X, W, names=("a", "b", "c"))


@pytest.mark.skipif(not _HAS_GPU, reason="no GPU (CUDA/MPS) available")
class TestGRMGpu:
    def test_gpu_fp32_matches_cpu_at_tier(self):
        """GPU float32 heritability/β match the CPU reference at the
        statistical-equivalence (GPU_FP32) tier."""
        for seed in (1, 2, 3):
            y, X, W = _gen(4000, 800, h2=0.5, seed=seed)
            rc = grm_lmm(y, X, W, backend="cpu")
            rg = grm_lmm(y, X, W, backend="gpu")
            assert abs(rc.heritability - rg.heritability) < 5e-3
            np.testing.assert_allclose(rg.coefficients, rc.coefficients,
                                       rtol=1e-3, atol=1e-4)

    def test_cf1_gate_refuses_ill_conditioned(self):
        y, X, W = _gen(2000, 100, seed=7)
        rng = np.random.default_rng(0)
        W[:, 50:] = W[:, :50] + 1e-7 * rng.standard_normal((2000, 50))
        with pytest.raises(NumericalError):
            grm_lmm(y, X, W, backend="gpu")
        # CPU float64 handles the same data.
        assert grm_lmm(y, X, W, backend="cpu").converged

    def test_cf1_gate_accepts_well_conditioned(self):
        y, X, W = _gen(3000, 500, seed=11)
        assert grm_lmm(y, X, W, backend="gpu").converged

    def test_force_bypasses_gate(self):
        """force=True skips the conditioning gate (runs or fails loud, never
        silently refuses a forced request)."""
        rng = np.random.default_rng(1)
        n = 3000
        B = rng.standard_normal((n, 30))
        W = np.hstack([B, B + 3e-3 * rng.standard_normal((n, 30))])
        W -= W.mean(0); W /= (W.std(0) + 1e-8)
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [1, 2] + (W @ (rng.standard_normal(60) * 0.7)) / np.sqrt(60) \
            + rng.standard_normal(n) * 0.7
        r = grm_lmm(y, X, W, backend="gpu", force=True)
        assert np.isfinite(r.heritability)

    @pytest.mark.skipif(_GPU_TYPE != "mps", reason="MPS-only behavior")
    def test_gpu_fp64_on_mps_raises(self):
        y, X, W = _gen(500, 100, seed=2)
        with pytest.raises(RuntimeError):
            grm_lmm(y, X, W, backend="gpu_fp64")

    @pytest.mark.skipif(_GPU_TYPE != "cuda", reason="gpu_fp64 is CUDA-only")
    def test_gpu_fp64_matches_cpu(self):
        """gpu_fp64 reproduces the CPU fp64 fit. Fixed effects match to fp64
        precision (the linear algebra is exact); the variance components match
        to the optimizer-stop level — the 1-D θ search lands a hair differently
        off ~1e-12 cuSOLVER-vs-LAPACK deviance rounding, the same effect
        documented for the polr gpu_fp64 path (the stop bounds the deviance,
        not the derived quantity)."""
        for seed in (1, 2, 3):
            y, X, W = _gen(6000, 800, seed=seed)
            rc = grm_lmm(y, X, W, backend="cpu")
            rg = grm_lmm(y, X, W, backend="gpu_fp64")
            np.testing.assert_allclose(rg.coefficients, rc.coefficients,
                                       rtol=1e-6, atol=1e-8)
            assert abs(rg.heritability - rc.heritability) < 1e-5


def test_auto_backend_falls_back_to_cpu_when_no_gpu(monkeypatch):
    """backend='auto' must succeed on CPU when no CUDA is present."""
    import pystatistics.core.compute.backend as bk
    monkeypatch.setattr(bk._device, "detect_gpu", lambda: None)
    y, X, W = _gen(400, 100, seed=8)
    r = grm_lmm(y, X, W, backend="auto")
    assert r.backend_name == "grm_cpu"
    assert r.converged
