"""
Tests for the multivariate analysis module.

Covers PCA, factor analysis, and rotation methods.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

from pystatistics.multivariate import pca, factor_analysis, PCASolution, FactorSolution
from pystatistics.multivariate._rotation import varimax, promax
from pystatistics.core.exceptions import ValidationError, ConvergenceError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def iris_like_data():
    """Synthetic data with known structure (4 variables, 50 obs)."""
    rng = np.random.default_rng(42)
    n = 50
    # Two latent factors
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    X = np.column_stack([
        2 * f1 + 0.1 * rng.standard_normal(n),
        1.5 * f1 + 0.5 * f2 + 0.1 * rng.standard_normal(n),
        0.3 * f1 + 2 * f2 + 0.1 * rng.standard_normal(n),
        0.1 * f1 + 1.8 * f2 + 0.1 * rng.standard_normal(n),
    ])
    return X


@pytest.fixture
def simple_matrix():
    """Simple 5x3 matrix for basic PCA tests."""
    rng = np.random.default_rng(123)
    return rng.standard_normal((5, 3))


@pytest.fixture
def factor_data():
    """Data with clear 2-factor structure for factor analysis tests."""
    rng = np.random.default_rng(99)
    n = 200
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    noise = 0.3
    X = np.column_stack([
        f1 + noise * rng.standard_normal(n),
        0.9 * f1 + noise * rng.standard_normal(n),
        0.8 * f1 + 0.2 * f2 + noise * rng.standard_normal(n),
        f2 + noise * rng.standard_normal(n),
        0.9 * f2 + noise * rng.standard_normal(n),
        0.2 * f1 + 0.8 * f2 + noise * rng.standard_normal(n),
    ])
    return X


def clean_2factor(seed: int) -> np.ndarray:
    """Clean 2-factor model (n=400, p=8), simple structure, varying loadings.

    Variables 0-3 load 0.70-0.80 on factor 1, variables 4-7 on factor 2,
    uniquenesses ~0.3-0.5. On several seeds (e.g. 24) the raw ML loadings
    need a substantial varimax rotation whose criterion plateaus -- the
    exact regime where the old *absolute* convergence test failed to
    converge in 1000 iterations while R's *relative* test converges.
    """
    rng = np.random.default_rng(seed)
    n = 400
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    t1 = rng.uniform(0.70, 0.80, 4)
    t2 = rng.uniform(0.70, 0.80, 4)
    cols = []
    for L in t1:
        cols.append(L * f1 + np.sqrt(1 - L ** 2) * rng.standard_normal(n))
    for L in t2:
        cols.append(L * f2 + np.sqrt(1 - L ** 2) * rng.standard_normal(n))
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# R cross-validation helpers (skipped when Rscript is unavailable)
# ---------------------------------------------------------------------------

_HAS_RSCRIPT = shutil.which("Rscript") is not None
_requires_r = pytest.mark.skipif(not _HAS_RSCRIPT, reason="Rscript not available")


def _r_factanal(X, tmp_path, *, n_factors, rotation, lower=0.005):
    """Run R ``stats::factanal`` and return (loadings, uniquenesses, objective).

    Loadings are returned in (p, m) shape (R stores them column-major).
    """
    csv = os.path.join(str(tmp_path), "fa_data.csv")
    np.savetxt(csv, np.asarray(X), delimiter=",")
    script = (
        f'X <- as.matrix(read.csv("{csv}", header=FALSE))\n'
        f'fit <- factanal(X, factors={n_factors}, rotation="{rotation}", lower={lower})\n'
        'cat("UNIQ:", paste(format(fit$uniquenesses, digits=15), collapse=","), "\\n")\n'
        'cat("CRIT:", format(fit$criteria["objective"], digits=15), "\\n")\n'
        'cat("DIM:", nrow(fit$loadings), ncol(fit$loadings), "\\n")\n'
        'cat("LOAD:", paste(format(as.vector(fit$loadings), digits=15), collapse=","), "\\n")\n'
    )
    out = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("Rscript failed:\n" + out.stderr)
    uniq = crit = dim = load_flat = None
    for line in out.stdout.splitlines():
        if line.startswith("UNIQ:"):
            uniq = np.array([float(v) for v in line[5:].strip().split(",")])
        elif line.startswith("CRIT:"):
            crit = float(line[5:].strip())
        elif line.startswith("DIM:"):
            dim = tuple(int(v) for v in line[4:].split())
        elif line.startswith("LOAD:"):
            load_flat = np.array([float(v) for v in line[5:].strip().split(",")])
    loadings = load_flat.reshape(dim, order="F")
    return loadings, uniq, crit


def _loading_diff_up_to_rotation(py, r):
    """Max abs difference between loadings up to column permutation + sign."""
    from itertools import permutations

    p, m = py.shape
    best = np.inf
    for perm in permutations(range(m)):
        cand = py[:, perm].astype(float).copy()
        for j in range(m):
            if np.sum((cand[:, j] - r[:, j]) ** 2) > np.sum((-cand[:, j] - r[:, j]) ** 2):
                cand[:, j] = -cand[:, j]
        best = min(best, np.max(np.abs(cand - r)))
    return best


# ===========================================================================
# PCA Tests
# ===========================================================================

class TestPCA:
    """Tests for principal component analysis."""

    def test_reconstruction(self, iris_like_data):
        """Reconstruct X from scores and loadings: X_centered ~ scores @ rotation.T.

        Uses backend='cpu' explicitly — this test checks a rank-preservation
        identity at atol=1e-10, which is a CPU-reference tolerance. The
        GPU path runs in FP32 by design and satisfies this identity only
        at ``GPU_FP32`` precision; testing that is the job of TestPCAGPU.
        """
        result = pca(iris_like_data, backend="cpu")
        X_centered = iris_like_data - result.center
        reconstructed = result.x @ result.rotation.T
        np.testing.assert_allclose(X_centered, reconstructed, atol=1e-10)

    def test_total_variance(self, iris_like_data):
        """sdev^2 sum equals total variance of centered data (CPU precision)."""
        result = pca(iris_like_data, backend="cpu")
        total_var_from_sdev = np.sum(result.sdev ** 2)
        X_centered = iris_like_data - np.mean(iris_like_data, axis=0)
        # Total variance = sum of column variances (ddof=1)
        total_var_from_data = np.sum(np.var(X_centered, axis=0, ddof=1))
        np.testing.assert_allclose(total_var_from_sdev, total_var_from_data, rtol=1e-10)

    def test_explained_variance_ratio_sums_to_one(self, iris_like_data):
        """Explained variance ratio sums to 1.0."""
        result = pca(iris_like_data)
        np.testing.assert_allclose(np.sum(result.explained_variance_ratio), 1.0, atol=1e-14)

    def test_scores_orthogonal(self, iris_like_data):
        """Scores are orthogonal (uncorrelated). CPU precision."""
        result = pca(iris_like_data, backend="cpu")
        # scores.T @ scores should be diagonal (up to scale)
        gram = result.x.T @ result.x
        # Off-diagonal elements should be ~0
        off_diag = gram - np.diag(np.diag(gram))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-10)

    def test_n_components_truncation(self, iris_like_data):
        """n_components truncation works correctly."""
        result = pca(iris_like_data, n_components=2)
        assert result.sdev.shape == (2,)
        assert result.rotation.shape == (4, 2)
        assert result.x.shape == (50, 2)

    def test_center_false(self, simple_matrix):
        """center=False skips centering."""
        result = pca(simple_matrix, center=False)
        np.testing.assert_allclose(result.center, np.zeros(3), atol=1e-14)

    def test_scale_true(self, iris_like_data):
        """scale=True produces correlation-based PCA. CPU precision."""
        result = pca(iris_like_data, scale=True, backend="cpu")
        assert result.scale is not None
        # With scaling, total variance should equal number of variables
        total_var = np.sum(result.sdev ** 2)
        np.testing.assert_allclose(total_var, iris_like_data.shape[1], rtol=1e-10)

    def test_single_component(self, iris_like_data):
        """Single component extraction."""
        result = pca(iris_like_data, n_components=1)
        assert result.sdev.shape == (1,)
        assert result.rotation.shape == (4, 1)
        assert result.x.shape == (50, 1)

    def test_sign_convention(self, iris_like_data):
        """Largest absolute loading is positive per component."""
        result = pca(iris_like_data)
        for j in range(result.rotation.shape[1]):
            col = result.rotation[:, j]
            max_abs_idx = np.argmax(np.abs(col))
            assert col[max_abs_idx] > 0, (
                f"Component {j}: largest absolute loading should be positive"
            )

    def test_names_propagate(self, iris_like_data):
        """Names propagate to var_names."""
        names = ["V1", "V2", "V3", "V4"]
        result = pca(iris_like_data, names=names)
        assert result.var_names == ("V1", "V2", "V3", "V4")

    def test_names_none_by_default(self, iris_like_data):
        """var_names is None when names not provided."""
        result = pca(iris_like_data)
        assert result.var_names is None

    def test_wrong_names_length_raises(self, iris_like_data):
        """Wrong names length raises ValidationError."""
        with pytest.raises(ValidationError, match="does not match"):
            pca(iris_like_data, names=["A", "B"])

    def test_too_few_observations_raises(self):
        """Single observation raises ValidationError."""
        X = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValidationError, match="at least 2"):
            pca(X)

    def test_n_components_too_large_raises(self, simple_matrix):
        """n_components > min(n,p) raises ValidationError."""
        with pytest.raises(ValidationError, match="exceeds max"):
            pca(simple_matrix, n_components=10)

    def test_n_components_zero_raises(self, simple_matrix):
        """n_components=0 raises ValidationError."""
        with pytest.raises(ValidationError, match="must be >= 1"):
            pca(simple_matrix, n_components=0)

    def test_zero_variance_with_scale_raises(self):
        """Constant column with scale=True raises ValidationError."""
        X = np.column_stack([
            np.ones(10),
            np.random.default_rng(0).standard_normal(10),
        ])
        with pytest.raises(ValidationError, match="zero variance"):
            pca(X, scale=True)

    def test_result_is_frozen(self, iris_like_data):
        """PCASolution is frozen dataclass."""
        result = pca(iris_like_data)
        with pytest.raises(AttributeError):
            result.sdev = np.zeros(4)  # type: ignore[misc]

    def test_summary_runs(self, iris_like_data):
        """summary() produces a non-empty string."""
        result = pca(iris_like_data)
        s = result.summary()
        assert isinstance(s, str)
        assert "Standard deviation" in s
        assert "Cumulative Proportion" in s

    def test_cumulative_variance_ratio(self, iris_like_data):
        """Cumulative variance ratio is monotonically increasing to 1."""
        result = pca(iris_like_data)
        cvr = result.cumulative_variance_ratio
        assert cvr[-1] == pytest.approx(1.0, abs=1e-14)
        # Monotonically non-decreasing
        assert np.all(np.diff(cvr) >= -1e-15)

    def test_list_input(self):
        """Accepts list-of-lists input."""
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = pca(X)
        assert isinstance(result, PCASolution)

    def test_n_obs_n_vars(self, iris_like_data):
        """n_obs and n_vars are set correctly."""
        result = pca(iris_like_data)
        assert result.n_obs == 50
        assert result.n_vars == 4


class TestPCAGPU:
    """Tests for the GPU PCA backend.

    These follow the project's two-tier validation rule: CPU is
    validated against R; GPU is validated against CPU. FP32 runs match
    CPU to ~1e-6 relative; FP64 on CUDA should match to machine
    precision.
    """

    def _gpu_available(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        # PCA GPU is CUDA-only: the SVD / symmetric eigendecomposition
        # PCA needs has no Metal kernel and silently falls back to CPU on
        # MPS, so there is no genuine GPU PCA on Apple Silicon (the
        # backend raises there — see test_pca_gpu_on_mps_raises).
        return torch.cuda.is_available()

    def test_gpu_unavailable_raises_explicitly(self, iris_like_data, monkeypatch):
        """backend='gpu' must raise when no GPU is available, not silently
        fall back to CPU (Rule 1: no hidden fallbacks when caller is
        explicit)."""
        from pystatistics.core.compute import device as dev_mod

        def no_gpu(*_a, **_k):
            return None
        monkeypatch.setattr(dev_mod, "detect_gpu", no_gpu)
        with pytest.raises(RuntimeError, match="No GPU available"):
            pca(iris_like_data, backend="gpu")

    def test_invalid_backend_raises(self, iris_like_data):
        with pytest.raises(ValidationError, match="backend"):
            pca(iris_like_data, backend="quantum")

    def test_explicit_svd_gram_on_mps_raise(self, iris_like_data):
        """On Apple Silicon, an *explicit* solver='svd'/'gram' must fail loud
        rather than silently running on the CPU (Rule 1 / A6): svd silently
        falls back to the CPU and eigh has no Metal kernel. The on-device
        randomized path (the default on MPS) is exercised by
        :class:`TestPCARandomizedMPS`."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        mps_only = (
            not torch.cuda.is_available()
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        if not mps_only:
            pytest.skip("requires an MPS-only machine")
        for solver in ("svd", "gram"):
            with pytest.raises(RuntimeError, match="Metal"):
                pca(iris_like_data, backend="gpu", solver=solver)

    def test_gpu_fp64_matches_cpu(self, iris_like_data):
        """FP64 GPU path should match CPU to machine precision."""
        if not self._gpu_available():
            pytest.skip("no GPU available")
        try:
            import torch
            if torch.cuda.is_available():
                dtype_ok = True
            else:
                # MPS does not support FP64 — the GPU path raises.
                dtype_ok = False
        except ImportError:
            dtype_ok = False
        if not dtype_ok:
            pytest.skip("MPS has no FP64")

        r_cpu = pca(iris_like_data, center=True, scale=True, backend="cpu")
        r_gpu = pca(
            iris_like_data, center=True, scale=True,
            backend="gpu_fp64",
        )
        np.testing.assert_allclose(r_cpu.sdev, r_gpu.sdev, rtol=1e-12)
        # Rotation can sign-flip between implementations; compare |.|
        np.testing.assert_allclose(
            np.abs(r_cpu.rotation), np.abs(r_gpu.rotation), atol=1e-12,
        )
        np.testing.assert_allclose(
            np.abs(r_cpu.x), np.abs(r_gpu.x), atol=1e-10,
        )

    def test_gpu_fp32_matches_cpu_at_gpu_fp32_tolerance(self, iris_like_data):
        """FP32 GPU path matches CPU at the project's GPU_FP32 tier.

        The README's two-tier validation rule says CPU is validated
        against R to 1e-10 and GPU is validated against CPU at the
        tolerance tier defined in ``pystatistics.core.compute.tolerances``.
        GPU_FP32 is rtol=1e-4, atol=1e-5 — "statistically equivalent".
        This is the tolerance we hold GPU to, not the (tighter) machine
        precision we hold CPU-vs-R to.
        """
        from pystatistics.core.compute.tolerances import GPU_FP32
        if not self._gpu_available():
            pytest.skip("no GPU available")
        r_cpu = pca(iris_like_data, center=True, scale=True, backend="cpu")
        r_gpu = pca(
            iris_like_data, center=True, scale=True,
            backend="gpu",
        )
        np.testing.assert_allclose(
            r_cpu.sdev, r_gpu.sdev,
            rtol=GPU_FP32.rtol, atol=GPU_FP32.atol,
        )
        np.testing.assert_allclose(
            np.abs(r_cpu.rotation), np.abs(r_gpu.rotation),
            rtol=GPU_FP32.rtol, atol=GPU_FP32.atol,
        )
        np.testing.assert_allclose(
            np.abs(r_cpu.x), np.abs(r_gpu.x),
            rtol=GPU_FP32.rtol, atol=GPU_FP32.atol,
        )

    def test_auto_backend_falls_back_to_cpu_when_no_gpu(self, iris_like_data, monkeypatch):
        """backend='auto' must NOT raise when no GPU is available — it
        falls back to CPU (that is the definition of 'auto')."""
        from pystatistics.core.compute import device as dev_mod
        monkeypatch.setattr(dev_mod, "detect_gpu", lambda *a, **k: None)
        # Should succeed, not raise.
        r = pca(iris_like_data, backend="auto")
        # And should match the explicit CPU path.
        r_cpu = pca(iris_like_data, backend="cpu")
        np.testing.assert_allclose(r.sdev, r_cpu.sdev, rtol=1e-14)

    def test_invalid_method_raises(self, iris_like_data):
        """method must be in {'svd', 'gram', 'auto'}."""
        with pytest.raises(ValidationError, match="method"):
            pca(iris_like_data, backend="gpu", solver="magic")

    def test_gram_matches_svd_well_conditioned(self):
        """Gram path matches SVD path at GPU_FP32 tier on tall-skinny
        well-conditioned data.

        Comparison target is the invariant part of the PCA answer:
            - singular values (uniquely determined)
            - subspace projection V V^T  (invariant to rotations of V
              within any degenerate eigenspace — two algorithms can
              legitimately produce slightly different V for loadings
              with near-equal singular values, but they must span the
              same subspace to the tier tolerance).
        Loadings compared elementwise can diverge above rtol when two
        sdev values are within ~1e-4 of each other even when both
        implementations are statistically indistinguishable.
        """
        from pystatistics.core.compute.tolerances import GPU_FP32
        if not self._gpu_available():
            pytest.skip("no GPU available")
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5000, 20))
        r_svd = pca(X, backend="gpu", solver="svd")
        r_gram = pca(X, backend="gpu", solver="gram")
        np.testing.assert_allclose(
            r_svd.sdev, r_gram.sdev,
            rtol=GPU_FP32.rtol, atol=GPU_FP32.atol,
        )
        # Subspace invariant: V V^T is well-defined regardless of
        # rotations within degenerate eigenspaces.
        proj_svd = r_svd.rotation @ r_svd.rotation.T
        proj_gram = r_gram.rotation @ r_gram.rotation.T
        np.testing.assert_allclose(
            proj_svd, proj_gram,
            rtol=GPU_FP32.rtol, atol=GPU_FP32.atol,
        )

    def test_gram_refuses_ill_conditioned(self):
        """Gram path raises NumericalError on near-rank-deficient data
        unless force=True (Rule 1: fail loud when cond(X') is past the
        safe threshold for the current precision)."""
        from pystatistics.core.exceptions import NumericalError
        if not self._gpu_available():
            pytest.skip("no GPU available")
        rng = np.random.default_rng(1)
        X = rng.standard_normal((1000, 20))
        # Duplicate a column to make cond(X) ≈ ∞.
        X[:, 0] = X[:, 1] + 1e-7 * rng.standard_normal(1000)
        with pytest.raises(NumericalError, match="cond"):
            pca(X, backend="gpu", solver="gram")

    def test_gram_force_bypasses_condition_check(self):
        """force=True bypasses the condition gate — the fit completes
        even on ill-conditioned data, though the numerical result is
        unreliable."""
        if not self._gpu_available():
            pytest.skip("no GPU available")
        rng = np.random.default_rng(2)
        X = rng.standard_normal((1000, 20))
        X[:, 0] = X[:, 1] + 1e-7 * rng.standard_normal(1000)
        r = pca(X, backend="gpu", solver="gram", force=True)
        assert r.sdev.shape == (20,)

    def test_auto_falls_back_to_svd_on_ill_conditioned(self):
        """solver='auto' must silently fall back to SVD when Gram's
        condition check fails — that is the explicit contract of
        'auto' (unlike solver='gram' which raises)."""
        if not self._gpu_available():
            pytest.skip("no GPU available")
        rng = np.random.default_rng(3)
        X = rng.standard_normal((1000, 20))
        X[:, 0] = X[:, 1] + 1e-7 * rng.standard_normal(1000)
        r_auto = pca(X, backend="gpu", solver="auto")
        r_svd = pca(X, backend="gpu", solver="svd")
        # auto should have fallen back to SVD, so results are identical
        # (same code path, same data).
        np.testing.assert_allclose(r_auto.sdev, r_svd.sdev, rtol=1e-14)


def _mps_available() -> bool:
    """True only on an Apple-Silicon machine with a working MPS backend."""
    try:
        import torch
    except ImportError:
        return False
    return (
        not torch.cuda.is_available()
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def _planted_lowrank(rng, n, p, rank, noise=0.1):
    """Low-rank-plus-noise data with a separated spectrum — the regime PCA is
    used in, and the regime randomized SVD targets. Mirrors the validated
    investigation prototype."""
    U = rng.standard_normal((n, rank))
    Vt = np.linalg.qr(rng.standard_normal((p, rank)))[0].T   # p×rank orthonormal
    strength = np.linspace(10.0, 3.0, rank)
    return ((U * strength) @ Vt + noise * rng.standard_normal((n, p))).astype(
        np.float64
    )


@pytest.mark.skipif(not _mps_available(), reason="requires an Apple-Silicon MPS machine")
class TestPCARandomizedMPS:
    """Tests for the randomized SVD PCA path on Apple Silicon (Metal/MPS).

    This is the only genuinely on-device PCA path on MPS. Per the two-tier
    validation rule, GPU is validated against the CPU reference at the
    ``GPU_FP32`` tier (rtol 1e-4, atol 1e-5). The algorithm itself was
    validated against the fp64 numpy reference at ~1e-7 in the investigation.
    """

    # ---- Normal cases: correctness vs the CPU fp64 reference ----

    @pytest.mark.parametrize(
        "n,p,rank,k",
        [
            (5000, 200, 8, 5),     # tall
            (2000, 1500, 10, 8),   # wide
            (1500, 1500, 12, 10),  # square
        ],
    )
    def test_randomized_matches_cpu_at_gpu_fp32(self, n, p, rank, k):
        """sdev, loadings, and scores match the CPU reference at the GPU_FP32
        tier across tall / wide / square shapes.

        The comparison targets the invariant part of the PCA answer (the same
        rule the Gram-vs-SVD test uses): singular values are uniquely
        determined; loadings are compared through the subspace projection
        ``V Vᵀ`` and scores through the rank-k reconstruction ``scores @ Vᵀ``,
        both invariant to rotations within near-degenerate eigenspaces (where
        two algorithms can legitimately produce different individual loading
        vectors that nonetheless span the same subspace)."""
        from pystatistics.core.compute.tolerances import GPU_FP32
        rng = np.random.default_rng(0)
        X = _planted_lowrank(rng, n, p, rank)
        r_cpu = pca(X, n_components=k, backend="cpu")
        r_gpu = pca(X, n_components=k, backend="gpu")   # → randomized on MPS
        assert r_gpu.backend_name == "gpu_pca (mps)"
        assert r_gpu.info["method"] == "randomized"

        # Singular values are uniquely determined → elementwise at the tier.
        np.testing.assert_allclose(
            r_gpu.sdev, r_cpu.sdev, rtol=GPU_FP32.rtol, atol=GPU_FP32.atol,
        )
        # Loadings: subspace projection is well-defined regardless of rotation
        # within degenerate eigenspaces.
        proj_cpu = r_cpu.rotation @ r_cpu.rotation.T
        proj_gpu = r_gpu.rotation @ r_gpu.rotation.T
        np.testing.assert_allclose(
            proj_gpu, proj_cpu, rtol=GPU_FP32.rtol, atol=GPU_FP32.atol,
        )
        # Scores: the rank-k reconstruction is rotation-invariant. Compare on a
        # scale normalized to the data magnitude (a single small score element
        # in fp32 is not meaningful at elementwise rtol).
        recon_cpu = r_cpu.x @ r_cpu.rotation.T
        recon_gpu = r_gpu.x @ r_gpu.rotation.T
        scale = np.max(np.abs(recon_cpu))
        assert np.max(np.abs(recon_gpu - recon_cpu)) <= (
            GPU_FP32.atol + GPU_FP32.rtol * scale
        )

    def test_default_solver_routes_to_randomized_on_mps(self, iris_like_data):
        """The naive path — no solver specified — must work on MPS (it routes
        to randomized), not raise."""
        r = pca(iris_like_data, backend="gpu")
        assert r.info["method"] == "randomized"

    def test_auto_solver_routes_to_randomized_on_mps(self, iris_like_data):
        """solver='auto' on MPS routes to the randomized path."""
        r = pca(iris_like_data, backend="gpu", solver="auto")
        assert r.info["method"] == "randomized"

    def test_torch_tensor_default_runs_randomized(self):
        """A torch.Tensor already on MPS with no explicit backend runs the
        randomized path (the device-resident DataSource entry point)."""
        import torch
        from pystatistics.core.compute.tolerances import GPU_FP32
        rng = np.random.default_rng(1)
        X = _planted_lowrank(rng, 3000, 100, 6)
        Xt = torch.as_tensor(X.astype(np.float32), device="mps")
        r_gpu = pca(Xt, n_components=5)
        r_cpu = pca(X, n_components=5, backend="cpu")
        assert r_gpu.info["method"] == "randomized"
        np.testing.assert_allclose(
            r_gpu.sdev, r_cpu.sdev, rtol=GPU_FP32.rtol, atol=GPU_FP32.atol,
        )

    def test_device_resident_keeps_tensors_on_device(self):
        """device_resident=True leaves numeric fields as MPS tensors."""
        import torch
        rng = np.random.default_rng(2)
        Xt = torch.as_tensor(
            _planted_lowrank(rng, 2000, 80, 6).astype(np.float32), device="mps",
        )
        r = pca(Xt, n_components=5, device_resident=True)
        assert r.device == "mps"
        assert isinstance(r.x, torch.Tensor)
        assert r.x.device.type == "mps"

    # ---- Edge cases ----

    @pytest.mark.parametrize(
        "center,scale",
        [(True, True), (True, False), (False, True), (False, False)],
    )
    def test_center_scale_combinations(self, center, scale):
        """All center/scale combinations match the CPU reference at the tier."""
        from pystatistics.core.compute.tolerances import GPU_FP32
        rng = np.random.default_rng(3)
        X = _planted_lowrank(rng, 2000, 50, 8)
        r_cpu = pca(X, n_components=8, center=center, scale=scale, backend="cpu")
        r_gpu = pca(X, n_components=8, center=center, scale=scale, backend="gpu")
        np.testing.assert_allclose(
            r_gpu.sdev, r_cpu.sdev, rtol=GPU_FP32.rtol, atol=GPU_FP32.atol,
        )

    def test_small_n(self):
        """Small n (below the GPU-win threshold) is still correct."""
        from pystatistics.core.compute.tolerances import GPU_FP32
        rng = np.random.default_rng(4)
        X = rng.standard_normal((30, 8))
        r_cpu = pca(X, n_components=4, backend="cpu")
        r_gpu = pca(X, n_components=4, backend="gpu")
        np.testing.assert_allclose(
            r_gpu.sdev, r_cpu.sdev, rtol=GPU_FP32.rtol, atol=GPU_FP32.atol,
        )

    def test_k_near_min_dimension(self):
        """n_components close to min(n, p) (sketch width capped at min(n, p))."""
        from pystatistics.core.compute.tolerances import GPU_FP32
        rng = np.random.default_rng(5)
        X = rng.standard_normal((200, 15))
        r_cpu = pca(X, n_components=15, backend="cpu")
        r_gpu = pca(X, n_components=15, backend="gpu")
        assert r_gpu.sdev.shape == (15,)
        np.testing.assert_allclose(
            r_gpu.sdev, r_cpu.sdev, rtol=GPU_FP32.rtol, atol=GPU_FP32.atol,
        )

    def test_seed_reproducibility(self):
        """Same seed → bitwise-identical result; different seed → it still
        matches the reference but is a distinct draw."""
        rng = np.random.default_rng(6)
        X = _planted_lowrank(rng, 4000, 100, 6)
        a = pca(X, n_components=5, backend="gpu", seed=7)
        b = pca(X, n_components=5, backend="gpu", seed=7)
        np.testing.assert_array_equal(a.sdev, b.sdev)
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.rotation, b.rotation)

    # ---- Failure cases ----

    def test_explicit_svd_raises_loud(self, iris_like_data):
        with pytest.raises(RuntimeError, match="Metal"):
            pca(iris_like_data, backend="gpu", solver="svd")

    def test_explicit_gram_raises_loud(self, iris_like_data):
        with pytest.raises(RuntimeError, match="Metal"):
            pca(iris_like_data, backend="gpu", solver="gram")

    def test_invalid_solver_raises(self, iris_like_data):
        with pytest.raises(ValidationError, match="method"):
            pca(iris_like_data, backend="gpu", solver="quantum")

    def test_ill_conditioned_refuses(self):
        """cond(X) past the fp32 gate refuses (NumericalError) unless force."""
        from pystatistics.core.exceptions import NumericalError
        rng = np.random.default_rng(8)
        X = rng.standard_normal((1000, 20))
        X[:, 0] = X[:, 1] + 1e-7 * rng.standard_normal(1000)   # near-collinear
        with pytest.raises(NumericalError, match="cond"):
            pca(X, backend="gpu", solver="randomized")

    def test_ill_conditioned_force_bypasses(self):
        """force=True bypasses the condition gate and completes."""
        rng = np.random.default_rng(9)
        X = rng.standard_normal((1000, 20))
        X[:, 0] = X[:, 1] + 1e-7 * rng.standard_normal(1000)
        r = pca(X, n_components=20, backend="gpu", solver="randomized", force=True)
        assert r.sdev.shape == (20,)


# ===========================================================================
# Factor Analysis Tests
# ===========================================================================

class TestFactorAnalysis:
    """Tests for maximum likelihood factor analysis."""

    def test_loadings_shape(self, factor_data):
        """Loadings shape is (p, n_factors)."""
        result = factor_analysis(factor_data, n_factors=2)
        assert result.loadings.shape == (6, 2)

    def test_uniquenesses_range(self, factor_data):
        """Uniquenesses between 0 and 1."""
        result = factor_analysis(factor_data, n_factors=2)
        assert np.all(result.uniquenesses > 0)
        assert np.all(result.uniquenesses < 1)

    def test_communalities_equal_one_minus_uniquenesses(self, factor_data):
        """Communalities = 1 - uniquenesses."""
        result = factor_analysis(factor_data, n_factors=2)
        np.testing.assert_allclose(
            result.communalities, 1.0 - result.uniquenesses, atol=1e-14
        )

    def test_communalities_bounded(self, factor_data):
        """Communalities <= 1."""
        result = factor_analysis(factor_data, n_factors=2)
        assert np.all(result.communalities <= 1.0 + 1e-10)

    def test_chi_squared_nonnegative(self, factor_data):
        """Chi-squared statistic >= 0."""
        result = factor_analysis(factor_data, n_factors=2)
        if result.chi_sq is not None:
            assert result.chi_sq >= 0

    def test_dof_correct(self, factor_data):
        """Degrees of freedom formula is correct."""
        p = 6
        m = 2
        expected_dof = ((p - m) ** 2 - (p + m)) // 2
        result = factor_analysis(factor_data, n_factors=m)
        assert result.dof == expected_dof

    def test_too_many_factors_raises(self, factor_data):
        """Too many factors raises ValueError."""
        with pytest.raises(ValidationError, match="too many"):
            factor_analysis(factor_data, n_factors=6)

    def test_varimax_orthogonal_rotation_matrix(self, factor_data):
        """Varimax rotation produces orthogonal rotation matrix."""
        result = factor_analysis(factor_data, n_factors=2, rotation="varimax")
        if result.rotation_matrix is not None:
            R = result.rotation_matrix
            product = R.T @ R
            np.testing.assert_allclose(product, np.eye(R.shape[1]), atol=1e-6)

    def test_promax_different_from_varimax(self, factor_data):
        """Promax rotation produces different loadings than varimax."""
        result_vm = factor_analysis(factor_data, n_factors=2, rotation="varimax")
        result_pm = factor_analysis(factor_data, n_factors=2, rotation="promax")
        # Loadings should differ (unless data is perfectly simple-structured)
        assert not np.allclose(result_vm.loadings, result_pm.loadings, atol=1e-4)

    def test_rotation_none(self, factor_data):
        """Rotation='none' returns unrotated loadings."""
        result = factor_analysis(factor_data, n_factors=2, rotation="none")
        assert result.rotation_matrix is None
        assert result.rotation_method == "none"

    def test_convergence_flag(self, factor_data):
        """Convergence flag is set."""
        result = factor_analysis(factor_data, n_factors=2)
        assert isinstance(result.converged, bool)

    def test_well_structured_data_recovers_factors(self, factor_data):
        """Well-structured data recovers known factor structure."""
        result = factor_analysis(factor_data, n_factors=2, rotation="varimax")
        # Variables 0,1,2 should load heavily on one factor
        # Variables 3,4,5 should load heavily on the other
        loadings = np.abs(result.loadings)
        # Each variable should have a dominant factor
        dominant = np.argmax(loadings, axis=1)
        # First three should share a factor, last three another
        assert dominant[0] == dominant[1]
        assert dominant[3] == dominant[4]
        assert dominant[0] != dominant[3]

    def test_invalid_method_raises(self, factor_data):
        """Invalid method raises ValidationError."""
        with pytest.raises(ValidationError, match="only 'ml'"):
            factor_analysis(factor_data, n_factors=2, method="uls")

    def test_invalid_rotation_raises(self, factor_data):
        """Invalid rotation raises ValidationError."""
        with pytest.raises(ValidationError, match="must be one of"):
            factor_analysis(factor_data, n_factors=2, rotation="oblimin")

    def test_result_is_frozen(self, factor_data):
        """FactorSolution is frozen dataclass."""
        result = factor_analysis(factor_data, n_factors=2)
        with pytest.raises(AttributeError):
            result.loadings = np.zeros((6, 2))  # type: ignore[misc]

    def test_summary_runs(self, factor_data):
        """summary() produces a non-empty string."""
        result = factor_analysis(factor_data, n_factors=2)
        s = result.summary()
        assert isinstance(s, str)
        assert "Factor" in s

    def test_names_propagate(self, factor_data):
        """Variable names propagate."""
        names = [f"Var{i}" for i in range(6)]
        result = factor_analysis(factor_data, n_factors=2, names=names)
        assert result.var_names == tuple(names)

    def test_single_factor(self, factor_data):
        """Single factor extraction works."""
        result = factor_analysis(factor_data, n_factors=1, rotation="none")
        assert result.loadings.shape == (6, 1)
        assert result.n_factors == 1

    def test_n_obs_n_vars(self, factor_data):
        """n_obs and n_vars are set correctly."""
        result = factor_analysis(factor_data, n_factors=2)
        assert result.n_obs == 200
        assert result.n_vars == 6

    # ---- F1: default varimax converges on clean multi-factor data ----

    @pytest.mark.parametrize("seed", [14, 24, 48, 62])
    def test_clean_multifactor_default_varimax_converges(self, seed):
        """Default ``factor_analysis(X, n_factors=2)`` converges on clean
        simple-structure data and recovers the block structure.

        These seeds drive the raw ML loadings into a varimax criterion that
        plateaus; the previous absolute stop-criterion raised
        ConvergenceError here (>1000 iters), R's relative test converges.
        """
        X = clean_2factor(seed)
        result = factor_analysis(X, n_factors=2)  # rotation defaults to varimax
        assert result.rotation_method == "varimax"
        # Simple structure: vars 0-3 share one factor, 4-7 the other.
        dominant = np.argmax(np.abs(result.loadings), axis=1)
        assert len(set(dominant[:4])) == 1
        assert len(set(dominant[4:])) == 1
        assert dominant[0] != dominant[4]
        # Uniquenesses in the spec'd 0.3-0.5 band, none degenerate.
        assert np.all(result.uniquenesses > 0.2)
        assert np.all(result.uniquenesses < 0.6)

    # ---- F2: uniqueness floor (`lower`) ----

    def test_lower_floors_uniquenesses(self):
        """A Heywood-prone variable is floored at the default ``lower``."""
        X = _heywood_data()
        result = factor_analysis(X, n_factors=1, rotation="none")
        assert np.all(result.uniquenesses >= 0.005 - 1e-9)
        # The floor is actually binding on the offending variable.
        assert np.isclose(result.uniquenesses.min(), 0.005, atol=1e-6)

    def test_lower_admits_heywood_when_relaxed(self):
        """Relaxing ``lower`` lets the Heywood uniqueness collapse toward 0,
        confirming the floor (not some other clamp) is what constrains it."""
        X = _heywood_data()
        floored = factor_analysis(X, n_factors=1, rotation="none", lower=0.005)
        relaxed = factor_analysis(X, n_factors=1, rotation="none", lower=1e-7)
        assert floored.uniquenesses.min() > relaxed.uniquenesses.min()
        assert relaxed.uniquenesses.min() < 1e-3
        # The floored optimum has the higher (more constrained) objective.
        assert floored.objective >= relaxed.objective - 1e-9

    def test_custom_lower_is_respected(self):
        """A non-default ``lower`` floors the uniquenesses at that value."""
        X = _heywood_data()
        result = factor_analysis(X, n_factors=1, rotation="none", lower=0.05)
        assert np.all(result.uniquenesses >= 0.05 - 1e-9)
        assert np.isclose(result.uniquenesses.min(), 0.05, atol=1e-6)

    def test_lower_recorded_in_info(self):
        """The resolved ``lower`` is exposed on the result info dict."""
        X = _heywood_data()
        result = factor_analysis(X, n_factors=1, rotation="none", lower=0.01)
        assert result.info["lower"] == 0.01

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_lower_raises(self, factor_data, bad):
        """``lower`` outside (0, 1) fails loud."""
        with pytest.raises(ValidationError, match="lower"):
            factor_analysis(factor_data, n_factors=2, lower=bad)


# ===========================================================================
# R cross-validation: factor analysis vs stats::factanal
# ===========================================================================

def _heywood_data() -> np.ndarray:
    """Single-factor data with one near-perfectly-explained variable.

    Variable 0 is the factor plus negligible noise, so its uniqueness wants
    to go to ~0 (a Heywood case) and is floored by ``lower``.
    """
    rng = np.random.default_rng(1)
    n = 150
    f = rng.standard_normal(n)
    return np.column_stack([
        f + 0.01 * rng.standard_normal(n),
        0.8 * f + 0.6 * rng.standard_normal(n),
        0.7 * f + 0.7 * rng.standard_normal(n),
        0.6 * f + 0.8 * rng.standard_normal(n),
        0.5 * f + 0.9 * rng.standard_normal(n),
    ])


@_requires_r
class TestFactorAnalysisVsR:
    """Confirm the multi-factor FA-vs-factanal agreement deferred at 4.4.0."""

    def test_multifactor_varimax_matches_factanal(self, tmp_path):
        """Clean 2-factor: uniquenesses + ML objective match factanal to a
        tight tolerance, loadings up to rotation/sign."""
        X = clean_2factor(24)
        py = factor_analysis(X, n_factors=2, rotation="varimax")
        r_load, r_uniq, r_crit = _r_factanal(
            X, tmp_path, n_factors=2, rotation="varimax"
        )
        assert np.max(np.abs(np.asarray(py.uniquenesses) - r_uniq)) < 1e-4
        assert abs(py.objective - r_crit) < 1e-6
        assert _loading_diff_up_to_rotation(np.asarray(py.loadings), r_load) < 1e-3

    def test_iris_1factor_matches_factanal_with_lower(self, tmp_path):
        """iris 1-factor Heywood case matches factanal at the shared default
        ``lower=0.005`` -- Petal.Length floored, objective ~0.585 (not 0.566)."""
        iris_csv = os.path.join(str(tmp_path), "iris.csv")
        subprocess.run(
            ["Rscript", "-e",
             f'write.table(iris[,1:4], "{iris_csv}", sep=",", '
             'row.names=FALSE, col.names=FALSE)'],
            check=True, capture_output=True, text=True,
        )
        iris = np.loadtxt(iris_csv, delimiter=",")
        py = factor_analysis(iris, n_factors=1, rotation="none", lower=0.005)
        r_load, r_uniq, r_crit = _r_factanal(
            iris, tmp_path, n_factors=1, rotation="none", lower=0.005
        )
        assert np.max(np.abs(np.asarray(py.uniquenesses) - r_uniq)) < 1e-4
        assert abs(py.objective - r_crit) < 1e-4
        # Matches R's floored optimum, not the unconstrained Heywood one.
        assert abs(py.objective - 0.585) < 0.01
        assert np.isclose(np.asarray(py.uniquenesses).min(), 0.005, atol=1e-4)


# ===========================================================================
# Rotation Tests
# ===========================================================================

class TestRotation:
    """Tests for varimax and promax rotation."""

    def test_varimax_orthogonal(self):
        """Varimax rotation matrix is orthogonal: R @ R.T ~ I."""
        rng = np.random.default_rng(77)
        loadings = rng.standard_normal((8, 3))
        _, R = varimax(loadings)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_varimax_preserves_communalities(self):
        """Varimax preserves communalities (row sums of squared loadings)."""
        rng = np.random.default_rng(77)
        loadings = rng.standard_normal((8, 3))
        communalities_before = np.sum(loadings ** 2, axis=1)
        rotated, _ = varimax(loadings)
        communalities_after = np.sum(rotated ** 2, axis=1)
        np.testing.assert_allclose(
            communalities_after, communalities_before, rtol=1e-10
        )

    def test_promax_power_parameter(self):
        """Promax target power parameter works (different m gives different results)."""
        rng = np.random.default_rng(77)
        loadings = rng.standard_normal((8, 3))
        rotated_m3, _ = promax(loadings, m=3)
        rotated_m6, _ = promax(loadings, m=6)
        assert not np.allclose(rotated_m3, rotated_m6, atol=1e-4)

    def test_identity_rotation_simple_structure(self):
        """Near-identity rotation when loadings already have simple structure."""
        # Create loadings with perfect simple structure
        loadings = np.array([
            [1.0, 0.0],
            [0.9, 0.0],
            [0.0, 1.0],
            [0.0, 0.8],
        ])
        rotated, R = varimax(loadings)
        # Rotation matrix should be close to identity (or a permutation)
        # Check that R @ R.T is identity (it's orthogonal)
        np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-10)
        # Rotated loadings should preserve the simple structure
        # (each variable loads primarily on one factor)
        for i in range(4):
            max_loading = np.max(np.abs(rotated[i]))
            min_loading = np.min(np.abs(rotated[i]))
            assert max_loading > 5 * min_loading, (
                f"Row {i}: simple structure not preserved after rotation"
            )

    def test_varimax_single_factor_noop(self):
        """Varimax with single factor is essentially a no-op."""
        loadings = np.array([[0.9], [0.8], [0.7]])
        rotated, R = varimax(loadings)
        np.testing.assert_allclose(np.abs(rotated), np.abs(loadings), atol=1e-10)

    def test_promax_single_factor(self):
        """Promax with single factor works without error."""
        loadings = np.array([[0.9], [0.8], [0.7]])
        rotated, R = promax(loadings)
        assert rotated.shape == (3, 1)

    def test_varimax_relative_convergence_on_plateau_loadings(self):
        """Loadings whose criterion plateaus (the F1 regime) converge under
        the relative test where the old absolute test would not.

        These are the raw ML loadings from ``clean_2factor(24)`` -- both
        factors loaded near-equally, so the varimax criterion creeps up by
        ~1%/iter and never reaches an absolute 1e-6 step in 1000 iters."""
        raw = factor_analysis(
            clean_2factor(24), n_factors=2, rotation="none"
        ).loadings
        rotated, R = varimax(np.asarray(raw))
        np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-8)
        # Communalities preserved by the orthogonal rotation.
        np.testing.assert_allclose(
            np.sum(rotated ** 2, axis=1),
            np.sum(np.asarray(raw) ** 2, axis=1),
            rtol=1e-8,
        )

    def test_varimax_nonconvergence_raises(self):
        """Genuine non-convergence still fails loud (Rule 1): a tiny
        max_iter on loadings that need real rotation raises."""
        raw = np.asarray(
            factor_analysis(clean_2factor(24), n_factors=2, rotation="none").loadings
        )
        with pytest.raises(ConvergenceError, match="did not converge"):
            varimax(raw, max_iter=2)


class TestPCADeviceResident:
    """Tests for device-resident PCASolution (1.9.0 feature).

    A PCASolution with ``.device != 'cpu'`` holds its numeric fields as
    ``torch.Tensor`` instances on the fit's device. This is the
    opt-in path that skips the D2H copy of the scores matrix —
    useful in multi-step GPU pipelines where PCA output feeds
    directly into a subsequent GPU computation.
    """

    def _gpu_available(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        # PCA GPU is CUDA-only: the SVD / symmetric eigendecomposition
        # PCA needs has no Metal kernel and silently falls back to CPU on
        # MPS, so there is no genuine GPU PCA on Apple Silicon (the
        # backend raises there — see test_pca_gpu_on_mps_raises).
        return torch.cuda.is_available()

    def test_default_result_is_numpy_backed(self):
        """Back-compat: default GPU path returns a numpy-backed result.

        ``device_resident=False`` (the default) must behave exactly as
        before 1.9.0 — fields are numpy ndarrays, ``.device`` reports
        ``'cpu'``. Breaking this silently would be a Rule 1 violation.
        """
        if not self._gpu_available():
            pytest.skip("no GPU available")
        X = np.random.RandomState(0).randn(200, 5).astype(np.float32)
        r = pca(X, backend="gpu")
        assert r.device == "cpu"
        assert isinstance(r.x, np.ndarray)
        assert isinstance(r.sdev, np.ndarray)

    def test_device_resident_fields_are_tensors(self):
        """Opt-in: ``device_resident=True`` keeps fields on GPU."""
        if not self._gpu_available():
            pytest.skip("no GPU available")
        import torch
        X = np.random.RandomState(0).randn(200, 5).astype(np.float32)
        r = pca(X, backend="gpu", device_resident=True)
        assert r.device != "cpu"
        assert isinstance(r.x, torch.Tensor)
        assert isinstance(r.sdev, torch.Tensor)
        assert isinstance(r.rotation, torch.Tensor)
        assert r.x.device.type in ("cuda", "mps")

    def test_to_numpy_matches_default_path(self):
        """``result.to_numpy()`` on a device-resident fit returns the
        same PCASolution values as the default numpy-return path."""
        if not self._gpu_available():
            pytest.skip("no GPU available")
        X = np.random.RandomState(0).randn(500, 10).astype(np.float32)
        r_np = pca(X, backend="gpu")
        r_dr = pca(X, backend="gpu", device_resident=True)
        r_dr_cpu = r_dr.to_numpy()
        np.testing.assert_array_equal(r_np.sdev, r_dr_cpu.sdev)
        np.testing.assert_array_equal(r_np.rotation, r_dr_cpu.rotation)
        np.testing.assert_array_equal(r_np.x, r_dr_cpu.x)
        assert r_dr_cpu.device == "cpu"

    def test_to_numpy_idempotent_on_numpy_result(self):
        """On a numpy-backed result, ``to_numpy`` returns ``self``."""
        X = np.random.RandomState(0).randn(100, 4)
        r = pca(X, backend="cpu")
        assert r.to_numpy() is r

    def test_device_resident_ignored_on_cpu(self):
        """``device_resident=True`` on CPU path is a no-op (result is
        still numpy-backed; no torch dependency incurred)."""
        X = np.random.RandomState(0).randn(100, 4)
        r = pca(X, backend="cpu", device_resident=True)
        assert r.device == "cpu"
        assert isinstance(r.x, np.ndarray)

    def test_explained_variance_ratio_polymorphic(self):
        """``explained_variance_ratio`` returns tensor on tensor-backed
        results, numpy on numpy-backed — same dtype as ``sdev``."""
        if not self._gpu_available():
            pytest.skip("no GPU available")
        import torch
        X = np.random.RandomState(0).randn(300, 8).astype(np.float32)
        r_np = pca(X, backend="gpu")
        r_dr = pca(X, backend="gpu", device_resident=True)
        assert isinstance(r_np.explained_variance_ratio, np.ndarray)
        assert isinstance(r_dr.explained_variance_ratio, torch.Tensor)
        # Numeric content matches.
        np.testing.assert_allclose(
            r_np.explained_variance_ratio,
            r_dr.explained_variance_ratio.cpu().numpy(),
            rtol=1e-6, atol=1e-8,
        )

    def test_summary_works_on_tensor_backed(self):
        """``summary()`` materialises internally so it works for both
        backings without the caller worrying about the device."""
        if not self._gpu_available():
            pytest.skip("no GPU available")
        X = np.random.RandomState(0).randn(200, 6).astype(np.float32)
        r = pca(X, backend="gpu", device_resident=True)
        s = r.summary()
        assert "Importance of components" in s
        assert "PC1" in s

    def test_to_moves_between_devices(self):
        """``result.to('cpu')`` is equivalent to ``to_numpy()``; round-
        tripping ``to('cuda').to('cpu')`` is a no-op on values."""
        if not self._gpu_available():
            pytest.skip("no GPU available")
        import torch
        if not torch.cuda.is_available():
            pytest.skip(".to('cuda') requires CUDA")
        X = np.random.RandomState(0).randn(150, 5).astype(np.float32)
        r_cpu = pca(X, backend="cpu")
        r_gpu = r_cpu.to("cuda")
        assert r_gpu.device == "cuda"
        r_back = r_gpu.to("cpu")
        np.testing.assert_array_equal(r_cpu.x, r_back.x)
        np.testing.assert_array_equal(r_cpu.sdev, r_back.sdev)
