"""
Tests for the multivariate analysis module.

Covers PCA, factor analysis, and rotation methods.
"""

import numpy as np
import pytest

from pystatistics.multivariate import pca, factor_analysis, PCAResult, FactorResult
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
        """PCAResult is frozen dataclass."""
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
        assert isinstance(result, PCAResult)

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
        return torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

    def test_gpu_unavailable_raises_explicitly(self, iris_like_data, monkeypatch):
        """backend='gpu' must raise when no GPU is available, not silently
        fall back to CPU (Rule 1: no hidden fallbacks when caller is
        explicit)."""
        from pystatistics.core.compute import device as dev_mod

        def no_gpu(*_a, **_k):
            return None
        monkeypatch.setattr(dev_mod, "detect_gpu", no_gpu)
        with pytest.raises(RuntimeError, match="no GPU"):
            pca(iris_like_data, backend="gpu")

    def test_invalid_backend_raises(self, iris_like_data):
        with pytest.raises(ValidationError, match="backend"):
            pca(iris_like_data, backend="quantum")

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
            backend="gpu", use_fp64=True,
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
            backend="gpu", use_fp64=False,
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
            pca(iris_like_data, backend="gpu", method="magic")

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
        r_svd = pca(X, backend="gpu", method="svd")
        r_gram = pca(X, backend="gpu", method="gram")
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
            pca(X, backend="gpu", method="gram")

    def test_gram_force_bypasses_condition_check(self):
        """force=True bypasses the condition gate — the fit completes
        even on ill-conditioned data, though the numerical result is
        unreliable."""
        if not self._gpu_available():
            pytest.skip("no GPU available")
        rng = np.random.default_rng(2)
        X = rng.standard_normal((1000, 20))
        X[:, 0] = X[:, 1] + 1e-7 * rng.standard_normal(1000)
        r = pca(X, backend="gpu", method="gram", force=True)
        assert r.sdev.shape == (20,)

    def test_auto_falls_back_to_svd_on_ill_conditioned(self):
        """method='auto' must silently fall back to SVD when Gram's
        condition check fails — that is the explicit contract of
        'auto' (unlike method='gram' which raises)."""
        if not self._gpu_available():
            pytest.skip("no GPU available")
        rng = np.random.default_rng(3)
        X = rng.standard_normal((1000, 20))
        X[:, 0] = X[:, 1] + 1e-7 * rng.standard_normal(1000)
        r_auto = pca(X, backend="gpu", method="auto")
        r_svd = pca(X, backend="gpu", method="svd")
        # auto should have fallen back to SVD, so results are identical
        # (same code path, same data).
        np.testing.assert_allclose(r_auto.sdev, r_svd.sdev, rtol=1e-14)


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
        """FactorResult is frozen dataclass."""
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
