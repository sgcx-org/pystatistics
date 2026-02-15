"""
Tests for the batched multi-problem OLS solver.

Validates that solving X @ B = Y with shared X gives the same
coefficients as solving each y_i individually. Tests both CPU
and GPU paths.
"""

import numpy as np
import pytest

from pystatistics.core.compute.linalg.batched import batched_ols_solve


class TestBatchedOLSCPU:
    """Tests for CPU batched solver."""

    def test_single_response(self):
        """Batched solver with k=1 matches standard OLS."""
        rng = np.random.default_rng(42)
        n, p = 50, 3
        X = rng.normal(0, 1, (n, p))
        y = X @ np.array([1.0, 2.0, 3.0]) + rng.normal(0, 0.1, n)

        # Batched solve
        Y = y[:, np.newaxis]
        B = batched_ols_solve(X, Y, device='cpu')

        # Reference: np.linalg.lstsq
        beta_ref, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        np.testing.assert_allclose(B[:, 0], beta_ref, rtol=1e-10)

    def test_multiple_responses(self):
        """Batched solver with k>1 matches individual solves."""
        rng = np.random.default_rng(42)
        n, p, k = 100, 4, 50
        X = rng.normal(0, 1, (n, p))
        true_B = rng.normal(0, 1, (p, k))
        Y = X @ true_B + rng.normal(0, 0.1, (n, k))

        # Batched solve
        B = batched_ols_solve(X, Y, device='cpu')

        # Reference: solve each column individually
        for j in range(k):
            beta_ref, _, _, _ = np.linalg.lstsq(X, Y[:, j], rcond=None)
            np.testing.assert_allclose(
                B[:, j], beta_ref, rtol=1e-10,
                err_msg=f"Column {j} mismatch",
            )

    def test_many_replicates(self):
        """Batched solver handles k=10000 replicates."""
        rng = np.random.default_rng(42)
        n, p, k = 50, 3, 10000
        X = rng.normal(0, 1, (n, p))
        Y = rng.normal(0, 1, (n, k))

        B = batched_ols_solve(X, Y, device='cpu')

        assert B.shape == (p, k)

        # Spot-check a few columns
        for j in [0, 100, 5000, 9999]:
            beta_ref, _, _, _ = np.linalg.lstsq(X, Y[:, j], rcond=None)
            np.testing.assert_allclose(B[:, j], beta_ref, rtol=1e-10)

    def test_intercept_model(self):
        """Works with intercept column."""
        rng = np.random.default_rng(42)
        n = 30
        x = rng.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x])
        y = 2.0 + 3.0 * x + rng.normal(0, 0.1, n)

        B = batched_ols_solve(X, y[:, np.newaxis], device='cpu')
        assert B[0, 0] == pytest.approx(2.0, abs=0.1)
        assert B[1, 0] == pytest.approx(3.0, abs=0.1)

    def test_1d_y_input(self):
        """1D y input is automatically reshaped."""
        rng = np.random.default_rng(42)
        n, p = 20, 2
        X = rng.normal(0, 1, (n, p))
        y = rng.normal(0, 1, n)

        B = batched_ols_solve(X, y, device='cpu')
        assert B.shape == (p, 1)

    def test_well_conditioned(self):
        """Well-conditioned problem gives accurate results."""
        # Orthogonal X
        n = 100
        X = np.eye(n, 5)
        Y = np.arange(n, dtype=float)[:, np.newaxis]

        B = batched_ols_solve(X, Y, device='cpu')
        # B should be the first 5 elements of Y
        np.testing.assert_allclose(B[:, 0], np.arange(5, dtype=float), rtol=1e-12)

    def test_bootstrap_use_case(self):
        """Simulates the residual bootstrap regression use case."""
        rng = np.random.default_rng(42)
        n, p = 50, 3
        X = rng.normal(0, 1, (n, p))
        true_beta = np.array([1.0, 2.0, 3.0])
        noise = rng.normal(0, 0.5, n)
        y = X @ true_beta + noise

        # OLS estimate + residuals
        beta_ols, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        fitted = X @ beta_ols
        residuals = y - fitted

        # Residual bootstrap: X stays fixed, resample residuals
        R = 1000
        Y_boot = np.empty((n, R))
        for b in range(R):
            resid_boot = rng.choice(residuals, size=n, replace=True)
            Y_boot[:, b] = fitted + resid_boot

        B = batched_ols_solve(X, Y_boot, device='cpu')
        assert B.shape == (p, R)

        # The mean of bootstrap coefficients should be close to OLS estimate
        mean_boot = np.mean(B, axis=1)
        np.testing.assert_allclose(mean_boot, beta_ols, rtol=0.05)


class TestBatchedOLSValidation:
    """Tests for input validation."""

    def test_dimension_mismatch(self):
        """X and Y row counts must match."""
        X = np.ones((10, 3))
        Y = np.ones((5, 2))
        with pytest.raises(ValueError, match="rows"):
            batched_ols_solve(X, Y)

    def test_X_must_be_2d(self):
        """X must be 2D."""
        with pytest.raises(ValueError, match="2D"):
            batched_ols_solve(np.ones(10), np.ones(10))


class TestBatchedOLSGPU:
    """Tests for GPU batched solver (skipped if no GPU)."""

    @pytest.fixture(autouse=True)
    def check_gpu(self):
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            if not (has_cuda or has_mps):
                pytest.skip("No GPU available")
            self.device = 'cuda' if has_cuda else 'mps'
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_gpu_matches_cpu(self):
        """GPU batched solver matches CPU results."""
        rng = np.random.default_rng(42)
        n, p, k = 100, 4, 500
        X = rng.normal(0, 1, (n, p))
        Y = rng.normal(0, 1, (n, k))

        B_cpu = batched_ols_solve(X, Y, device='cpu')
        B_gpu = batched_ols_solve(X, Y, device=self.device)

        # GPU is FP32, so tolerances are wider
        np.testing.assert_allclose(B_gpu, B_cpu, rtol=1e-4, atol=1e-5)

    def test_gpu_many_replicates(self):
        """GPU handles many replicates efficiently."""
        rng = np.random.default_rng(42)
        n, p, k = 50, 3, 5000
        X = rng.normal(0, 1, (n, p))
        Y = rng.normal(0, 1, (n, k))

        B = batched_ols_solve(X, Y, device=self.device)
        assert B.shape == (p, k)

        # Spot-check against CPU
        B_cpu = batched_ols_solve(X, Y[:, :3], device='cpu')
        np.testing.assert_allclose(B[:, :3], B_cpu, rtol=1e-4)
