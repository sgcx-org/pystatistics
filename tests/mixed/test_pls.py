"""Tests for the Penalized Least Squares solver."""

import numpy as np
import pytest

from pystatistics.mixed._random_effects import (
    parse_random_effects, build_z_matrix, build_lambda,
)
from pystatistics.mixed._pls import solve_pls


class TestSolvePLS:
    """Tests for the PLS inner solver."""

    def test_basic_random_intercept(self):
        """PLS produces reasonable estimates for a simple random intercept."""
        rng = np.random.default_rng(42)
        n_groups = 5
        n_per = 20
        n = n_groups * n_per

        # True parameters
        beta_true = np.array([10.0, 2.0])
        sigma_re = 3.0
        sigma_resid = 1.0

        group_effects = rng.normal(0, sigma_re, n_groups)
        group = np.repeat(np.arange(n_groups), n_per)
        x = rng.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x])
        y = X @ beta_true + group_effects[group] + rng.normal(0, sigma_resid, n)

        # Build Z
        groups = {'g': group}
        specs = parse_random_effects(groups, None, None, n)
        Z = build_z_matrix(specs)

        # Use theta=1 (moderate random effect)
        theta = np.array([1.0])
        Lambda = build_lambda(theta, specs)

        result = solve_pls(X, Z, y, Lambda, reml=True)

        # Fixed effects should be close to truth
        np.testing.assert_allclose(result.beta, beta_true, atol=1.5)

        # Residuals + fitted should reconstruct y
        np.testing.assert_allclose(result.fitted + result.residuals, y, atol=1e-10)

        # Penalized RSS should be positive
        assert result.pwrss > 0

        # sigma_sq should be positive
        assert result.sigma_sq > 0

    def test_penalty_shrinks_b_to_zero(self):
        """When Λ → 0, random effects b → 0 (infinite penalty)."""
        rng = np.random.default_rng(123)
        n_groups = 3
        n_per = 10
        n = n_groups * n_per

        group = np.repeat(np.arange(n_groups), n_per)
        x = rng.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x])
        y = 5.0 + 2.0 * x + rng.normal(0, 1, n)

        groups = {'g': group}
        specs = parse_random_effects(groups, None, None, n)
        Z = build_z_matrix(specs)

        # Very small theta → strong penalty
        theta = np.array([0.01])
        Lambda = build_lambda(theta, specs)

        result = solve_pls(X, Z, y, Lambda, reml=True)

        # b should be near zero
        np.testing.assert_allclose(result.b, 0.0, atol=0.5)

    def test_weighted_pls(self):
        """PLS with unit weights should equal unweighted PLS."""
        rng = np.random.default_rng(456)
        n = 30
        group = np.repeat(np.arange(3), 10)
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = X @ [5, 2] + rng.normal(0, 1, n)

        groups = {'g': group}
        specs = parse_random_effects(groups, None, None, n)
        Z = build_z_matrix(specs)
        Lambda = build_lambda(np.array([1.0]), specs)

        result_no_w = solve_pls(X, Z, y, Lambda)
        result_w = solve_pls(X, Z, y, Lambda, weights=np.ones(n))

        np.testing.assert_allclose(result_w.beta, result_no_w.beta, atol=1e-10)
        np.testing.assert_allclose(result_w.b, result_no_w.b, atol=1e-10)

    def test_l_factor_shape(self):
        """L factor from PLS has correct shape (q × q)."""
        n = 20
        group = np.repeat(np.arange(4), 5)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)

        groups = {'g': group}
        specs = parse_random_effects(groups, None, None, n)
        Z = build_z_matrix(specs)
        q = Z.shape[1]
        Lambda = build_lambda(np.array([1.0]), specs)

        result = solve_pls(X, Z, y, Lambda)
        assert result.L.shape == (q, q)

    def test_rx_factor_shape(self):
        """RX factor from PLS has correct shape (p × p)."""
        n = 20
        p = 3
        group = np.repeat(np.arange(4), 5)
        X = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
        y = np.random.randn(n)

        groups = {'g': group}
        specs = parse_random_effects(groups, None, None, n)
        Z = build_z_matrix(specs)
        Lambda = build_lambda(np.array([1.0]), specs)

        result = solve_pls(X, Z, y, Lambda)
        assert result.RX.shape == (p, p)
