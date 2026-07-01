"""Tests for fixed-effect standard errors (F2a — form A p×p Schur).

The standard errors must be machine-identical to the dense n×n
Var(β̂) = σ²(X'V*⁻¹X)⁻¹ formulation they replaced, while never forming
the n×n V* matrix.
"""

import numpy as np
import pytest

from pystatistics.mixed import lmm
from pystatistics.mixed._random_effects import (
    parse_random_effects, build_z_matrix, build_lambda,
)
from pystatistics.mixed._pls import solve_pls
from pystatistics.mixed.solvers import _compute_se


def _dense_se(pls, X, Z, Lambda):
    """Reference: the old dense n×n SE, Var(β̂) = σ²(X'V*⁻¹X)⁻¹."""
    n = X.shape[0]
    V_star = Z @ Lambda @ Lambda.T @ Z.T + np.eye(n)
    C = np.linalg.inv(X.T @ np.linalg.solve(V_star, X))
    vcov = pls.sigma_sq * C
    return np.sqrt(np.maximum(np.diag(vcov), 0.0))


def _make_model(n_groups=8, n_per=15, p=3, q_terms=("1",), seed=0):
    rng = np.random.default_rng(seed)
    n = n_groups * n_per
    group = np.repeat(np.arange(n_groups), n_per)
    X = np.column_stack([np.ones(n)] + [rng.standard_normal(n) for _ in range(p - 1)])
    re = rng.standard_normal(n_groups) * 1.5
    y = X @ np.arange(1, p + 1) + re[group] + rng.standard_normal(n)
    # Random-slope variables reuse the fixed-effect covariate columns by name.
    random_data = {}
    for t in q_terms:
        if t != "1":
            col = int(t[1:])  # "X1" -> column 1
            random_data[t] = X[:, col]
    specs = parse_random_effects({"g": group}, {"g": list(q_terms)}, random_data, n)
    Z = build_z_matrix(specs)
    return X, Z, y, specs


class TestComputeSEFormA:
    def test_form_a_matches_dense_intercept(self):
        """Form A SE == dense n×n SE to machine precision (random intercept)."""
        X, Z, y, specs = _make_model(seed=1)
        Lambda = build_lambda(np.array([0.8]), specs)
        pls = solve_pls(X, Z, y, Lambda, reml=True)

        se_form_a = _compute_se(pls)
        se_dense = _dense_se(pls, X, Z, Lambda)

        np.testing.assert_allclose(se_form_a, se_dense, rtol=0, atol=1e-12)

    def test_form_a_matches_dense_slope(self):
        """Form A SE == dense n×n SE for a random intercept + slope model."""
        X, Z, y, specs = _make_model(p=2, q_terms=("1", "X1"), seed=2)
        Lambda = build_lambda(np.array([0.9, 0.1, 0.5]), specs)
        pls = solve_pls(X, Z, y, Lambda, reml=True)

        se_form_a = _compute_se(pls)
        se_dense = _dense_se(pls, X, Z, Lambda)

        np.testing.assert_allclose(se_form_a, se_dense, rtol=0, atol=1e-11)

    def test_se_positive_and_finite(self):
        """SEs from a full fit are positive and finite."""
        X, Z, y, specs = _make_model(seed=3)
        result = lmm(y, X, groups={"g": np.repeat(np.arange(8), 15)})
        assert np.all(result.standard_errors > 0)
        assert np.all(np.isfinite(result.standard_errors))

    def test_singular_rx_falls_back_to_pinv(self):
        """A collinear fixed-effects design routes through the pinv fallback
        without raising."""
        X, Z, y, specs = _make_model(p=3, seed=4)
        X[:, 2] = X[:, 1]  # exact collinearity → singular RX
        Lambda = build_lambda(np.array([0.8]), specs)
        # solve_pls itself raises on singular RtR; SE fallback is exercised by
        # constructing a PLSResult with a deliberately singular RX.
        pls = solve_pls(X[:, :2], Z, y, Lambda, reml=True)
        from pystatistics.mixed._pls import PLSResult
        singular_RX = np.array([[1.0, 0.0], [1.0, 0.0]])  # rank-deficient
        bad = PLSResult(
            beta=pls.beta, u=pls.u, b=pls.b, sigma_sq=pls.sigma_sq,
            pwrss=pls.pwrss, L=pls.L, RX=singular_RX,
            fitted=pls.fitted, residuals=pls.residuals,
        )
        se = _compute_se(bad)
        assert np.all(np.isfinite(se))
