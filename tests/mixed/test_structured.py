"""Tests for the structure-exploiting PLS solver (F2b).

The structured solver must produce results identical (to machine precision)
to the dense ``solve_pls`` at the same θ, across single-factor, random-slope,
crossed, and nested designs — including ragged (unequal) group sizes — while
never materializing the dense Z. It must also run group counts / crossed
designs the dense path could not.
"""

import warnings

import numpy as np
import pytest

from pystatistics.mixed import lmm
from pystatistics.mixed._random_effects import (
    parse_random_effects, build_z_matrix, build_lambda,
)
from pystatistics.mixed._pls import solve_pls
from pystatistics.mixed._deviance import profiled_deviance_lmm
from pystatistics.mixed._pls_structured import (
    build_structured_context, solve_structured, deviance_structured,
    deviance_and_grad_structured, has_analytic_gradient,
)


class TestAnalyticGradient:
    """The analytic θ-gradient (A.3) must return the SAME deviance as the plain
    path and a gradient matching finite differences to ~1e-6 — across random
    intercept (q=1) and correlated random slope (q=2) single-factor designs."""

    def _ctx(self, q_terms, seed=0):
        rng = np.random.default_rng(seed)
        G, per = 30, 12
        n = G * per
        g = np.repeat(np.arange(G), per)
        x = rng.normal(0, 1, n)
        y = 10 + 2 * x + rng.normal(0, 3, n) + rng.normal(0, 2, G)[g]
        X = np.column_stack([np.ones(n), x])
        re = {"g": list(q_terms)}
        rd = {"x": x} if "x" in q_terms else {}
        specs = parse_random_effects({"g": g}, re, rd, n, build_dense=False)
        return build_structured_context(X, y, specs, reml=True)

    @pytest.mark.parametrize("q_terms,theta", [
        (("1",), np.array([0.8])),
        (("1", "x"), np.array([1.2, 0.3, 0.7])),
    ])
    def test_deviance_matches_and_grad_matches_fd(self, q_terms, theta):
        ctx = self._ctx(q_terms)
        assert has_analytic_gradient(ctx)
        dev_plain = deviance_structured(theta, ctx)
        dev_g, grad = deviance_and_grad_structured(theta, ctx)
        # Same objective value to round-off.
        np.testing.assert_allclose(dev_g, dev_plain, rtol=0, atol=1e-9)
        # Gradient matches central finite differences.
        fd = np.zeros_like(theta)
        h = 1e-6
        for i in range(len(theta)):
            tp = theta.copy(); tp[i] += h
            tm = theta.copy(); tm[i] -= h
            fd[i] = (deviance_structured(tp, ctx) - deviance_structured(tm, ctx)) / (2 * h)
        np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-5)

    def test_crossed_has_no_analytic_gradient(self):
        """Crossed/nested (sparse) designs fall back to finite differences."""
        rng = np.random.default_rng(1)
        n = 200
        s = rng.integers(0, 20, n); it = rng.integers(0, 15, n)
        x = rng.normal(size=n); y = rng.normal(size=n)
        X = np.column_stack([np.ones(n), x])
        specs = parse_random_effects({"s": s, "it": it},
                                     {"s": ["1"], "it": ["1"]}, {}, n,
                                     build_dense=False)
        ctx = build_structured_context(X, y, specs, reml=True)
        assert not has_analytic_gradient(ctx)


def _compare(y, X, groups, re, rd, theta, reml=True):
    """Structured vs dense solve_pls / deviance at fixed θ."""
    specs = parse_random_effects(groups, re, rd, len(y))  # dense + value_cols
    Z = build_z_matrix(specs)
    Lam = build_lambda(theta, specs)
    dense = solve_pls(X, Z, y, Lam, reml=reml)
    dev_dense = profiled_deviance_lmm(theta, X, Z, y, specs, reml=reml)

    ctx = build_structured_context(X, y, specs, reml)
    struct = solve_structured(theta, ctx)
    dev_struct = deviance_structured(theta, ctx)

    return dense, struct, dev_dense, dev_struct


class TestStructuredEqualsDense:
    """Machine-precision agreement of the structured and dense solvers."""

    def _data_single(self, q_terms=("1",), ragged=True, seed=0):
        rng = np.random.default_rng(seed)
        if ragged:
            sizes = rng.integers(5, 25, 10)
        else:
            sizes = np.full(10, 15)
        group = np.repeat(np.arange(10), sizes)
        n = len(group)
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])
        re = rng.standard_normal((10, len(q_terms)))
        y = X @ [2.0, 1.0] + re[group, 0] + rng.standard_normal(n)
        rd = {"x": x} if "x" in q_terms else None
        return y, X, {"g": group}, {"g": list(q_terms)}, rd

    def test_single_intercept_ragged(self):
        y, X, g, re, rd = self._data_single(("1",), ragged=True, seed=1)
        dense, struct, dd, ds = _compare(y, X, g, re, rd, np.array([0.7]))
        np.testing.assert_allclose(struct.beta, dense.beta, atol=1e-11)
        np.testing.assert_allclose(struct.b, dense.b, atol=1e-11)
        assert abs(struct.sigma_sq - dense.sigma_sq) < 1e-11
        assert abs(ds - dd) < 1e-8

    def test_single_slope_ragged(self):
        y, X, g, re, rd = self._data_single(("1", "x"), ragged=True, seed=2)
        dense, struct, dd, ds = _compare(y, X, g, re, rd, np.array([0.8, 0.1, 0.5]))
        np.testing.assert_allclose(struct.beta, dense.beta, atol=1e-10)
        np.testing.assert_allclose(struct.b, dense.b, atol=1e-10)
        assert abs(ds - dd) < 1e-8

    def test_single_slope_ml(self):
        y, X, g, re, rd = self._data_single(("1", "x"), ragged=True, seed=3)
        dense, struct, dd, ds = _compare(y, X, g, re, rd,
                                         np.array([0.8, 0.1, 0.5]), reml=False)
        np.testing.assert_allclose(struct.beta, dense.beta, atol=1e-10)
        assert abs(ds - dd) < 1e-8

    def test_crossed_intercept(self):
        rng = np.random.default_rng(4)
        n = 800
        a = rng.integers(0, 20, n)
        b = rng.integers(0, 13, n)
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])
        y = X @ [1.0, 1.5] + rng.standard_normal(20)[a] + rng.standard_normal(13)[b] \
            + rng.standard_normal(n)
        dense, struct, dd, ds = _compare(
            y, X, {"a": a, "b": b}, {"a": ["1"], "b": ["1"]}, None,
            np.array([0.6, 0.9]))
        np.testing.assert_allclose(struct.beta, dense.beta, atol=1e-10)
        np.testing.assert_allclose(struct.b, dense.b, atol=1e-10)
        assert abs(ds - dd) < 1e-7

    def test_crossed_with_slope(self):
        rng = np.random.default_rng(5)
        n = 600
        a = rng.integers(0, 15, n)
        b = rng.integers(0, 11, n)
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])
        y = X @ [1.0, 1.5] + rng.standard_normal(15)[a] + rng.standard_normal(11)[b] \
            + rng.standard_normal(n)
        dense, struct, dd, ds = _compare(
            y, X, {"a": a, "b": b}, {"a": ["1", "x"], "b": ["1"]}, {"x": x},
            np.array([0.6, 0.2, 0.4, 0.9]))
        np.testing.assert_allclose(struct.beta, dense.beta, atol=1e-9)
        np.testing.assert_allclose(struct.b, dense.b, atol=1e-9)
        assert abs(ds - dd) < 1e-7


class TestStructuredScaling:
    """The structured path runs sizes the dense path cannot."""

    def test_large_single_factor_no_oom(self):
        """G=3000 random intercepts (dense Z'Z would be huge) fits quickly."""
        rng = np.random.default_rng(7)
        G, n_per = 3000, 8
        n = G * n_per
        group = np.repeat(np.arange(G), n_per)
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])
        re = rng.standard_normal(G) * 1.5
        y = X @ [2.0, 1.0] + re[group] + rng.standard_normal(n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = lmm(y, X, groups={"g": group}, compute_satterthwaite=False)
        assert r.converged
        np.testing.assert_allclose(r.coefficients[1], 1.0, atol=0.1)
        assert r.var_components[0].variance > 1.0

    def test_crossed_recovers_truth(self):
        """A crossed design recovers the simulated variance components."""
        rng = np.random.default_rng(8)
        na, nb, n = 300, 200, 12000
        a = rng.integers(0, na, n)
        b = rng.integers(0, nb, n)
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])
        y = X @ [1.0, 1.5] + rng.standard_normal(na)[a] * 0.7 \
            + rng.standard_normal(nb)[b] * 0.5 + rng.standard_normal(n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = lmm(y, X, groups={"a": a, "b": b}, compute_satterthwaite=False)
        assert r.converged
        np.testing.assert_allclose(r.coefficients[1], 1.5, atol=0.05)
        variances = sorted(v.variance for v in r.var_components)
        # true variances 0.25 (=0.5²) and 0.49 (=0.7²)
        np.testing.assert_allclose(variances, [0.25, 0.49], atol=0.1)


class TestStructuredParseLean:
    def test_build_dense_false_skips_block(self):
        rng = np.random.default_rng(9)
        n = 100
        group = rng.integers(0, 8, n)
        specs = parse_random_effects({"g": group}, {"g": ["1"]}, None, n,
                                     build_dense=False)
        assert specs[0].Z_block is None
        assert specs[0].value_cols is not None
        assert specs[0].value_cols.shape == (n, 1)

    def test_build_dense_true_has_both(self):
        rng = np.random.default_rng(10)
        n = 100
        group = rng.integers(0, 8, n)
        specs = parse_random_effects({"g": group}, {"g": ["1"]}, None, n)
        assert specs[0].Z_block is not None
        assert specs[0].value_cols is not None
