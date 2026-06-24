"""
Unit tests for the scaled optimisation driver (backends/_optimize.py).

These exercise the helper in isolation from the backends: the per-observation
scaling must not move the optimum, must make a large-objective fit converge,
must keep reporting non-convergence for starved fits, and must fail loud on a
misconfigured objective.
"""

import numpy as np
import pytest

from pystatistics.core.exceptions import NumericalError
from pystatistics.mvnmle._objectives.cpu import CPUObjectiveFP64
from pystatistics.mvnmle.backends._optimize import (
    run_scaled_minimize,
    ScaledMinimizeResult,
)


def _clean_data(n, p=5, seed=0):
    """Well-conditioned MCAR data with no all-missing rows."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, p))
    sigma = A @ A.T + p * np.eye(p)
    mu = rng.standard_normal(p)
    X = mu + rng.standard_normal((n, p)) @ np.linalg.cholesky(sigma).T
    mask = rng.random((n, p)) < 0.15
    mask[mask.all(axis=1)] = False
    X[mask] = np.nan
    return X


class TestScale:
    """The scale factor is the count of observed scalar values."""

    def test_scale_equals_observed_count(self):
        X = _clean_data(n=1000)
        obj = CPUObjectiveFP64(X, validate=False)
        opt = run_scaled_minimize(
            obj, obj.get_initial_parameters(),
            method='BFGS', tol=1e-5, max_iter=200,
        )
        assert opt.scale == float((~np.isnan(X)).sum())

    def test_missing_count_fails_loud(self):
        """A non-positive observed-scalar count must raise, not divide by zero."""
        X = _clean_data(n=500)
        obj = CPUObjectiveFP64(X, validate=False)
        obj.n_observed_scalars = 0
        with pytest.raises(NumericalError, match="n_observed_scalars"):
            run_scaled_minimize(
                obj, obj.get_initial_parameters(),
                method='BFGS', tol=1e-5, max_iter=10,
            )


class TestConvergence:
    """Scaling makes convergence judged on an O(1), scale-invariant gradient."""

    def test_large_objective_fit_converges(self):
        X = _clean_data(n=5000)
        obj = CPUObjectiveFP64(X, validate=False)
        opt = run_scaled_minimize(
            obj, obj.get_initial_parameters(),
            method='BFGS', tol=1e-5, max_iter=100,
        )
        assert isinstance(opt, ScaledMinimizeResult)
        assert opt.success
        # Reported gradient norm is the scaled one, below gtol.
        assert opt.gradient_norm is not None
        assert opt.gradient_norm < 1e-5

    def test_starved_fit_does_not_converge(self):
        X = _clean_data(n=5000)
        obj = CPUObjectiveFP64(X, validate=False)
        opt = run_scaled_minimize(
            obj, obj.get_initial_parameters(),
            method='BFGS', tol=1e-5, max_iter=2,
        )
        assert not opt.success


class TestOptimumUnchanged:
    """Scaling by a positive constant must not move the argmin."""

    def test_estimates_match_unscaled_optimum(self):
        from scipy.optimize import minimize

        X = _clean_data(n=2000)
        obj = CPUObjectiveFP64(X, validate=False)
        theta0 = obj.get_initial_parameters()

        # Unscaled reference run, allowed many iterations to reach the optimum
        # despite the precision-loss termination.
        ref = minimize(
            obj.compute_objective, theta0, jac=obj.compute_gradient,
            method='BFGS', options={'maxiter': 100000, 'gtol': 1e-12},
        )
        mu_ref, sigma_ref, loglik_ref = obj.extract_parameters(ref.x)

        opt = run_scaled_minimize(
            obj, theta0, method='BFGS', tol=1e-8, max_iter=100000,
        )
        mu, sigma, loglik = obj.extract_parameters(opt.x)

        assert abs(loglik - loglik_ref) < 1e-6
        np.testing.assert_allclose(mu, mu_ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(sigma, sigma_ref, rtol=1e-5, atol=1e-5)

    def test_objective_value_reported_unscaled(self):
        """objective_value stays on the -2*loglik scale (consistent with loglik)."""
        X = _clean_data(n=2000)
        obj = CPUObjectiveFP64(X, validate=False)
        opt = run_scaled_minimize(
            obj, obj.get_initial_parameters(),
            method='BFGS', tol=1e-5, max_iter=200,
        )
        _, _, loglik = obj.extract_parameters(opt.x)
        assert abs(opt.objective_value - (-2.0 * loglik)) < 1e-6


class _FusedAdapter:
    """Wrap an objective to expose ``compute_value_and_gradient`` and count which
    callables the driver actually invokes.

    The fused method returns the same value/gradient as the separate callables
    (it delegates to them), so any optimum difference is attributable purely to
    which scipy code path was taken — exactly what the wiring tests below check.
    """

    def __init__(self, obj):
        self._obj = obj
        self.n_fused = 0
        self.n_sep_obj = 0

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def compute_objective(self, theta):
        self.n_sep_obj += 1
        return self._obj.compute_objective(theta)

    def compute_value_and_gradient(self, theta):
        self.n_fused += 1
        return self._obj.compute_objective(theta), self._obj.compute_gradient(theta)


class TestFusedValueGradient:
    """The driver folds value+gradient into one fused callable when the objective
    exposes ``compute_value_and_gradient`` (one host<->device sync per eval on a
    GPU objective), and falls back to the two-callable path otherwise."""

    def test_fused_path_used_when_available(self):
        X = _clean_data(n=2000)
        obj = _FusedAdapter(CPUObjectiveFP64(X, validate=False))
        opt = run_scaled_minimize(
            obj, obj.get_initial_parameters(),
            method='L-BFGS-B', tol=1e-5, max_iter=200,
        )
        assert opt.n_iter > 0
        # The fused callable drove the optimisation; scipy never called the
        # standalone objective callable (it is fed jac=True instead).
        assert obj.n_fused > 0
        assert obj.n_sep_obj == 0

    def test_fallback_path_when_method_absent(self):
        """An objective without ``compute_value_and_gradient`` (the numpy CPU
        reference) uses the original two-callable path unchanged."""
        X = _clean_data(n=2000)
        obj = CPUObjectiveFP64(X, validate=False)
        assert not hasattr(obj, 'compute_value_and_gradient')
        opt = run_scaled_minimize(
            obj, obj.get_initial_parameters(),
            method='L-BFGS-B', tol=1e-5, max_iter=200,
        )
        assert opt.n_iter > 0

    def test_fused_and_fallback_reach_same_optimum(self):
        X = _clean_data(n=2000)
        theta0 = CPUObjectiveFP64(X, validate=False).get_initial_parameters()
        fused = _FusedAdapter(CPUObjectiveFP64(X, validate=False))
        sep = CPUObjectiveFP64(X, validate=False)  # no fused method -> fallback
        o_f = run_scaled_minimize(fused, theta0, method='BFGS',
                                  tol=1e-8, max_iter=100000)
        o_s = run_scaled_minimize(sep, theta0, method='BFGS',
                                  tol=1e-8, max_iter=100000)
        np.testing.assert_allclose(o_f.x, o_s.x, rtol=1e-6, atol=1e-7)
        assert abs(o_f.objective_value - o_s.objective_value) < 1e-6
