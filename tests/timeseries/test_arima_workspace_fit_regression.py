"""Full-fit regression: the fused-workspace ML path must be indistinguishable
from the historical allocating path across the entire fit driver.

The workspace changes only *where scratch is allocated*, never the arithmetic —
`kalman_arma_loglik` returns a bit-identical nll with or without a workspace
(pinned in test_arima_kalman_cython_parity.py). Because L-BFGS-B is
deterministic in the objective values, the whole optimizer trajectory — and
therefore every downstream quantity — must be identical. This module proves that
end to end by running each fit BOTH ways:

  * default: workspace wired (production path),
  * forced allocating: monkeypatch ``_new_workspace`` to return None,

and asserting equality of fitted parameters, objective/information criteria,
convergence status, iteration counts, method used (CSS→ML), variance-covariance
(retry/veto-sensitive), residuals/fitted values, and emitted warnings — across
non-seasonal, seasonal, pure-ML, CSS-ML, near-nonstationary, auto_arima, and
degenerate-failure scenarios.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pystatistics.timeseries import _arima_fit
from pystatistics.timeseries import arima, auto_arima

# Solution fields that must match bit-for-bit between the two paths.
_ARRAY_FIELDS = ["ar", "ma", "seasonal_ar", "seasonal_ma", "residuals",
                 "fitted_values", "vcov"]
_SCALAR_FIELDS = ["mean", "log_likelihood", "aic", "aicc", "bic", "sigma2",
                  "converged", "n_iter", "method", "n_params", "n_obs",
                  "n_used", "order", "seasonal_order"]


def _series(seed, n=120, drift=0.0, level=40.0):
    rng = np.random.default_rng(seed)  # NON-DETERMINISTIC: fixed seed
    return np.cumsum(rng.standard_normal(n) + drift) + level


def _fit_both(monkeypatch, thunk):
    """Run *thunk* with the workspace (default) and with it forced off."""
    with warnings.catch_warnings(record=True) as w_ws:
        warnings.simplefilter("always")
        sol_ws = thunk()
    monkeypatch.setattr(_arima_fit, "_new_workspace", lambda *a, **k: None)
    with warnings.catch_warnings(record=True) as w_alloc:
        warnings.simplefilter("always")
        sol_alloc = thunk()
    msgs = lambda ws: sorted(str(x.message) for x in ws)
    return sol_ws, msgs(w_ws), sol_alloc, msgs(w_alloc)


def _assert_equal_field(name, a, b):
    if a is None or b is None:
        assert a is None and b is None, f"{name}: {a!r} vs {b!r}"
        return
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b),
                                      err_msg=f"field {name} differs")
    else:
        assert a == b, f"field {name}: {a!r} vs {b!r}"


def _assert_solutions_identical(sol_ws, w_ws, sol_alloc, w_alloc):
    for f in _ARRAY_FIELDS + _SCALAR_FIELDS:
        _assert_equal_field(f, getattr(sol_ws, f), getattr(sol_alloc, f))
    # Warnings emitted must be identical (CSS-ML / convergence / veto messages).
    assert w_ws == w_alloc, f"warnings differ:\n  ws={w_ws}\n  alloc={w_alloc}"


# --- scenarios covering the fit-driver branches --------------------------

@pytest.mark.parametrize("order,method,seed,mean_series", [
    ((2, 1, 1), "css-ml", 0, True),    # default path: CSS warm-start -> ML (+ ml2)
    ((2, 1, 1), "ml", 1, True),        # pure ML
    ((1, 1, 1), "css-ml", 2, False),   # no mean -> skips the ml2 second-start
    ((0, 1, 2), "ml", 3, True),        # pure MA
    ((3, 0, 0), "css-ml", 4, True),    # pure AR, no differencing (mean matters)
    ((1, 0, 1), "css-ml", 5, True),    # ARMA with mean: exercises ml2 veto/keep
])
def test_fit_identical_workspace_vs_allocating(monkeypatch, order, method, seed,
                                               mean_series):
    y = _series(seed, drift=0.3 if mean_series else 0.0)
    thunk = lambda: arima(y, order=order, method=method)
    _assert_solutions_identical(*_fit_both(monkeypatch, thunk))


def test_fit_identical_seasonal(monkeypatch):
    """SARIMA: larger effective r + factored-Hessian path."""
    rng = np.random.default_rng(11)
    # 4-period seasonal signal
    t = np.arange(160)
    y = (10 * np.sin(2 * np.pi * t / 4) + np.cumsum(rng.standard_normal(160)) + 50)
    thunk = lambda: arima(y, order=(1, 1, 1), seasonal=(1, 0, 1, 4),
                          method="css-ml")
    _assert_solutions_identical(*_fit_both(monkeypatch, thunk))


def test_fit_identical_near_nonstationary(monkeypatch):
    """A near-unit-root series makes the optimizer probe nonstationary regions,
    exercising the diffuse-init fallback inside the workspace."""
    rng = np.random.default_rng(21)
    y = np.zeros(200)
    for i in range(1, 200):
        y[i] = 0.98 * y[i - 1] + rng.standard_normal()
    thunk = lambda: arima(y, order=(1, 0, 1), method="ml")
    _assert_solutions_identical(*_fit_both(monkeypatch, thunk))


def test_auto_arima_identical(monkeypatch):
    """auto_arima creates a workspace per candidate order; the selected model
    and all its fields must match the allocating search exactly."""
    y = _series(31, n=140, drift=0.2)
    thunk = lambda: auto_arima(y, max_p=2, max_q=2)
    a, w_a, b, w_b = _fit_both(monkeypatch, thunk)
    assert a.best_order == b.best_order
    assert a.best_aic == b.best_aic
    assert a.best_seasonal == b.best_seasonal
    _assert_solutions_identical(a.best_model, w_a, b.best_model, w_b)


def test_degenerate_series_identical(monkeypatch):
    """A degenerate (constant → all-zero difference) series drives the fit into
    its edge handling (singular Hessian, etc.); both paths must resolve it
    identically — same convergence, warnings, and (possibly None) vcov."""
    y = np.full(90, 3.0)
    thunk = lambda: arima(y, order=(1, 1, 1), method="css-ml")
    _assert_solutions_identical(*_fit_both(monkeypatch, thunk))


def test_objective_bit_identical_including_penalties(monkeypatch):
    """Objective-level failure-path guard: for feasible AND infeasible parameter
    vectors (nonstationary / non-invertible), the workspace objective returns
    the identical value — including the 1e18 penalty — as the allocating path.
    This deterministically exercises the diffuse-fallback and penalty branches
    of the fused kernel."""
    from pystatistics.timeseries._arima_kalman import _new_workspace
    from pystatistics.timeseries._arima_likelihood import arima_negloglik

    y = _series(5, n=100, drift=0.1)
    order = (1, 1)             # objective-level (p, q): p=1, q=1 -> r = max(1, 2) = 2
    include_mean = True
    ws = _new_workspace(1, 1, len(y))
    mu = float(np.mean(y))

    param_vectors = [
        np.array([0.4, 0.3, mu]),      # feasible
        np.array([2.0, 0.3, mu]),      # nonstationary AR -> diffuse fallback
        np.array([0.4, 3.0, mu]),      # non-invertible MA
        np.array([1.0, -1.0, mu]),     # unit-root edge
        np.array([50.0, 50.0, mu]),    # wildly infeasible -> penalty
    ]
    for params in param_vectors:
        nll_ws = arima_negloglik(params, y, order, include_mean, "ml",
                                 _workspace=ws)
        nll_alloc = arima_negloglik(params, y, order, include_mean, "ml")
        assert nll_ws == nll_alloc, f"objective differs at {params}: {nll_ws} vs {nll_alloc}"
