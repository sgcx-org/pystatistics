"""Full-fit regression: the ETS workspace path == the allocating path.

The ML objective reuses a fit-scoped EtsWorkspace instead of allocating
fitted/residuals/s each evaluation. ``EtsWorkspace.recurse`` returns
fitted/residuals bit-identical to ``ets_recursion(..., want_states=False)``
(pinned in test_ets_kernel_parity.py), so the optimizer trajectory and every
downstream quantity must be identical. This proves it end to end by fitting
each model both ways — workspace (default) and with the workspace helper
monkeypatched to the allocating path — and asserting the solutions match.
"""

from __future__ import annotations

import numpy as np
import pytest

from pystatistics.timeseries import _ets_fit, ets
from pystatistics.timeseries._ets_models import ets_recursion

_FIELDS = ["alpha", "beta", "gamma", "phi", "aic", "aicc", "bic",
           "log_likelihood", "fitted_values", "residuals", "states", "converged"]


def _alloc_path(y, spec, params, init_states, ws_cell):
    """Allocating replacement for ets_recursion_ws (no workspace reuse)."""
    fitted, residuals, _ = ets_recursion(y, spec, params, init_states,
                                         want_states=False)
    return fitted, residuals


def _eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


@pytest.mark.parametrize("model,seed,seasonal", [
    ("AAN", 0, False),    # additive error/trend, no season
    ("AAA", 1, True),     # additive triple (seasonal)
    ("ANN", 2, False),    # simple exponential smoothing
    ("MAM", 3, True),     # multiplicative error + season
    ("AAdN", 4, False),   # damped trend
])
def test_ets_fit_identical_workspace_vs_allocating(monkeypatch, model, seed,
                                                   seasonal):
    rng = np.random.default_rng(seed)
    n = 96
    base = np.cumsum(rng.standard_normal(n)) + 100.0
    if seasonal:
        base = base + 10.0 * np.sin(2 * np.pi * np.arange(n) / 12)
    y = np.ascontiguousarray(base)
    period = 12 if seasonal else 1

    sol_ws = ets(y, model=model, period=period)
    monkeypatch.setattr(_ets_fit, "ets_recursion_ws", _alloc_path)
    sol_alloc = ets(y, model=model, period=period)

    for f in _FIELDS:
        assert _eq(getattr(sol_ws, f), getattr(sol_alloc, f)), f"field {f} differs"
