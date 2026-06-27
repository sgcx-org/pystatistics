"""GPU GLM float32 convergence-ACCEPTANCE (A6).

A correct, well-conditioned float32 fit that reaches the float32 deviance floor
must be ACCEPTED — not rejected for failing an unreachable float64 tolerance.
A genuinely-unreliable float32 fit (non-stationary, or a broken inner solve)
must still fail loud. The decision is the relative Newton decrement, not the
deviance-change flag.

The Newton-decrement unit tests need no GPU; the acceptance tests are GPU-gated.
"""

import numpy as np
import pytest

from pystatistics.regression import Design, fit
from pystatistics.regression.families import resolve_family
from pystatistics.regression.backends.gpu_glm import (
    _newton_decrement, _FP32_REL_DECREMENT_TOL,
)
from pystatistics.core.exceptions import NumericalError


def _gpu_available():
    try:
        import torch
        if torch.cuda.is_available():
            return True
        mps = getattr(torch.backends, "mps", None)
        return bool(mps and mps.is_available())
    except ImportError:
        return False


gpu_only = pytest.mark.skipif(not _gpu_available(), reason="GPU (CUDA or MPS) required")


# --- Newton decrement (no GPU needed) ----------------------------------------

def test_newton_decrement_zero_at_optimum():
    rng = np.random.default_rng(0)
    n = 2000
    X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ [0.3, 0.8, -0.5])))).astype(float)
    fam = resolve_family("binomial")
    beta_hat = fit(Design.from_arrays(X, y), family="binomial", backend="cpu").coefficients
    wt = np.ones(n)
    dec_opt = _newton_decrement(X, y, wt, beta_hat, fam.link, fam)
    dec_off = _newton_decrement(X, y, wt, beta_hat + np.array([1.0, 1.0, 1.0]),
                                fam.link, fam)
    assert dec_opt < 1e-6           # at the optimum the decrement vanishes
    assert dec_off > 1.0            # far from it, it is large
    assert dec_opt < dec_off


def test_newton_decrement_invariant_to_benign_scaling():
    # Scaling a column makes X'X ill-conditioned but the FIT unchanged; the
    # affine-invariant decrement must stay ~0 at the optimum regardless.
    rng = np.random.default_rng(1)
    n = 3000
    X = np.column_stack([np.ones(n), rng.standard_normal(n) * 1e4,
                         rng.standard_normal(n) * 1e-3])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ [0.2, 1e-4, 50.0])))).astype(float)
    fam = resolve_family("binomial")
    beta_hat = fit(Design.from_arrays(X, y), family="binomial", backend="cpu").coefficients
    assert np.linalg.cond(X.T @ X) > 1e8
    assert _newton_decrement(X, y, np.ones(n), beta_hat, fam.link, fam) < 1e-6


# --- float32 acceptance on the GPU -------------------------------------------

@gpu_only
def test_fp32_well_conditioned_binomial_accepts():
    # Person-period-style design (many interval dummies) — the case that used to
    # be falsely rejected on MPS. It must now accept and match the CPU fit.
    rng = np.random.default_rng(0)
    n, K = 6000, 20
    interval = rng.integers(0, K, size=n)
    Xi = np.zeros((n, K)); Xi[np.arange(n), interval] = 1.0
    Xc = rng.standard_normal((n, 3))
    X = np.column_stack([Xi, Xc])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ (rng.standard_normal(K + 3) * 0.5))))).astype(float)
    d = Design.from_arrays(X, y)
    cpu = fit(d, family="binomial", backend="cpu")
    gpu = fit(d, family="binomial", backend="gpu")   # must NOT raise
    rel = np.max(np.abs(gpu.coefficients - cpu.coefficients)
                 / np.maximum(np.abs(cpu.coefficients), 1e-6))
    assert rel < 1e-3                                  # fp32 tier


@gpu_only
def test_ill_conditioned_never_returns_silently_wrong():
    # A genuinely ill-conditioned log-link design at scale. The float32 outcome is
    # platform-dependent — on MPS the float32 Cholesky breaks down and it fails
    # loud; on CUDA the more accurate float32 solves it correctly. EITHER is fine
    # under A6; what must never happen is a silently-wrong accepted fit. So: it
    # raises, or it returns a result that matches the CPU fit. Never a wrong one.
    rng = np.random.default_rng(5)
    n, p = 300000, 120
    X = np.column_stack([np.ones(n)] + [rng.standard_normal(n) for _ in range(p - 1)])
    eta = np.clip(X @ (rng.standard_normal(p) * 2.5), -12, 12)
    y = rng.poisson(np.exp(eta)).astype(float)
    d = Design.from_arrays(X, y)
    cpu = fit(d, family="poisson", backend="cpu")
    try:
        gpu = fit(d, family="poisson", backend="gpu")
    except NumericalError:
        return                       # failing loud is an acceptable outcome (A6)
    rel = np.max(np.abs(gpu.coefficients - cpu.coefficients)
                 / np.maximum(np.abs(cpu.coefficients), 1e-6))
    assert rel < 1e-2, f"accepted a silently-wrong float32 fit (rel={rel:.2e})"


def test_threshold_is_relative_and_small():
    # Guards the calibrated constant against accidental edits.
    assert 0 < _FP32_REL_DECREMENT_TOL <= 1e-5
