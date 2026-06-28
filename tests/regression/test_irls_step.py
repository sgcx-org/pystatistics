"""IRLS step-control: float32 convergence robustness (CONVENTIONS Prime Directive).

The float32 GPU GLM path used to FALSELY REJECT person-period designs that float32
is perfectly capable of fitting: the IRLS loop quit at the √n·eps deviance-change
floor while a slowly-converging low-hazard interval dummy was still moving, leaving a
non-stationary iterate that the strict Newton-decrement gate then (correctly) refused.
The fix iterates until the iterate is genuinely stationary (Newton-decrement stop) and
damps float32 overshoot (step-halving), keeping the strict acceptance gate intact.

This module tests:
  - the step-control helpers (``step_halve``, ``relative_newton_decrement``) — no GPU;
  - that a person-period design which previously failed loud on MPS now CONVERGES and
    matches the CPU float64 fit (GPU-gated);
  - that the strict gate is unmoved — a non-stationary iterate is still rejected.
"""

import numpy as np
import pytest

from pystatistics.regression import Design, fit
from pystatistics.regression.families import resolve_family
from pystatistics.regression.backends._irls_step import (
    step_halve, relative_newton_decrement,
)
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


gpu_only = pytest.mark.skipif(
    not _gpu_available(), reason="GPU (CUDA or MPS) required"
)


# ---------------------------------------------------------------------------
# A deterministic person-period design that reproduces the failure mode.
# ---------------------------------------------------------------------------
def _person_period_with_slow_directions(seed=0):
    """A high-count bulk (flattens the deviance fast) plus moderate low-hazard
    interval dummies and TWO near-separated very-low-hazard dummies. The latter
    march far from the (y+0.5)/2 mustart and keep the early-iteration Newton
    decrement high — exactly the structure that made the old loop quit early and
    the gate then reject. Returns (X, y, identifiable_mask)."""
    rng = np.random.default_rng(seed)
    blocks = [(20000, 0.30)]                         # bulk
    blocks += [(1500, 0.05) for _ in range(8)]      # moderate low-hazard
    blocks += [(400, 0.004), (350, 0.004)]          # near-separated (~1-2 events)
    K = len(blocks)
    rows_d, rows_y, rows_c = [], [], []
    for j, (m, h) in enumerate(blocks):
        for _ in range(m):
            d = np.zeros(K); d[j] = 1.0
            rows_d.append(d)
            rows_y.append(1.0 if rng.uniform() < h else 0.0)
            rows_c.append(rng.standard_normal())
    X = np.column_stack([np.array(rows_d), np.array(rows_c)])
    y = np.array(rows_y)
    return X, y


# ===========================================================================
# step_halve  (no GPU)
# ===========================================================================
class TestStepHalve:
    def _setup(self, n=400, seed=0):
        rng = np.random.default_rng(seed)
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ [0.3, 0.8])))).astype(float)
        fam = resolve_family("binomial")
        return X, y, np.ones(n), fam

    def test_finite_step_is_returned_unchanged(self):
        # In the default mode a finite full Newton step passes through untouched
        # (bit-identical) — the float32 path's iteration 1 relies on this.
        X, y, wt, fam = self._setup()
        coef = np.array([0.25, 0.7])
        prev = np.array([0.1, 0.4])
        out_c, out_eta, out_mu, out_dev = step_halve(
            coef, prev, X, fam.link, fam, y, wt, dev_old=1e9
        )
        np.testing.assert_array_equal(out_c, coef)          # no halving
        np.testing.assert_allclose(out_eta, X @ coef)

    def test_increasing_finite_step_accepted_by_default(self):
        # Default (R-exact): a finite but deviance-INCREASING step is accepted —
        # Fisher scoring is not a descent method and R accepts such steps.
        X, y, wt, fam = self._setup()
        coef = np.array([5.0, 5.0])                         # large, worse fit
        prev = np.zeros(2)
        dev_prev = fam.deviance(y, fam.link.linkinv(X @ prev), wt)
        out_c, _, _, out_dev = step_halve(
            coef, prev, X, fam.link, fam, y, wt, dev_old=dev_prev
        )
        np.testing.assert_array_equal(out_c, coef)          # NOT halved
        assert out_dev > dev_prev                           # increase accepted

    def test_require_decrease_halves_an_increasing_step(self):
        # Damped mode (float32 GPU path): an increasing finite step IS halved back
        # toward the previous iterate until the deviance no longer increases.
        X, y, wt, fam = self._setup()
        prev = np.array([0.3, 0.7])                         # near the optimum
        dev_prev = fam.deviance(y, fam.link.linkinv(X @ prev), wt)
        coef = prev + np.array([6.0, 6.0])                  # a big overshoot
        out_c, _, _, out_dev = step_halve(
            coef, prev, X, fam.link, fam, y, wt, dev_old=dev_prev,
            require_decrease=True,
        )
        assert out_dev <= dev_prev + 1e-8                   # descent enforced
        assert not np.array_equal(out_c, coef)              # step was halved

    def test_nonfinite_step_is_always_halved_back(self):
        # The defensive backstop: a non-finite deviance is halved back to a finite
        # iterate even in the default (R-exact) mode. The real families clip η to
        # [-500, 500] so their deviance is always finite and never reaches this
        # branch; to exercise the guard honestly we use an UNCLIPPED stub link
        # (μ=exp(η)) so a large step genuinely overflows to a non-finite deviance.
        class _UnclippedLog:
            def linkinv(self, eta):
                with np.errstate(over="ignore"):
                    return np.exp(eta)        # no clip → overflows to inf (intended)

        class _SSEFamily:
            def deviance(self, y, mu, wt):
                with np.errstate(over="ignore", invalid="ignore"):
                    return float(np.sum(wt * (y - mu) ** 2))   # inf if any μ is inf

        n = 100
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)
        wt = np.ones(n)
        link, fam = _UnclippedLog(), _SSEFamily()
        coef = np.array([1000.0, 0.0])                      # μ = exp(1000) = inf
        prev = np.array([0.2, 0.3])
        assert not np.isfinite(fam.deviance(y, link.linkinv(X @ coef), wt))

        out_c, _, _, out_dev = step_halve(
            coef, prev, X, link, fam, y, wt, dev_old=1e9
        )
        assert np.isfinite(out_dev)
        assert not np.array_equal(out_c, coef)


# ===========================================================================
# relative_newton_decrement  (no GPU)
# ===========================================================================
class TestRelativeNewtonDecrement:
    def _normal_eqs(self, X, y, coef, fam):
        """Form XᵀWX / XᵀWz at ``coef`` the way the backend does each iteration."""
        link = fam.link
        eta = X @ coef
        mu = link.linkinv(eta)
        mu_eta = link.mu_eta(eta)
        var = fam.variance(mu)
        z = eta + (y - mu) / mu_eta
        w = np.maximum((mu_eta ** 2) / var, 1e-30)
        XtWX = (X * w[:, None]).T @ X
        XtWz = (X * w[:, None]).T @ z
        return XtWX, XtWz

    def test_zero_at_optimum_large_off_it(self):
        rng = np.random.default_rng(0)
        n = 3000
        X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ [0.3, 0.8, -0.5])))).astype(float)
        fam = resolve_family("binomial")
        beta = fit(Design.from_arrays(X, y), family="binomial", backend="cpu").coefficients
        dev = fam.deviance(y, fam.link.linkinv(X @ beta), np.ones(n))

        A, b = self._normal_eqs(X, y, beta, fam)
        assert relative_newton_decrement(A, b, beta, dev) < 1e-9   # stationary

        off = beta + np.array([0.5, 0.5, 0.5])
        A2, b2 = self._normal_eqs(X, y, off, fam)
        dev_off = fam.deviance(y, fam.link.linkinv(X @ off), np.ones(n))
        assert relative_newton_decrement(A2, b2, off, dev_off) > 1e-3  # not stationary

    def test_matches_independent_newton_decrement_form(self):
        # The normal-equation form must agree with the gate's _newton_decrement.
        rng = np.random.default_rng(2)
        n = 2000
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ [0.2, 0.9])))).astype(float)
        fam = resolve_family("binomial")
        coef = np.array([0.0, 0.0])                         # a non-optimum point
        dev = fam.deviance(y, fam.link.linkinv(X @ coef), np.ones(n))
        A, b = self._normal_eqs(X, y, coef, fam)
        rel = relative_newton_decrement(A, b, coef, dev)
        gate = _newton_decrement(X, y, np.ones(n), coef, fam.link, fam) / (abs(dev) + 0.1)
        np.testing.assert_allclose(rel, gate, rtol=1e-10)

    def test_singular_information_is_not_stationary(self):
        # A singular XᵀWX returns inf (no usable step) → never passes the gate.
        A = np.array([[1.0, 1.0], [1.0, 1.0]])              # rank 1
        b = np.array([1.0, 2.0])
        assert relative_newton_decrement(A, b, np.zeros(2), 10.0) == np.inf


# ===========================================================================
# GPU integration: the false-negative is fixed, the strict gate is intact
# ===========================================================================
@gpu_only
def test_person_period_previously_failed_now_converges():
    # This design fails loud on the pre-fix MPS path (the loop quits at the float32
    # deviance floor with the near-separated dummies still mid-flight). After the
    # fix it must CONVERGE and match the CPU float64 fit on every identifiable
    # coefficient (the near-separated dummies are a genuinely flat direction whose
    # MLE is ±∞ — excluded, exactly as for R's NA-aliased coefficients).
    X, y = _person_period_with_slow_directions(seed=0)
    d = Design.from_arrays(X, y)
    cpu = fit(d, family="binomial", backend="cpu")
    gpu = fit(d, family="binomial", backend="gpu")          # must NOT raise

    identifiable = np.abs(cpu.coefficients) < 6.0
    rel = np.max(
        np.abs((gpu.coefficients - cpu.coefficients)[identifiable])
        / np.maximum(np.abs(cpu.coefficients[identifiable]), 1e-6)
    )
    assert rel < 1e-3, f"identifiable coefficients off by {rel:.2e} (fp32 tier is ~1e-4)"
    # the well-determined fit (deviance) must match too
    np.testing.assert_allclose(gpu.deviance, cpu.deviance, rtol=1e-3)


@gpu_only
def test_strict_gate_unmoved_rejects_nonstationary_fit():
    # Make the iteration stop one step short by capping max_iter low on a design
    # whose slow directions need more iterations: the returned iterate is NOT
    # stationary, so the gate must FAIL LOUD rather than accept it (no false
    # positive). force=True then returns it with a warning, and it is genuinely off.
    X, y = _person_period_with_slow_directions(seed=0)
    d = Design.from_arrays(X, y)
    with pytest.raises(NumericalError):
        fit(d, family="binomial", backend="gpu", max_iter=2)
    forced = fit(d, family="binomial", backend="gpu", max_iter=2, force=True)
    assert forced.warnings  # returned under protest, flagged loudly
