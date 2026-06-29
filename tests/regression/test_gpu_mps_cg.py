"""MPS float32 GLM inner solve: matrix-free conjugate gradient (squaring-free).

The MPS plain-float32 IRLS inner solve uses a matrix-free CG step on the operator
``H v = Xᵀ(W(X v))`` instead of Cholesky on the on-device weighted normal-equations
matrix ``XᵀWX``. Forming ``XᵀWX`` squares the condition number, and on a cold MPS
context the float32 Gram matrix at person-period quarterly scale can come out
not-positive-definite and crash Cholesky (study §4 — always fail-loud, never
silently wrong). The squaring-free CG step removes that failure at its root.

This file covers, on the shipped torch version:
  - the CG solver matches a float64 reference on a weighted normal-equations system
    across conditioning (device-agnostic — no GPU needed);
  - a person-period-style binomial design that exercises the squaring-fragile regime
    converges via CG to float32 tier on MPS (method ``irls_cg_gpu``);
  - a genuinely float32-infeasible design still REFUSES loudly, and the refuse
    message recommends ``backend='cpu'`` and does NOT recommend ``backend='gpu_fp64'``
    on MPS (Metal has no float64, so gpu_fp64 there is a guaranteed CUDA-required
    error — A6: name only the real same-machine remedy).

The full cold-context crash reproduction (old Cholesky crashes 24/24 cold at the
flchain quarterly knife edge; CG converges cold) lives in the pystatistics-validation
survival harness — it needs the flchain dataset and a fresh cold process launch, so
it is exercised there during benchmark re-validation, not in this isolated suite.

NOTE: MPS kernel behaviour is torch-version-sensitive; this solver path is validated
per shipped torch version. The host float64 Newton-decrement acceptance gate
(``_newton_decrement`` / ``_FP32_REL_DECREMENT_TOL``, unchanged) is the
version-INDEPENDENT guarantee that an unreliable fit fails loud.
"""

import numpy as np
import pytest

from pystatistics.regression import Design, fit
from pystatistics.regression.backends.gpu_glm import _mps_cg_solve
from pystatistics.core.exceptions import NumericalError


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _mps_available():
    torch = _torch()
    return bool(torch and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available())


mps_only = pytest.mark.skipif(not _mps_available(), reason="MPS (Apple Silicon) required")
needs_torch = pytest.mark.skipif(_torch() is None, reason="torch required")


# ---------------------------------------------------------------------------
# CG solver — device-agnostic correctness (no GPU needed: torch CPU device)
# ---------------------------------------------------------------------------

@needs_torch
@pytest.mark.parametrize("cond", [1.0, 1e2, 1e4])
def test_mps_cg_solve_matches_fp64_normal_equations(cond):
    """CG on H v = Xᵀ(W(X v)) recovers the float64 weighted normal-equations
    solution WITHOUT forming XᵀWX, across a range of design conditioning."""
    torch = _torch()
    rng = np.random.default_rng(0)
    n, p = 4000, 12
    # Controlled conditioning: scale columns so κ(X) ≈ cond.
    X = rng.standard_normal((n, p))
    X[:, 0] *= cond
    w = np.abs(rng.standard_normal(n)) * 0.3 + 0.05
    z = rng.standard_normal(n)

    # float64 reference solve of (XᵀWX) b = Xᵀ(W z) — solved directly, the answer
    # CG must reproduce (CG never forms XᵀWX).
    A = X.T @ (w[:, None] * X)
    b_ref = np.linalg.solve(A, X.T @ (w * z))

    Xt = torch.from_numpy(X).to(dtype=torch.float64)
    wt = torch.from_numpy(w).to(dtype=torch.float64)
    rhs = Xt.T @ (wt * torch.from_numpy(z).to(dtype=torch.float64))
    b0 = torch.zeros(p, dtype=torch.float64)
    # Driven to a tight residual so the solution-error floor (≈ tol·κ(X)²) is well
    # below the assertion across the conditioning range — confirming CG converges
    # to the SAME fixed point as the direct solve, not merely an approximation.
    b_cg = _mps_cg_solve(Xt, wt, rhs, b0, tol=1e-12, max_iter=2000).numpy()

    rel = np.max(np.abs(b_cg - b_ref) / (np.abs(b_ref) + 1e-12))
    assert rel < 1e-6, f"CG solve disagreed with fp64 normal equations (rel={rel:.2e})"


@needs_torch
def test_mps_cg_solve_warm_start_is_consistent():
    """Warm-starting CG from a prior iterate lands at the same solution as a
    cold start (it only changes the iteration count, not the fixed point)."""
    torch = _torch()
    rng = np.random.default_rng(1)
    n, p = 3000, 8
    X = rng.standard_normal((n, p))
    w = np.abs(rng.standard_normal(n)) * 0.3 + 0.05
    z = rng.standard_normal(n)
    Xt = torch.from_numpy(X).to(dtype=torch.float64)
    wt = torch.from_numpy(w).to(dtype=torch.float64)
    rhs = Xt.T @ (wt * torch.from_numpy(z).to(dtype=torch.float64))

    cold = _mps_cg_solve(Xt, wt, rhs, torch.zeros(p, dtype=torch.float64)).numpy()
    warm = _mps_cg_solve(Xt, wt, rhs, torch.from_numpy(cold * 0.9)).numpy()
    assert np.allclose(cold, warm, rtol=1e-7, atol=1e-9)


# ---------------------------------------------------------------------------
# End-to-end MPS fit via the CG path
# ---------------------------------------------------------------------------

def _person_period_design(n_subj, n_int, n_cov, rng):
    """Synthetic person-period binomial design: [interval one-hot | covariates],
    no intercept — the layout that, at quarterly scale, puts the float32 XᵀWX on
    the not-PD knife edge. Independent covariates (no artificial collinearity), so
    coefficient error reflects the solver, not unidentifiability."""
    rows_oh, rows_cov, ys = [], [], []
    beta_cov = rng.standard_normal(n_cov) * 0.5
    base = rng.standard_normal(n_int) * 0.3
    for _ in range(n_subj):
        cov = rng.standard_normal(n_cov)
        for j in range(n_int):
            eta = base[j] + cov @ beta_cov
            y = 1.0 if rng.uniform() < 1 / (1 + np.exp(-eta)) else 0.0
            oh = np.zeros(n_int); oh[j] = 1.0
            rows_oh.append(oh); rows_cov.append(cov); ys.append(y)
            if y == 1.0:
                break
    X = np.column_stack([np.array(rows_oh), np.array(rows_cov)])
    return X, np.array(ys), n_cov


@mps_only
def test_mps_cg_person_period_converges_to_fp32_tier():
    """A person-period-style binomial design fits via the matrix-free CG path on
    MPS and matches the CPU float64 fit on the identifiable covariate columns to
    float32 tier — the regime where forming XᵀWX in float32 is fragile."""
    rng = np.random.default_rng(42)
    X, y, n_cov = _person_period_design(n_subj=6000, n_int=40, n_cov=5, rng=rng)
    d = Design.from_arrays(X, y)

    gpu = fit(d, family="binomial", backend="gpu")
    cpu = fit(d, family="binomial", backend="cpu")

    assert gpu.info["method"] == "irls_cg_gpu"      # CG path, not Cholesky
    assert gpu.info["device"].startswith("mps")
    assert gpu.converged
    cov = slice(X.shape[1] - n_cov, X.shape[1])      # identifiable covariate cols
    rel = np.max(np.abs(gpu.coefficients[cov] - cpu.coefficients[cov])
                 / (np.abs(cpu.coefficients[cov]) + 1e-12))
    assert rel < 1e-3, f"CG fit not float32 tier (covariate rel err={rel:.2e})"


@mps_only
def test_mps_well_conditioned_binomial_matches_cpu():
    """Sanity: an ordinary logistic fit goes through the CG path and matches CPU."""
    rng = np.random.default_rng(3)
    n, p = 5000, 6
    X = np.column_stack([np.ones(n)] + [rng.standard_normal(n) for _ in range(p - 1)])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ (rng.standard_normal(p) * 0.6))))).astype(float)
    d = Design.from_arrays(X, y)
    gpu = fit(d, family="binomial", backend="gpu")
    cpu = fit(d, family="binomial", backend="cpu")
    assert gpu.info["method"] == "irls_cg_gpu"
    rel = np.max(np.abs(gpu.coefficients - cpu.coefficients)
                 / np.maximum(np.abs(cpu.coefficients), 1e-6))
    assert rel < 1e-3


# ---------------------------------------------------------------------------
# Refuse path: fail loud with the MPS-correct remedy
# ---------------------------------------------------------------------------

@mps_only
@pytest.mark.slow
def test_mps_infeasible_refuses_and_recommends_cpu_not_fp64():
    """A genuinely float32-infeasible log-link design at scale still REFUSES (the
    host float64 gate stays a true positive). On MPS the refuse message must
    recommend backend='cpu' (the only same-machine higher-precision fallback) and
    must NOT recommend backend='gpu_fp64' — Metal has no float64, so gpu_fp64 there
    would only produce a guaranteed CUDA-required error (A6)."""
    rng = np.random.default_rng(11)
    n, p = 60000, 80
    X = np.column_stack([np.ones(n)] + [rng.standard_normal(n) for _ in range(p - 1)])
    eta = np.clip(X @ (rng.standard_normal(p) * 3.5), -15, 15)
    y = rng.gamma(2.0, np.exp(eta) / 2.0).astype(float) + 1e-6
    d = Design.from_arrays(X, y)

    with pytest.raises(NumericalError) as exc:
        fit(d, family="Gamma", backend="gpu")        # must fail loud, not silently wrong
    msg = str(exc.value)
    assert "backend='cpu'" in msg                    # the real MPS remedy is named
    assert "gpu_fp64" not in msg                     # NOT offered on MPS (CUDA-only)
