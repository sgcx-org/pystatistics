"""
Covariance-sensitive validation of the GPU ``polr`` posterior draw.

The GPU MICE ``polr`` backend draws ``[thresholds, slopes]`` from the model's
posterior normal approximation. The threshold posterior covariance governs the
between-imputation variability of ordered-factor imputations, so it propagates
into Rubin's-rules variance and interval coverage. Marginal category proportions
are insensitive to it — a proportion check passes even when the draw uses the
wrong covariance — so this module checks the covariance directly.

``MASS::polr`` (and the CPU ``polr`` it is validated against) report and draw in
*natural* threshold coordinates. The GPU fit observes the Hessian in the *raw*
(log-gap) parameterization and must map it to natural coordinates via the
raw->natural Jacobian (delta method) before drawing. These tests pin the GPU
draw covariance to the CPU ``OrdinalSolution.vcov`` (natural coordinates,
R-validated) and verify it is NOT the raw-coordinate covariance — the exact
parameterization error that a proportions-only check cannot catch.

Device-agnostic core (FP64 on CPU torch, tight tolerance); an on-device MPS
variant at the FP32 tolerance tier validates the actual ``mice(..., backend='gpu')``
code path. The CUDA path shares the same device-split linear algebra and is
exercised by the same core test on a CUDA host.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pystatistics.ordinal import polr
from pystatistics.mice.backends._gpu_polr import (
    batched_polr_newton,
    draw_natural_theta,
)


def _ordered_dataset(seed: int, n: int = 1200, q: int = 3, K: int = 4):
    """Ordered target with non-trivially spaced thresholds, so the raw and
    natural covariances genuinely differ (the test must have teeth)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, q))
    beta = rng.standard_normal(q) * 0.9
    eta = X @ beta
    cuts = np.array([-1.0, 0.2, 1.1])[: K - 1]
    u = rng.uniform(size=n)
    cum = 1.0 / (1.0 + np.exp(-(cuts[None, :] - eta[:, None])))
    y = (u[:, None] > cum).sum(axis=1).astype(np.intp)
    for lv in range(K):  # guarantee all levels present
        if not np.any(y == lv):
            y[lv] = lv
    return y, X, K


def _fit_cpu_and_gpu(y, X, K, device, dtype):
    """CPU ``polr`` reference fit plus the GPU batched Newton fit (one chain) on
    ``device``/``dtype``. Returns ``(fit, alpha, beta, raw, L)``."""
    fit = polr(y, X)
    y_obs = torch.tensor(y, dtype=dtype, device=device).unsqueeze(0)
    X_obs = torch.tensor(X, dtype=dtype, device=device).unsqueeze(0)
    alpha, beta, raw, L = batched_polr_newton(y_obs, X_obs, K)
    return fit, alpha, beta, raw, L


def _empirical_draw_cov(alpha, beta, raw, L, device, n_draws, seed):
    """Empirical covariance of ``n_draws`` independent natural-coordinate draws,
    produced by the exact production helper batched ``n_draws``-wide (all chains
    share the single fit)."""
    ae = alpha.expand(n_draws, -1).contiguous()
    be = beta.expand(n_draws, -1).contiguous()
    re = raw.expand(n_draws, -1).contiguous()
    Le = L.expand(n_draws, -1, -1).contiguous()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    theta = draw_natural_theta(ae, be, re, Le, gen)
    return np.cov(theta.detach().cpu().to(torch.float64).numpy().T)


def _raw_cov(L):
    """Raw-coordinate covariance ``(L L^T)^{-1}`` — the pre-fix (buggy) draw cov."""
    H_raw = (L[0] @ L[0].transpose(0, 1)).detach().cpu().to(torch.float64).numpy()
    return np.linalg.inv(H_raw)


# ---------------------------------------------------------------- device-agnostic

class TestNaturalCoordinateDrawCPU:
    """FP64 core (CPU torch): the draw covariance is the natural-coordinate vcov
    that ``MASS::polr`` / CPU ``polr`` report, NOT the raw-coordinate one."""

    def test_point_estimate_matches_cpu_polr(self):
        """The batched Newton fit lands on the same MLE as CPU ``polr``."""
        y, X, K = _ordered_dataset(seed=3)
        fit, alpha, beta, _, _ = _fit_cpu_and_gpu(
            y, X, K, torch.device("cpu"), torch.float64
        )
        np.testing.assert_allclose(
            alpha[0].numpy(), fit.threshold_values, atol=1e-3
        )
        np.testing.assert_allclose(beta[0].numpy(), fit.coefficients, atol=1e-3)

    def test_draw_cov_matches_cpu_natural_vcov(self):
        """Empirical draw covariance == CPU ``fit.vcov`` (natural coordinates)."""
        y, X, K = _ordered_dataset(seed=3)
        fit, alpha, beta, raw, L = _fit_cpu_and_gpu(
            y, X, K, torch.device("cpu"), torch.float64
        )
        emp = _empirical_draw_cov(
            alpha, beta, raw, L, torch.device("cpu"), n_draws=60000, seed=0
        )
        vcov_nat = fit.vcov
        rel = np.linalg.norm(emp - vcov_nat) / np.linalg.norm(vcov_nat)
        assert rel < 0.05, f"draw cov not natural vcov (rel frob {rel:.4f})"

    def test_draw_cov_is_not_raw_vcov(self):
        """Regression guard for the issue #6 bug: the draw must NOT carry the
        raw-coordinate covariance. The two parameterizations differ substantially
        on this data, so a raw-coordinate draw (pre-fix) would be caught."""
        y, X, K = _ordered_dataset(seed=3)
        fit, alpha, beta, raw, L = _fit_cpu_and_gpu(
            y, X, K, torch.device("cpu"), torch.float64
        )
        emp = _empirical_draw_cov(
            alpha, beta, raw, L, torch.device("cpu"), n_draws=60000, seed=0
        )
        vcov_nat = fit.vcov
        vcov_raw = _raw_cov(L)
        nat_norm = np.linalg.norm(vcov_nat)
        # The data must actually distinguish the two parameterizations, else the
        # test has no teeth.
        assert np.linalg.norm(vcov_raw - vcov_nat) / nat_norm > 0.1
        # The empirical draw cov is far closer to natural than to raw.
        d_nat = np.linalg.norm(emp - vcov_nat)
        d_raw = np.linalg.norm(emp - vcov_raw)
        assert d_nat < 0.25 * d_raw, f"d_nat={d_nat:.5f} d_raw={d_raw:.5f}"


# --------------------------------------------------------------------- on-device

def _accelerators() -> list[str]:
    """Available non-CPU torch devices (CUDA and/or MPS)."""
    devs = []
    if torch.cuda.is_available():
        devs.append("cuda")
    if bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


@pytest.mark.skipif(not _accelerators(), reason="No CUDA/MPS device available")
class TestNaturalCoordinateDrawOnDevice:
    """Validates the fix on the actual GPU device path (CUDA and/or MPS) at the
    FP32 tolerance tier — the fit, Jacobian and draw all run on-device through the
    device-split linear algebra (``_gpu_spd``). The covariance is estimated with
    more draws to absorb FP32 + GPU-RNG noise; the natural-vs-raw discrimination
    still holds. This is the covariance-sensitive check on the real
    ``mice(..., backend='gpu')`` draw path that a proportions check cannot make."""

    @pytest.mark.parametrize("device", _accelerators())
    def test_draw_cov_matches_cpu_natural_vcov(self, device):
        y, X, K = _ordered_dataset(seed=3)
        dev = torch.device(device)
        fit, alpha, beta, raw, L = _fit_cpu_and_gpu(y, X, K, dev, torch.float32)
        emp = _empirical_draw_cov(alpha, beta, raw, L, dev, n_draws=120000, seed=0)
        vcov_nat = fit.vcov
        vcov_raw = _raw_cov(L)
        rel = np.linalg.norm(emp - vcov_nat) / np.linalg.norm(vcov_nat)
        assert rel < 0.10, f"{device} draw cov not natural vcov (rel frob {rel:.4f})"
        d_nat = np.linalg.norm(emp - vcov_nat)
        d_raw = np.linalg.norm(emp - vcov_raw)
        assert d_nat < 0.25 * d_raw, f"d_nat={d_nat:.5f} d_raw={d_raw:.5f}"
