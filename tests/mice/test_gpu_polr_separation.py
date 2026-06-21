"""
Separation regression for the GPU ``polr`` fit (issue #8).

Chained equations routinely drive an ordered column into (quasi-)complete
separation against a continuous predictor: a sparse extreme category that a
numeric covariate orders (nearly) perfectly. The proportional-odds MLE is then
unbounded, and the GPU batched Newton — a full-step Newton with no
globalisation — overshoots the *unpenalised thresholds* into the saturated tail
of the cumulative logits on the very first step, where the gradient vanishes and
the iterate sticks at a degenerate ``|alpha| ~ 1e4..1e6`` fit that assigns every
missing row a single category. On the GSS mixed problem this collapsed every
ordered column (total-variation ~0.93 vs R mice).

The fix has two complementary parts in ``backends/_gpu_polr.py``:

  * a scale-aware objective ridge on the SLOPES (mirroring the CPU
    ``methods/polr._slope_ridge``), and
  * a per-chain backtracking line search that damps each Newton step until the
    penalised NLL decreases — the part that actually bounds the thresholds.

This module pins the fixed behaviour on a deterministically separated ordinal
(self-contained; no GSS dependency): the fit stays finite and bounded, lands on
the CPU penalised MLE, and the end-to-end imputed-category proportions track the
CPU ``polr`` (which is R-validated). The existing balanced-data suite missed the
bug because a well-identified fit never separates; this case has teeth — the
pre-fix code reaches ``|alpha| > 1e4`` here.

Device-agnostic FP64 core (CPU torch, tight tolerance) plus an on-device variant
at the FP32 tolerance tier exercising the real ``mice(..., backend='gpu')`` path.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pystatistics.mice.backends._gpu_polr import (
    batched_polr_newton,
    gpu_polr_impute,
)
from pystatistics.mice.methods.polr import PolrMethod, _slope_ridge
from pystatistics.ordinal import polr

# The fixed fit lands on the CPU penalised MLE, whose largest threshold here is
# ~100 (separation -> large but finite). The bound is far below the pre-fix
# collapse (|alpha| > 1e4) yet comfortably above the legitimate finite fit, so a
# regression to the undamped step is caught while a correct fit passes.
_FINITE_BOUND = 1.0e3


def _separated_dataset(seed: int, n: int = 4000, K: int = 4):
    """Ordinal with a sparse extreme category perfectly ordered by a continuous
    predictor. ``y = digitize(x, quantile_cuts(x))`` makes ``x`` order the target
    deterministically (complete separation); the top cut at the 0.97 quantile
    makes the extreme category sparse — the GSS failure signature. ``z`` is an
    inert second predictor. All K levels are present by construction."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    cuts = np.quantile(x, [0.30, 0.60, 0.97])[: K - 1]
    y = np.digitize(x, cuts).astype(np.intp)
    for lv in range(K):  # guarantee every level is present among observed
        if not np.any(y == lv):
            y[lv] = lv
    return y, np.column_stack([x, z]), K


def _split(y, X, seed, n_mis):
    """Disjoint observed/missing split for the end-to-end imputation check."""
    idx = np.random.default_rng(seed).permutation(len(y))
    mis, obs = idx[:n_mis], idx[n_mis:]
    return y[obs], X[obs], X[mis], y[mis]


def _props(values, K):
    """Category proportions over 0..K-1 of rounded class indices."""
    counts = np.array([(np.rint(values) == lv).sum() for lv in range(K)], float)
    return counts / counts.sum()


def _tv(p, q):
    """Total-variation distance between two proportion vectors."""
    return 0.5 * float(np.abs(p - q).sum())


def _gpu_fit(y, X, K, device, dtype):
    """One-chain GPU batched-Newton fit on ``device``/``dtype``."""
    y_obs = torch.tensor(y, dtype=dtype, device=device).unsqueeze(0)
    X_obs = torch.tensor(X, dtype=dtype, device=device).unsqueeze(0)
    alpha, beta, _, _ = batched_polr_newton(y_obs, X_obs, K)
    return alpha[0], beta[0]


# ---------------------------------------------------------------- device-agnostic

class TestSeparatedFitCPU:
    """FP64 core (CPU torch): the damped Newton stays finite and reproduces the
    CPU penalised MLE on a perfectly separated ordinal."""

    def test_fit_finite_and_bounded(self):
        """Thresholds and slopes are finite and bounded — the pre-fix undamped
        step diverges to ``|alpha| > 1e4`` on exactly this data."""
        y, X, K = _separated_dataset(seed=0)
        alpha, beta = _gpu_fit(y, X, K, torch.device("cpu"), torch.float64)
        assert torch.isfinite(alpha).all() and torch.isfinite(beta).all()
        assert alpha.abs().max().item() < _FINITE_BOUND
        assert beta.abs().max().item() < _FINITE_BOUND

    def test_fit_matches_cpu_ridged(self):
        """The GPU fit lands on the CPU penalised MLE (same scale-aware slope
        ridge), threshold-by-threshold and slope-by-slope."""
        y, X, K = _separated_dataset(seed=0)
        fit = polr(y, X, ridge=_slope_ridge(X.astype(np.float64)))
        alpha, beta = _gpu_fit(y, X, K, torch.device("cpu"), torch.float64)
        np.testing.assert_allclose(
            alpha.numpy(), fit.threshold_values, atol=1e-2, rtol=1e-3
        )
        np.testing.assert_allclose(
            beta.numpy(), fit.coefficients, atol=1e-2, rtol=1e-3
        )

    def test_imputed_proportions_track_cpu(self):
        """End-to-end: GPU ``polr`` imputed-category proportions track the CPU
        ``polr`` (R-validated), recovering the sparse extreme category instead of
        collapsing onto one category."""
        y, X, K = _separated_dataset(seed=0)
        y_obs, X_obs, X_mis, _ = _split(y, X, seed=0, n_mis=1500)
        cpu_imp = PolrMethod().impute(y_obs, X_obs, X_mis, np.random.default_rng(1))
        gen = torch.Generator(device="cpu")
        gen.manual_seed(1)
        gpu_imp = gpu_polr_impute(
            torch.tensor(y_obs, dtype=torch.float64).unsqueeze(0),
            torch.tensor(X_obs, dtype=torch.float64).unsqueeze(0),
            torch.tensor(X_mis, dtype=torch.float64).unsqueeze(0),
            gen,
            n_classes=K,
        )[0].numpy()
        p_cpu, p_gpu = _props(cpu_imp, K), _props(gpu_imp, K)
        # The sparse extreme category must carry real (non-zero) mass on both.
        assert p_gpu[K - 1] > 0.5 * p_cpu[K - 1] > 0.0
        assert _tv(p_cpu, p_gpu) < 0.05, f"TV(cpu, gpu) = {_tv(p_cpu, p_gpu):.4f}"


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
class TestSeparatedFitOnDevice:
    """Validates the fix on the real GPU device path at the FP32 tolerance tier:
    the on-device damped Newton stays bounded and the imputed proportions track
    the CPU ``polr`` despite FP32 + GPU-RNG noise."""

    @pytest.mark.parametrize("device", _accelerators())
    def test_fit_finite_and_bounded(self, device):
        y, X, K = _separated_dataset(seed=0)
        alpha, beta = _gpu_fit(y, X, K, torch.device(device), torch.float32)
        assert torch.isfinite(alpha).all() and torch.isfinite(beta).all()
        assert alpha.abs().max().item() < _FINITE_BOUND
        assert beta.abs().max().item() < _FINITE_BOUND

    @pytest.mark.parametrize("device", _accelerators())
    def test_imputed_proportions_track_cpu(self, device):
        y, X, K = _separated_dataset(seed=0)
        y_obs, X_obs, X_mis, _ = _split(y, X, seed=0, n_mis=1500)
        cpu_imp = PolrMethod().impute(y_obs, X_obs, X_mis, np.random.default_rng(1))
        dev = torch.device(device)
        gen = torch.Generator(device=dev)
        gen.manual_seed(1)
        gpu_imp = gpu_polr_impute(
            torch.tensor(y_obs, dtype=torch.float32, device=dev).unsqueeze(0),
            torch.tensor(X_obs, dtype=torch.float32, device=dev).unsqueeze(0),
            torch.tensor(X_mis, dtype=torch.float32, device=dev).unsqueeze(0),
            gen,
            n_classes=K,
        )[0].cpu().numpy()
        p_cpu, p_gpu = _props(cpu_imp, K), _props(gpu_imp, K)
        assert p_gpu[K - 1] > 0.5 * p_cpu[K - 1] > 0.0
        assert _tv(p_cpu, p_gpu) < 0.06, f"{device} TV = {_tv(p_cpu, p_gpu):.4f}"
