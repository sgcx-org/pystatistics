"""
Separation/precision regression for the GPU discrete-GLM imputers (issue #8).

The GPU ``logreg`` (binary), ``polyreg`` (multinomial) and ``polr`` (ordinal)
fits are ill-conditioned under the (quasi-)separation chained equations routinely
induce. Run in FP32 the fit and posterior draw lose precision and the imputation
**silently collapses** every missing cell onto a single category — binary to
all-0 (``u < NaN`` is False), ordinal/nominal to category 0 (``argmax`` of an
all-False mask). On a mixed sweep the collapsed column then feeds a constant
predictor into every other column, so the damage is not local: it was measured
as a total-variation ~0.5 (vs ~0.1 in FP64) on the GSS mixed problem even after
the per-column ordinal fit was made finite.

The fix computes these discrete-GLM fits in FP64 where the device supports it
(``discrete_glm_compute_dtype``; MPS keeps FP32) and makes the samplers fail
loud — a non-finite probability yields NaN, never a silent category, so a
genuinely degenerate fit reaches the backend's end-of-sweep guard.

This module pins both halves: the fail-loud samplers (device-agnostic) and an
integrated on-device mixed sweep that must not collapse any column. The prior
suite exercised only balanced data and per-column fits, so it missed the
sweep-level collapse.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pystatistics.mice import mice
from pystatistics.mice.backends._gpu_logreg import gpu_logreg_impute
from pystatistics.mice.backends._gpu_polyreg import (
    _sample_categories as poly_sample_categories,
)
from pystatistics.mice.design import MICEDesign


def _accelerators() -> list[str]:
    devs = []
    if torch.cuda.is_available():
        devs.append("cuda")
    if bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


# ----------------------------------------------------------------- fail-loud

class TestSamplersFailLoud:
    """The categorical samplers emit NaN (never a silent category 0/all-0) on
    non-finite probabilities, so a degenerate fit reaches the backend guard."""

    def test_polyreg_sampler_non_finite_row_yields_nan(self):
        probs = torch.ones(1, 3, 4) / 4.0
        probs[0, 1, :] = float("nan")
        gen = torch.Generator()
        gen.manual_seed(0)
        out = poly_sample_categories(probs, gen)
        assert torch.isnan(out[0, 1])
        assert torch.isfinite(out[0, 0]) and torch.isfinite(out[0, 2])

    def test_logreg_non_finite_predictor_yields_nan(self):
        """A non-finite predicted probability must surface as NaN, not a silent 0.
        (A non-finite fit drives ``eta`` -> NaN; ``eta`` is clamped, which tames
        +/-inf but propagates NaN, so a NaN predictor is the faithful proxy.)"""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 2)).astype(np.float64)
        y = (X[:, 0] > 0).astype(np.float64)
        X_mis = X[:5].copy()
        X_mis[0, 0] = np.nan  # -> NaN eta -> NaN p for row 0
        gen = torch.Generator()
        gen.manual_seed(0)
        out = gpu_logreg_impute(
            torch.tensor(y).unsqueeze(0),
            torch.tensor(X).unsqueeze(0),
            torch.tensor(X_mis).unsqueeze(0),
            gen,
        )
        assert torch.isnan(out[0, 0]), "non-finite predictor must yield NaN"
        assert torch.isfinite(out[0, 1:]).all()


# --------------------------------------------------------------------- on-device

def _mixed_separated_problem(seed: int, n: int = 3000):
    """A mixed-type problem engineered to drive the discrete fits into
    (quasi-)separation: a binary column near-perfectly ordered by a continuous
    predictor (logreg separation), and a 7-level ordered column with a near-empty
    intermediate category (polr). Numeric columns are well-conditioned. Returns
    ``(X, column_kinds)`` with ~20% missing completely at random per column."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    # binary near-separated by x
    b = (x > 0.8).astype(float)
    # ordered (7 levels) with a near-empty interior category, ordered by x
    cuts = np.quantile(x, [0.05, 0.07, 0.93, 0.95, 0.952, 0.99])
    o = np.digitize(x, cuts).astype(float)
    full = np.column_stack([b, o, x, z])
    kinds = ["binary", "ordered", "numeric", "numeric"]
    X = full.copy()
    # Missing only in the two discrete columns (the fits under test); keep the
    # numeric predictors fully observed so no row is all-missing and every
    # discrete fit has informative predictors.
    for j in (0, 1):
        X[rng.random(X.shape[0]) < 0.20, j] = np.nan
        X[:2, j] = full[:2, j]  # keep >=2 observed per column
    return X.astype(np.float64), kinds


@pytest.mark.skipif(not _accelerators(), reason="No CUDA/MPS device available")
class TestMixedSweepNoCollapse:
    """The default GPU sweep must not collapse any imputed column on a mixed
    problem with separated discrete columns — the issue #8 failure mode."""

    @pytest.mark.parametrize("device", _accelerators())
    def test_no_column_collapses(self, device):
        X, kinds = _mixed_separated_problem(seed=0)
        # MPS has no FP64; CUDA uses the double-precision path.
        gpu_backend = "gpu" if device == "mps" else "gpu_fp64"
        design = MICEDesign.from_array(X, method="auto", column_kinds=kinds)
        sol = mice(design, n_imputations=3, max_iter=5, seed=0, backend=gpu_backend)
        for col in sol.incomplete_columns:
            imp = sol.imputations(col)
            assert np.isfinite(imp).all(), f"col {col} produced non-finite imputations"
            # A collapsed fit imputes a single constant; a real fit varies.
            if kinds[col] in ("binary", "ordered"):
                assert np.unique(np.rint(imp)).size >= 2, (
                    f"col {col} ({kinds[col]}) collapsed to a single category"
                )

    @pytest.mark.parametrize("device", _accelerators())
    def test_default_precision_matches_fp64(self, device):
        """On FP64-capable devices the discrete fits run in FP64 internally, so
        the *default* (FP32 sweep) ordered-category proportions track the FP64
        sweep — i.e. the default no longer silently degrades."""
        if device == "mps":
            pytest.skip("MPS has no FP64 to compare against")
        X, kinds = _mixed_separated_problem(seed=0)
        ordered = [j for j, k in enumerate(kinds) if k == "ordered"]

        def run(fp64):
            d = MICEDesign.from_array(X, method="auto", column_kinds=kinds)
            return mice(d, n_imputations=4, max_iter=5, seed=0,
                        backend="gpu_fp64" if fp64 else "gpu")

        sol32, sol64 = run(False), run(True)
        for col in ordered:
            K = int(np.nanmax(X[:, col])) + 1
            p32 = np.bincount(
                np.rint(sol32.imputations(col).ravel()).astype(int), minlength=K
            ).astype(float)
            p64 = np.bincount(
                np.rint(sol64.imputations(col).ravel()).astype(int), minlength=K
            ).astype(float)
            p32 /= p32.sum()
            p64 /= p64.sum()
            tv = 0.5 * float(np.abs(p32 - p64).sum())
            assert tv < 0.1, f"{device} col {col}: default-vs-fp64 TV={tv:.3f}"
