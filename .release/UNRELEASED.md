# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py <version>`
> That script uses this file to build the CHANGELOG entry, bumps versions
> everywhere, and resets this file for the next cycle.

## Changes

### BREAKING: Removed MoM MCAR test and the entire `nonparametric_mcar` subpackage

`pystatistics.mvnmle.mom_mcar_test` and the full
`pystatistics.nonparametric_mcar` subpackage (propensity / HSIC /
MissMech + their shared `NonparametricMCARResult`) have been removed.
These were added in 2.2.0 and 2.3.0 respectively, specifically in
service of Lacuna's missingness-mechanism cache, NOT as general-
purpose statistical methods. That was a scope-boundary mistake on
pystatistics's part: we're a general-purpose statistics library, and
these tests are project-specific feature-extraction helpers.

**What remains in pystatistics:**
  - `pystatistics.mvnmle.little_mcar_test` — the canonical Little (1988)
    MLE-plug-in test. Textbook, general-purpose, stays.
  - `MCARTestResult` dataclass — retained (downstream packages that
    implement their own MCAR variants can and do reuse it).
  - All of `mlest`, the EM / SQUAREM machinery, `analyze_patterns`,
    `PatternInfo`, etc. — unchanged.

**Where the removed tests live now:** the cache-scale variants have
moved into Lacuna as `lacuna.analysis.mcar`:

  - `lacuna.analysis.mcar.mom_mcar_test` (method-of-moments plug-in)
  - `lacuna.analysis.mcar.propensity_mcar_test` (RF/GBM + analytical or
    permutation null)
  - `lacuna.analysis.mcar.hsic_mcar_test` (Gretton gamma or permutation null)
  - `lacuna.analysis.mcar.missmech_mcar_test` (Jamshidian-Jalal-style)
  - `lacuna.analysis.mcar.NonparametricMCARResult`

All the 2.3.0/2.4.0 performance work that was in-flight in this file
(analytical AUC p-value for propensity, HGB default, Gretton gamma
null for HSIC, vectorised MissMech permutation, etc.) moved with the
code. Downstream: Lacuna's cache build is unaffected — it now imports
from `lacuna.analysis.mcar` instead of `pystatistics.nonparametric_mcar`.

**Migration for external users (if any):**
  - `from pystatistics.mvnmle import mom_mcar_test` →
    `from lacuna.analysis.mcar import mom_mcar_test` (if using Lacuna).
    Alternatively, the MoM algorithm is ~100 LOC of pairwise-deletion
    moments + per-pattern chi-square and is straightforward to inline
    if you don't want a Lacuna dependency.
  - `from pystatistics.nonparametric_mcar import *` →
    `from lacuna.analysis.mcar import *`.

**Why a major bump (3.0.0):** public API surface shrunk; imports that
worked in 2.x now fail. Even though our active user population is
small (internal), the convention matters — breaking removals are
semver-major.

**Dropped files:**
  - `pystatistics/nonparametric_mcar/` (subpackage, 5 files)
  - `pystatistics/mvnmle/mcar_test.py` lost `mom_mcar_test`,
    `_resolve_mom_backend`, `_pairwise_deletion_moments`, and the
    `_MOM_GPU_WORTH_IT_THRESHOLD` constant. `little_mcar_test` and
    `MCARTestResult` remain.
  - `pyproject.toml` lost the `nonparametric_mcar` optional extra.
  - `tests/mvnmle/test_mom_mcar.py` and `tests/nonparametric_mcar/`.

(The dead batched MCAR chi-square machinery that this removal left
stranded has also been cleaned up — see the "Dead-code cleanup"
section below.)

### Fixed flaky GAM GPU FP64 `total_edf` test

`tests/gam/test_gam.py::TestGAMGPU::test_gpu_fp64_matches_cpu_fitted_and_gcv`
had been flaking intermittently since 1.8.0. Measured CPU/GPU drift on
the `sine_data` fixture:

- `fitted_values` max abs diff: ~7e-8 (machine precision)
- `deviance` relative diff: ~1e-7
- `gcv` relative diff: ~3e-5
- `total_edf` relative diff: **~1.4e-3** (narrowly failing the 1e-3 bar)

Diagnosis: the primary fit statistics (fitted_values, deviance, GCV)
sit at the GCV-minimising λ, i.e. at a local minimum of `GCV(λ)`, so
they are insensitive to the tiny λ-drift the L-BFGS-B outer search
lands on across CPU/GPU. `total_edf`, however, passes through a trace
of `(X'WX + λ·S)⁻¹·X'WX`, which is LINEAR in λ near the optimum. With
the penalised normal-equation matrix at cond ≈ 1e16-17 the EDF trace
moves O(10⁻³) per O(10⁻⁶) λ shift — structurally more sensitive than
the fit statistics that a 1e-4 tolerance gated.

Fix: tolerance widened from `rel=1e-3` to `rel=5e-3` on `total_edf`
only. The other three assertions (fitted_values, deviance, GCV) keep
their 1e-4 margin. Test comment updated with the λ-sensitivity
analysis so future maintainers don't re-tighten it without
understanding why it's at 5e-3.

### Fixed GAM GPU smooth-term chi-squared bug (null-space absorption)

Surfaced while tightening the `total_edf` tolerance above: the chi-
squared statistic reported in `smooth_terms` was diverging ~8×
between CPU and GPU backends — CPU ≈ 19, GPU ≈ 156 on the
`sine_data` fixture — despite fitted values agreeing to FP64
precision. This was a latent correctness bug in the GPU backend
going back to whenever the GAM GPU path was first written.

Root cause: the penalised normal matrix `A = X'WX + Σ λⱼ Sⱼ` has
condition number up to ~1e17 when λ is small — the penalty does not
fully eliminate the design matrix's null space (the constant / linear
directions in the spline basis's null space). On such `A`,
`torch.linalg.solve` (LU on device) and `np.linalg.cholesky`
(CPU reference path) converge to the same `X·β` (fitted values) but
pick DIFFERENT null-space-representative `β` (coefficients).
Concretely on `sine_data`: fitted values agreed to 7e-8 but
coefficients were shifted by a constant ±1.73 between the parametric
intercept and every smooth basis coefficient — exactly the null-space
direction the penalty fails to constrain. Since `smooth_terms.chi_sq`
is computed directly from `β` (via `β_j'·β_j / scale`), the null-
space shift broke it despite the rest of the fit being numerically
sound.

Fix: the GPU backend now canonicalises the final `β` by re-solving
`A·β = b` via numpy's Cholesky-with-LU-fallback — the same logic
the CPU path uses in `_pirls_step`. This lives in a new
`_canonicalise_beta` method; `fit_fixed` and `edf_per_term` both call
it after their P-IRLS loops converge. In-loop P-IRLS still uses
torch's fast LU on device (null-space ambiguity doesn't matter
there — only the resulting `μ` drives convergence). The D2H
round-trip for the final solve is ~100 µs on a p×p matrix with p
typically ≤ 80; the dominant GPU work (the n×p GEMM in each P-IRLS
step) stays on device.

Test: `TestGAMGPU::test_gpu_fp64_matches_cpu_fitted_and_gcv` gained
coefficient-level assertions (`rtol=1e-3, atol=1e-4`) and chi-squared
agreement (`rel=1e-3`) so this regression is caught if it returns.
Measured post-fix: coef max abs diff drops from 1.73 to 9e-5;
chi_sq rel diff drops from ~8× to 1.6e-5.

### Dead-code cleanup

While removing ``mom_mcar_test``, noticed that the batched MCAR
chi-square machinery (``chi_square_mcar_batched_np`` and
``chi_square_mcar_batched_torch`` in ``backends/_em_batched_{np,torch}.py``,
re-exported via the shim at ``backends/_em_batched.py``) was only ever
called from ``mom_mcar_test``. Those functions are now gone; shim
dropped their re-exports; backend file docstrings updated to record
the removal. Net: `_em_batched_np.py` went from 416 → 300 lines and
`_em_batched_torch.py` from 381 → 271 lines, and `backends/_em_batched.py`
lost two of its eleven shim exports.

### Process change

Added **Rule 9 "Cross-Project Scope Boundary"** to
`CLAUDE.md`. Future Claude Code sessions working on pystatistics must
not modify sibling projects (Lacuna, pystatsbio, etc.) without
explicit per-session user authorisation. Functionality needed by
another project belongs in that project until demonstrably general.
This removal is the retroactive correction of previous boundary
crossings. The same rule was added to Lacuna's and pystatsbio's
CLAUDE.md files (§8 and §9 respectively, numbered by where the rule
fit in each file's existing structure).
