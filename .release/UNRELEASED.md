# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- Fixed GPU MICE imputation silently collapsing categorical/ordinal columns onto
  a single category on imbalanced real-world data (the 3.16.1 line-search fix
  addressed a different, synthetic failure mode and did not cover this). Two
  root causes, both verified on CUDA against the GSS mixed-survey problem:
  1. **Non-convex ordinal Hessian.** `batched_polr_newton`
     (`mice/backends/_gpu_polr.py`) optimizes in the raw (log-gap)
     parameterization, where the proportional-odds NLL is non-convex, so the
     observed Hessian is indefinite away from the optimum. The fixed tiny ridge
     could not Cholesky-factor it (cuSOLVER returned a non-finite factor),
     stalling the step and poisoning the posterior-draw factor — `alpha` ran to
     NaN and every missing row was imputed as one category. Replaced the fixed
     ridge with per-chain Levenberg-Marquardt damping (`_pd_cholesky`) that
     escalates until the matrix is positive definite (a true descent step), plus
     an `exp` overflow clamp in the raw->threshold transform and a finite-trial
     guard in the line search.
  2. **FP32 precision.** The logistic/multinomial/proportional-odds fits are
     ill-conditioned under (quasi-)separation; in FP32 the fit and draw lost all
     precision and the imputation collapsed (binary -> all-0, ordinal/nominal ->
     one level), which then corrupted every column using it as a predictor. The
     GPU `logreg`, `polyreg` and `polr` fits now compute in FP64 where the device
     supports it (`discrete_glm_compute_dtype`; MPS keeps FP32) regardless of the
     sweep's global precision. Measured GPU-vs-CPU total variation on the GSS
     ordered columns dropped from ~0.93 (collapsed) / ~0.55 to ~0.10 — the CPU's
     own seed-to-seed run-to-run variance is ~0.08-0.17, so the GPU now tracks
     the CPU reference at the stochastic floor.
- Fail loud (Rule 1): the GPU `logreg`, `polyreg` and `polr` samplers now emit
  NaN on a non-finite probability instead of defaulting to category 0 (binary
  `u < NaN` and `argmax` of an all-False mask both silently returned 0), so a
  genuinely degenerate fit propagates to the imputations and is caught by the
  backend's end-of-sweep non-finite guard rather than masquerading as valid
  category-0 draws.
- Added GPU separation/precision regression tests
  (`tests/mice/test_gpu_glm_separation.py`, `tests/mice/test_gpu_polr_separation.py`):
  near-empty-intermediate-category ordinal fits stay finite/bounded and match the
  CPU ridged fit; the samplers fail loud on non-finite probabilities; and an
  integrated on-device (CUDA/MPS) mixed sweep with separated binary + ordinal
  columns must not collapse any column. The prior suite exercised only balanced
  per-column data and so missed the sweep-level collapse.
