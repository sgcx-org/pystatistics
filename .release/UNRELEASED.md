# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **`lmm()` fixed-effect standard errors are now O(p³) instead of O(n³).**
  `_compute_se` previously formed the dense n×n matrix `V* = ZΛΛ'Z' + I` and
  solved it to get `Var(β̂) = σ²(X'V*⁻¹X)⁻¹` — an O(n³) step that dominated
  `lmm()` runtime (~83% at large group counts). It now computes the same
  variance from the p×p Schur factor already produced by the PLS solve:
  `Var(β̂) = σ²·(RX⁻¹)ᵀ(RX⁻¹)`. Results are machine-identical to the previous
  path (~1e-14) and match R/lme4; the speed-up at large n is several orders of
  magnitude (the n×n solve is gone entirely). No API change.

- **`lmm()` now reports boundary (singular) fits.** New `LMMSolution.is_singular`
  accessor and a `RuntimeWarning` on singular fits, mirroring lme4's
  `isSingular()`: the fit is flagged when a random-effects variance has
  collapsed to ~0 or an implied correlation has reached ±1 (detected as a
  Cholesky-factor diagonal at its 0 lower bound, tol 1e-4). The estimates are
  unchanged — still the correct boundary MLE — but users now get the same
  signal R gives them that the random-effects structure may be too rich for the
  data. Additive API (`is_singular`); nothing removed.

- **`lmm()` now uses a structure-exploiting solver and scales to large /
  crossed designs.** The deviance evaluation previously built a dense random-
  effects design `Z` (n × ΣJ_k·q_k) and a dense Gram each iteration, which made
  realistic group counts impossibly slow or out-of-memory (e.g. a single factor
  with 2000 groups took ~3 minutes and ~27 GB; crossed designs at scale were
  infeasible). `lmm()` now never materializes the dense `Z`:
  - a single grouping factor uses a batched per-group dense Cholesky (the
    penalized system is block-diagonal);
  - crossed / nested designs use a sparse factorization with a fill-reducing
    ordering (SuperLU, `MMD_AT_PLUS_A`).
  Estimates are identical to the previous solver at the same parameters (the
  REML/ML optimum is unchanged; agreement is to ~1e-11) and continue to match
  R/lme4. Measured: the 2000-group single-factor fit drops from ~197 s to
  ~0.05 s; a 5000-group / 100k-observation fit completes in ~0.1 s; crossed
  designs (e.g. 2000×1000 levels, 60k observations) that previously ran out of
  memory now fit. Implemented in pure numpy/scipy (no new dependency); the GLMM
  path is unchanged. No API change.

- **New model: `grm_lmm()` — a low-rank / GRM mixed model with a GPU backend.**
  A mixed model with a single variance component whose covariance is low-rank,
  `K = WW'/M` (a genomic relatedness matrix from M standardized markers, or any
  reduced-rank random effect). It fits by REML/ML, reports the fixed-effect
  table, the genetic/residual variance components, narrow-sense heritability
  `h² = σ²_g/(σ²_g+σ²_e)`, and genetic-value BLUPs (`GRMSolution`). Because its
  deviance reduces to a dense M×M Gram + n×M GEMMs — the cuBLAS/cuSOLVER regime —
  it exposes a `backend=` (`'cpu'` float64 reference, `'gpu'` float32,
  `'gpu_fp64'` CUDA-only exact, `'auto'`). This is a *separate, honestly-named
  model* for the genomics / quantitative-genetics audience; it is **not** the
  general `lmm`, and `lmm`/`glmm` remain CPU-only with no `backend=`. The float32
  GPU path forms `W'W`, so it is guarded: the conditioning of `W` is checked in
  float64 up front and a design past the float32-safe boundary is **refused
  loudly** (`NumericalError`) rather than returned quietly biased — use
  `backend='gpu_fp64'` or `backend='cpu'` for those, or `force=True` to bypass.
  CPU results match an independent double-precision reference to machine
  precision; the GPU float32 path matches the CPU reference at the `GPU_FP32`
  statistical-equivalence tier.
