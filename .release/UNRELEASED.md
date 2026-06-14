# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- GPU backend for `mice` (CUDA). `mice(..., backend='gpu')` runs the `m`
  imputation chains batched on the GPU as the leading tensor dimension: each
  sweep step is a batched Bayesian linear solve (`torch.linalg.solve` + batched
  Cholesky) and, for PMM, a batched nearest-neighbour donor search
  (`(m, n_mis, n_obs)` distances → `topk`). `backend='auto'` selects CUDA when
  available, else CPU (never MPS, matching the other modules). New `use_fp64`
  argument runs the GPU path in double precision on CUDA (default FP32).
- GPU results match the CPU reference distributionally at the GPU/FP32 tolerance
  tier (PMM donor copies are exact in FP64, exact-to-FP32 otherwise). Validated
  in `tests/mice/test_gpu.py` against the CPU path rather than R directly.
- Measured PMM speedups vs the CPU backend on an RTX 5070 Ti: ~39× at n=1000
  (p=8, m=20, maxit=8) and ~135× at n=3000 (p=10, m=20, maxit=8); the gain grows
  with n because the donor search dominates and batches well on the GPU.
- Internal: the MICE backend contract now takes a `seed` (each backend spawns
  its own RNG streams) instead of pre-spawned streams, so the CPU and GPU
  backends share one `run()` signature. No public API change.
- MPS (Apple Silicon) is not yet validated for MICE; `backend='gpu'` raises a
  clear error on MPS rather than running unverified. CUDA is supported.
