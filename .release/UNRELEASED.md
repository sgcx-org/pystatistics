# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- MICE GPU backend now supports **categorical predictors** for numeric-target
  imputation. Previously `backend='gpu'` refused any data containing a
  categorical column; it now treatment-dummy-encodes categorical predictor
  columns (matching the CPU path) and only refuses categorical *targets*
  (incomplete categorical columns), with a clear message. Numeric-only problems
  are unchanged. Categorical-target imputation on GPU is still pending.

- Internal (no user-facing change): unified the GPU batched triangular-factor
  inverse onto one primitive. The MVNMLE GPU objective now uses the same
  matmul-series inverse as the MICE GPU draw
  (`core.compute.linalg.batched_tri_inv_series`); the older block-recursion
  inverse (`batched_tri_inv`) was removed. The series inverse is now autograd-safe
  — a differentiable Newton step from a detached, already-accurate iterate yields
  the exact matrix-inverse gradient (matches a `solve_triangular` oracle to ~5e-16)
  — so it is a full drop-in. MVNMLE GPU results are unchanged within the GPU/FP32
  tolerance (MPS end-to-end vs CPU: max |Δmu| ~6e-5, max |ΔSigma| ~5e-4); CUDA/CPU
  paths are unaffected.
