# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- MICE `backend='gpu'` now runs on Apple Silicon (MPS), not just CUDA. Previously
  `mice(..., backend='gpu')` raised `NotImplementedError` on a Mac; it now runs
  the batched chained-equations sweep on the MPS GPU in FP32. `backend='auto'`
  still selects CPU on a Mac (never MPS) for consistency with the rest of the
  library — opt into the Mac GPU explicitly with `backend='gpu'`. `use_fp64=True`
  is rejected on MPS (Apple's GPU has no float64); use FP32 there or a CUDA GPU
  for double precision. Validated against the CPU reference at the GPU/FP32
  tolerance tier (imputed-value distribution, Rubin's-rules pooled estimates,
  exact PMM donor copies, determinism) for both `pmm` and `norm`. Measured ~12×
  faster than the CPU backend at n=20000, p=20, m=100 (3.3 s vs 42 s).

- MICE GPU posterior draw reworked to factor the ridged Gram matrix once with a
  Cholesky and solve via triangular substitution, instead of computing a separate
  matrix `solve` and inverse. This removes the matrix inverse and the eigenvalue
  fallback from the hot path — it is faster on CUDA as well as MPS (the inverse
  was redundant work) and is better conditioned. A degenerate (near-collinear)
  predictor design now fails loud after ridge + jitter rather than silently
  returning a clipped factor.

- MICE GPU PMM donor search now selects the k nearest donors via a contiguous
  sorted-window block (the window is already sorted, so the nearest k are
  contiguous) instead of `topk`, which is slow on both MPS and CUDA. On MPS the
  donor insertion-rank is computed from a combined sort rather than
  `searchsorted` (which is pathologically slow on MPS at scale); CUDA keeps
  `searchsorted`. Donor results are unchanged.
