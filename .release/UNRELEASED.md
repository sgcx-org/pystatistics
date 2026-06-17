# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- MICE GPU backend is ~1.6–2.4x faster on Apple Silicon (MPS), at no cost to
  accuracy (imputations unchanged within the GPU/FP32 tolerance; CUDA path
  unaffected). Two changes combine:
  1. The batched per-sweep-step posterior draw no longer forces a GPU<->CPU
     synchronization on every step (it previously did, via a per-step Cholesky
     check). Degenerate (near-collinear) predictors are now caught by a single
     non-finite check at the end of the sweep — same fail-loud behaviour, far
     less overhead.
  2. For larger problems on MPS the draw inverts the predictor Gram's Cholesky
     factor with a matmul-only series (a few large matmuls) instead of
     `solve_triangular`, whose MPS kernel is ~250x slower than its matmul.
  Measured (p=20, m=100, idle machine): ~1.8x at n=2000, ~2.0x at n=8000,
  ~1.7x at n=20000.

- Added shared triangular-inverse primitives to `core.compute.linalg`
  (`batched_tri_inv`, used by the MVNMLE GPU objective, and the faster
  `batched_tri_inv_series` used by the MICE GPU draw on MPS), plus an MPS-only
  device dispatch. On MPS the MICE draw uses the series inverse above a problem-
  size threshold (observed rows >= 3000), where it beats `solve_triangular`;
  smaller problems and the CUDA/CPU paths keep the triangular solve. No change to
  results.
