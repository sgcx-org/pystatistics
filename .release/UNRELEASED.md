# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **GPU MVN MLE: pattern-chunking for bounded memory (scales to wide data).**
  The batched objective held O(P·v²) memory — all P patterns' (P, v, v) tensors
  at once — which exhausted GPU memory for wide data with many distinct patterns
  (e.g. p=100 survey data with ~43k patterns → CUDA out-of-memory on a 16 GB
  card). `_batched_cholesky.py` now provides `objective_value` (chunked forward
  sum) and `accumulate_gradient` (per-chunk forward+backward — grad of a sum is
  the sum of grads — freeing each chunk's autograd graph), plus `auto_chunk_size`
  / `chunk_bounds`. `gpu_fp32.py` / `gpu_fp64.py` route
  `compute_objective`/`compute_gradient` through them with an auto-sized
  `chunk_size` (override via the objective constructor). Peak memory is one
  chunk's tensors rather than the whole pattern set; value and gradient are
  identical to the unchunked computation (tested). Verified on CUDA (RTX 5070 Ti):
  wvs/gss p=100 now complete (~260–300 s) where they previously raised CUDA OOM.
