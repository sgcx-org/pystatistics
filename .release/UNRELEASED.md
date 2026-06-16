# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **GPU MVN MLE on Apple Silicon (MPS): matmul-only blocked inversion for the
  per-pattern trace term.** Metal optimizes batched `cholesky` and matmul but
  runs the triangular-solve / inverse family ~300x slower, which made the trace
  term `tr(Sigma_k^{-1} M_k)` the bottleneck of the batched objective on MPS. The
  trace is now computed by forming `Sigma_k^{-1}` via a divide-and-conquer
  blocked inversion of the Cholesky factor (`_tri_inv_blocked` in
  `_objectives/_batched_cholesky.py`) — recursing to a closed-form 2x2 base, so
  the whole computation is matmul + elementwise, no triangular solve. Exact to
  floating precision, numerically stable across conditioning (validated to
  correlation 0.99 and v up to 200), autodiff-correct. CUDA/CPU continue to use
  triangular solves (fast there); the blocked path is selected only on MPS.
  End-to-end: an afrobarometer-derived problem at p=25 fits in ~8s, and a WVS
  problem at p=50 with ~20k distinct missingness patterns fits in ~5 min on an
  M-series GPU (previously impractical, solve-bound).
