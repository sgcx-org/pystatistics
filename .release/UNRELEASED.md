# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **`montecarlo` GPU backends: fixed a silently-wrong path where a non-mean
  statistic could be computed as the mean.** The GPU bootstrap/permutation
  backends previously *inferred* whether the user's `statistic` was the sample
  mean / difference-in-means by evaluating it on a single resample and comparing
  to the mean (tolerance `1e-10` for bootstrap; `1e-4` on one permutation for the
  permutation test). A statistic that matched at that single point but differed
  elsewhere (e.g. a lightly-trimmed mean) would be silently replaced by the mean
  for all replicates on `backend='gpu'` — a wrong result with no warning. The
  one-sample inference is removed. `boot()` and `permutation_test()` now take an
  explicit `gpu_statistic` declaration (`'mean'` / `'mean_diff'`), and the GPU
  path is fail-loud (Guarantee 2):
    - `backend='gpu'` without `gpu_statistic` **raises** `ValidationError`
      (arbitrary Python statistics cannot run on the GPU) — it no longer silently
      falls back to CPU.
    - `backend='gpu'` on a non-vectorizable configuration (balanced/parametric
      bootstrap, strata, 2-D data / 2-D groups) **raises**.
    - A declared statistic that does not match the observed value on the original
      data (`statistic(data, all-indices) != mean(data)`, or
      `statistic(x, y) != mean(x)-mean(y)`) **raises** — it is never silently
      computed as the mean.
    - `backend='auto'` with no GPU-supported declaration runs on the CPU
      (disclosed via `solution.backend_name`); `auto` never silently computes a
      different statistic.
  Files: `montecarlo/backends/gpu.py`, `montecarlo/solvers.py`,
  `montecarlo/design.py`. Measured GPU speedups where the path applies (Apple
  MPS): ~6.9× for the bootstrap mean (n=5000, R=50000) and ~4–7.6× for the
  permutation mean-difference vs the CPU backend. `backend='cpu'` (the default)
  is unaffected.
