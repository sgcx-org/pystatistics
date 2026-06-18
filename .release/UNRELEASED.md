# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- Fixed an over-tight assertion in the ordinal `polr` GPU FP64 test
  (`test_gpu_fp64_matches_cpu_coefs`). It compared GPU FP64 coefficients to
  the CPU fit at `rtol=1e-8`, but the CPU path stops on a half-Newton-decrement
  criterion that bounds the remaining log-likelihood gain, not the coefficients
  — leaving the CPU estimate up to ~1e-6 from the exact MLE on steep problems,
  while the GPU FP64 estimate is actually closer to it. Both backends converge
  to the same unique MLE under tight Newton iteration. Loosened the coefficient
  comparison to `rtol=1e-5` (the decrement-limited accuracy of the comparison);
  the standard-error check is unchanged. No library behaviour change — test and
  documentation only. Resolves a CUDA-only failure observed on NVIDIA Blackwell
  (sm_120) GPUs.
