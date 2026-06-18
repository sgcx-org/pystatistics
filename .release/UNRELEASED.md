# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- Fixed the GPU MICE `polr` (proportional-odds) posterior draw to use the
  **natural** threshold parameterization, matching the CPU `polr` /
  `MASS::polr`. The batched GPU backend (`mice(..., backend='gpu')`, CUDA and
  MPS) drew the threshold+slope posterior with the covariance on the raw
  (log-gap) parameterization (`theta* ~ N(natural_mean, vcov_raw)`), the same
  error fixed for the CPU path in 3.15.1 but never applied to the separate GPU
  implementation. `pystatistics/mice/backends/_gpu_polr.py` now maps the
  raw-coordinate Hessian draw deviation to natural coordinates via the
  raw→natural Jacobian (delta method) before adding it to the natural mean, so
  the between-imputation variance of ordered-factor imputations — and the
  Rubin's-rules variance / interval coverage that depends on it — is now
  correct. Marginal category proportions are unchanged (they are insensitive to
  the threshold-draw covariance). Added `tests/mice/test_gpu_polr_draw.py`,
  which pins the GPU draw covariance to the CPU natural-coordinate `vcov` and
  guards against the raw-coordinate regression; tightened the GPU-vs-CPU `polr`
  category-proportion tolerance.
