# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **GPU GLM (`regression.fit(..., backend='gpu')`) no longer falsely rejects
  correct float32 fits.** The unpenalized float32 IRLS path held the strict
  float64 convergence tolerance and declared non-convergence — failing loud —
  whenever the deviance plateaued at the float32 round-off floor, even when the
  fit was correct and well-conditioned. This misfired most on Apple Silicon
  (MPS, noisier float32) and made `survival.discrete_time(backend='gpu')`
  intermittently unusable at moderate n. Acceptance is now decided by the
  **relative Newton decrement** at the final coefficients (a stationarity test,
  evaluated in float64): a float32 fit that has reached a genuine optimum at the
  float32 floor is accepted and matches the CPU fit to the float32 tier; a
  non-stationary fit, or one whose float32 inner solve breaks down, still fails
  loud. No silent CPU fallback or precision downgrade — the error names the
  explicit options (`backend='cpu'`, `backend='gpu_fp64'`, ridge, or
  `force=True`). Verified on MPS and CUDA across binomial/Poisson/Gamma.

- **`survival.discrete_time` now accepts `backend='gpu_fp64'`** (CUDA-only
  exact double-precision GPU path), in addition to `'cpu'`/`'gpu'`/`'auto'`.
  Forwarded to the person-period logistic regression; on Apple Silicon it raises
  the canonical CUDA-required error (no silent fallback).

- **Constitution (CONVENTIONS.md): added amendment A6** — "PyStatistics does not
  'help'." An explicit request is honored exactly or fails loud with the
  reason; it is never silently substituted with a different device, precision,
  algorithm, or estimator. The rule has two symmetric halves: do not silently
  help (no silent CPU fallback), and do not falsely refuse (a correct float32
  fit at the float32 floor must be accepted — correctness checks are calibrated
  to the requested precision). The GPU GLM acceptance fix above resolves a
  latent violation of the second half.
