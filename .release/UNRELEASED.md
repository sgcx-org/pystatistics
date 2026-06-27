# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **Ridge (L2-penalized) regression.** New `fit(design, family=..., l2=lambda)` and a
  thin `ridge(X, y, lam=lambda, family=...)` wrapper. Works for LM and all GLM
  families. Predictors are standardized, the intercept is unpenalized, and `l2` is
  the penalty on the standardized scale — matching `MASS::lm.ridge` to ~1e-15 (LM)
  and `glmnet(alpha=0)` (GLM). Penalized fits report NaN standard errors / t / p
  values (not valid for a biased estimator) rather than misleading numbers. New
  module `regression/_penalty.py`; `Ridge` is structured so a future elastic-net
  solver can subsume it.
- **Ridge runs on the GPU at very large scale.** The ridge penalty makes the
  float32 `XᵀWX` Cholesky well-conditioned, so `ridge(..., backend='gpu')` fits a
  GLM *fast and stably* on the GPU where a plain GLM is ill-conditioned and fails in
  float32. Uses mixed precision (float32 solve, float64 working quantities) and an
  float32-appropriate convergence tolerance for the penalized path only; results
  match the CPU ridge fit to float32 tolerance. Plain (unpenalized) GLM keeps the
  strict tolerance and fails loud when float32 is unreliable.
- **New `backend='gpu_fp64'` (double-precision GPU).** Runs the GPU fit in float64.
  Valid only on CUDA (Metal/Apple Silicon has no float64 and raises a clear error).
  Numerically equivalent to the CPU reference (~1e-15). A correctness option for
  data-center GPUs, not a speed claim on consumer cards. The backend-string
  convention is now `<device>[_<precision>]`: `cpu` (float64), `gpu` (float32),
  `gpu_fp64` (CUDA float64).
- **Clearer failure for an unreliable GPU GLM.** When a plain GLM on the GPU cannot
  converge in float32 (log-link families at large n), the error now lists the
  options — use `backend='cpu'`, use a ridge-penalized fit, or pass `force=True` to
  return the float32 fit anyway — instead of just pointing at the CPU. `force=True`
  is now honored by GPU GLM (returns the non-converged fit, flagged).
