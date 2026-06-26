# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **GPU GLM now fails loudly instead of returning non-converged coefficients.**
  In 3.19.0, a GLM fit with `backend='gpu'` for a log-link family (Poisson/Gamma)
  at large sample size could fail to converge in float32 — most often on Apple
  Silicon (MPS), which has no float64 — and return the non-converged (wrong)
  coefficients with only a convergence warning. The float32 conditioning of XᵀWX is
  insufficient in that regime. The GPU IRLS backend now raises `NumericalError`
  when it does not converge or the inner solve breaks down, telling the caller to
  re-run with `backend='cpu'` for a correct double-precision fit. It never silently
  returns wrong numbers and never silently switches backends. Small/medium GLM fits
  (and large fits on CUDA, which does converge) are unaffected.
