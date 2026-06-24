# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- Fixed a false `converged=False` on `mlest(..., algorithm='direct')` for good
  fits on large datasets. The optimizer's objective is the summed
  `-2*log-likelihood`, whose magnitude grows with the data (~1e5+ for
  survey-scale data); at that scale scipy's BFGS could not drive the *absolute*
  gradient below the default `gtol` and terminated with "Desired error not
  necessarily achieved due to precision loss", reporting non-convergence even
  though the parameters were at the optimum. The optimizer now runs on the
  objective scaled to a per-observation mean (`backends/_optimize.py`), so the
  gradient magnitude is O(1) and the convergence test is meaningful and
  independent of dataset size. Estimates and log-likelihood are unchanged
  (scaling a positive constant does not move the optimum). Affects the CPU
  backend and both GPU backends (FP64/FP32), which shared the same
  unscaled-objective pattern.
- As a result, fits that previously surfaced spurious "Optimization did not
  converge" warnings on clean, well-conditioned data (e.g. the bundled
  `missvals` dataset via `algorithm='direct'`) now converge and match R's
  `mvnmle::mlest` reference. `little_mcar_test` no longer takes the
  non-convergence error path on such data.
- `MVNSolution.gradient_norm` for the direct optimizer is now the
  per-observation (scaled) gradient infinity-norm — the quantity convergence is
  judged against — rather than the raw summed-objective gradient, which was
  dominated by floating-point noise at large objective magnitudes and could read
  as "large" even at a genuine optimum. The reported `objective_value` and
  `loglik` are unchanged.
