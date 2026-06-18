# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **`ordinal.polr` / MICE `polr` method: ~4x faster fits and reliable
  convergence on correlated data.** The variance-covariance matrix is now
  built by forward-differencing the analytic gradient (`d + 1` gradient
  evaluations) instead of an element-wise numerical Hessian (`d * (d + 1)`
  evaluations); a single proportional-odds fit at n=10000 drops from ~158 ms
  to ~40 ms. End to end, imputing an ordered factor with `mice` (n=10000,
  m=10, maxit=10) drops from ~21 s to ~7 s — from ~1.7x slower than R's
  `mice`/`MASS::polr` to ~1.6x faster, on identical data, with imputed
  category proportions matching R within Monte-Carlo tolerance.
- **`ordinal.polr` convergence is now decided by the maximum-likelihood
  score condition, not the optimizer's status flag.** L-BFGS-B previously
  reported spurious non-convergence on realistic correlated/high-signal data
  (it had in fact reached the optimum), which made the MICE `polr` method fall
  back to a marginal draw and degrade imputations. The fit is now polished to
  the MLE with safeguarded Newton steps and accepted via a parameterization-
  invariant decrement test; genuinely separated data (no finite estimate) is
  still rejected with a clear `ConvergenceError`. Near-separation cases that
  previously failed now converge.
- **`ordinal.polr` threshold standard errors and variance-covariance are now
  reported in natural threshold coordinates, matching `MASS::polr`.** They
  were previously on the internal unconstrained (log-gap) parameterization,
  so threshold SEs beyond the first disagreed with R. Slope coefficient
  standard errors are unchanged. This also corrects the MICE `polr` method's
  posterior draw of the thresholds, which is taken in natural coordinates.
