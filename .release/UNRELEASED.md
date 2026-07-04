# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **timeseries: benign `RuntimeWarning: overflow encountered in exp` from the
  ETS optimiser silenced (no numeric change).** `_inv_logit` in
  `timeseries/_ets_fit.py` computed `np.exp(-z)` on unclipped logit-scale
  parameters, so L-BFGS-B line-search probes at large negative `z` overflowed
  to `inf` — mathematically the sigmoid correctly saturates to the lower
  bound, but the warning fired on every affected fit. The logit argument is
  now clipped to [-500, 500] before exponentiation; beyond that range the
  sigmoid is saturated far below one ulp, so all fitted results are
  bit-identical (verified: identical alpha/log-likelihood/fitted-value sums
  on all 10 `ets_r_reference.json` selection cases pre/post change, and the
  full timeseries suite fits with `RuntimeWarning` promoted to error).
