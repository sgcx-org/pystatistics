# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **`DiscreteTimeSolution` gains `.conf_int` and `.conf_level`, and
  `discrete_time` gains a `conf_level=` parameter** (default 0.95), completing
  the uniform `.conf_int` accessor across the coefficient models. Wald intervals
  for the covariate coefficients using the normal quantile (the person-period
  logistic fit is asymptotic-normal); `exp(.conf_int)` gives discrete-time
  hazard-ratio intervals. Additive — nothing else changes.
