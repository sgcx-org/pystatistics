# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **`montecarlo` bootstrap intervals now match R `boot::boot.ci` numerically.**
  `boot_ci` previously used numpy's default (type-7) quantile interpolation and
  Efron's jackknife acceleration for BCa, so its intervals differed from R's
  `boot.ci` by a small but systematic amount even on identical replicates. Two
  changes bring it to parity:
    - the basic, percentile, studentized and BCa endpoints now use R's
      `norm.inter` rule (interpolation of the order statistics on the normal
      scale) instead of type-7 — basic/percentile now match `boot.ci` to machine
      precision on shared replicates;
    - the BCa acceleration now uses the regression estimate of the empirical
      influence values (R's `empinf` default), computed for the ordinary
      bootstrap by regenerating the resample frequencies from the seed (no extra
      memory; a jackknife fallback is used, with a self-check, for
      balanced/parametric/stratified resampling). BCa now agrees with `boot.ci`
      to ~1e-4 on shared replicates (was ~1e-2).
  Files: `montecarlo/_ci.py`, `montecarlo/_influence.py`. Interval *values* shift
  slightly toward R's; the coverage of every interval type is unchanged (the
  methods were already well-calibrated).
- **`montecarlo` `permutation_test` two-sided p-value corrected for non-centred
  statistics.** The two-sided p-value counted `|perm| >= |obs|`, which is only
  correct when the statistic's permutation null is centred at zero (e.g. a
  difference in means). For a statistic that is not — a ratio of means, say — it
  returned a wrong value (~0.89 where the correct two-sided was ~0.40). It now
  uses the standard tail-doubling rule `min(1, 2*min(p_greater, p_less))`, which
  is correct for any statistic: it equals the previous result for a difference
  statistic (both tails equal) and is now correct for a ratio. One-sided
  p-values are unchanged. File: `montecarlo/_common.py` (shared by the CPU and
  GPU backends).
