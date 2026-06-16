# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **MVN MLE: fix pattern-code integer overflow for > 62 variables.** The
  missingness-pattern grouping in `_objectives/base.py` (`_apply_mysort`) coded
  each pattern as `presence_absence @ 2**arange(n_vars)` in int64. For
  `n_vars > 62`, `2**i` overflows int64, so distinct missingness patterns
  collided onto the same code; rows with different masks were then grouped
  together and the group's observed mask sliced NaN (missing) cells into the
  "observed" `pattern.data`, producing NaN objectives/estimates (and NaN on the
  GPU path). Fixed by using arbitrary-precision Python integers for the pattern
  code when `n_vars > 62`; the int64 path (byte-identical, R-compatible) is kept
  for `n_vars <= 62`. Surfaced by survey-scale benchmarks at p=100 (e.g. WVS:
  ~85% singleton patterns) where all GPU configs returned NaN. Regression test
  added in `tests/mvnmle/test_pattern_codes_large_p.py`.
