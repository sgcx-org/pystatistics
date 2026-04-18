# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py <version>`
> That script uses this file to build the CHANGELOG entry, bumps versions
> everywhere, and resets this file for the next cycle.

## Changes

### README — catch-up for 1.8.0 and 1.9.0

- **README's "What's New" section updated** to reflect the 1.8.0
  and 1.9.0 releases (the full GPU backend sweep, Whittle ARIMA,
  GPU-resident PCAResult, `arima_batch`). Both prior releases
  shipped with README still describing 1.7.0 as current. Doc-only
  change — no code is touched.

- **`.release/CHECKLIST.md` already states** that the feature-
  summary prose in README is manual work that `release.py` does
  not auto-generate; the issue was the step being skipped, not the
  checklist being missing.
