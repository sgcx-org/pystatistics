# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- Survival Solution classes now expose a `.warnings` property. `KMSolution`,
  `LogRankSolution`, `CoxSolution`, and `DiscreteTimeSolution` previously had no
  way to surface non-fatal warnings, so warnings computed by the solvers (e.g.
  the Cox Newton-Raphson non-convergence note) were unreachable. The property
  delegates to the underlying result, matching every other domain.
- `survdiff` (log-rank test) now emits non-fatal warnings when the chi-square
  approximation is questionable: when any group's expected event count is below
  5, or when a group has zero observed events. These surface via
  `LogRankSolution.warnings`.
