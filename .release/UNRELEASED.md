# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

### 5.0 — pre-launch consistency sweep (BREAKING)

The single pre-launch breaking cut. Removals and renames land as their own
commits; this section accumulates them.

**Scheduled removals cleared (were deprecated with a `DeprecationWarning`):**

- Removed `mvnmle.mlest(backend='cpu-reference')`, the deprecated alias for
  `solver='reference'`. `backend=` encodes device+precision only; the R-exact
  numpy inverse-Cholesky path is a numerical-routine (`solver`) choice. Use
  `mlest(X, solver='reference')`. Dropped `'cpu-reference'` from `BackendChoice`,
  the alias branch in `mvnmle/solvers.mlest`, and the dead `_get_backend`
  branch; the PyTorch-missing fallback warning now points at `solver='reference'`.
- Removed the `gam` `SmoothInfo.lambda_` / `.s_scale` scalar property shims. A
  tensor `te()`/`ti()` smooth carries one smoothing parameter per margin, so the
  accessors are the tuples `SmoothInfo.lambdas` / `.s_scales` (deprecated in
  4.8.0).
