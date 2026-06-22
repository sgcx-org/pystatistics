# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- Sped up the GPU MICE `polr` (ordinal) fit, the dominant new cost on Apple
  Silicon (MPS) since the 3.16.1 line search. `_backtracking_step` in
  `mice/backends/_gpu_polr.py` now selects the step by **safeguarded quadratic
  interpolation** (Nocedal & Wright) instead of plain halving: under
  (quasi-)separation the full Newton step overshoots by orders of magnitude, so
  halving needed ~12 objective evaluations per step to reach an acceptable step;
  interpolation jumps toward the minimiser in ~3-4. The acceptance test (per-chain
  decrease) is unchanged, so the convergence guarantee and the issue #8
  anti-collapse behaviour are identical. Also reuses the objective value already
  computed in `_grad_and_hessian` (`f0`) instead of recomputing `NLL(params)`,
  saving one more evaluation per Newton step. Measured on a synthetic mixed GPU
  MICE run (MPS, n=10000, m=20, maxit=10): `_nll_per_chain` evaluations
  32148 -> 12279 (2.6x fewer) and wall-clock 69s -> 55s; the remaining cost is
  the autograd gradient/Hessian, which is inherent. CUDA is unaffected
  (cheaper dispatch).
- Added line-search contract tests (`tests/mice/test_gpu_polr_separation.py`):
  the interpolating step must decrease the per-chain objective (the invariant the
  damped Newton's convergence rests on), and a full step that already decreases
  is accepted without backtracking.
