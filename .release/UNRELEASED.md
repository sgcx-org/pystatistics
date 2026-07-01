# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- Fixed `lmm()` failing to converge (and, in a related case, converging to a
  silently wrong optimum) in the extreme variance-ratio regime — when the
  intraclass correlation approaches 1 (residual variance orders of magnitude
  below the between-group variance). The optimal relative random-effects scale θ
  is then very large (O(1e2)–O(1e3)) and the profiled REML/ML deviance is flat
  and ill-scaled, where the gradient-based L-BFGS-B optimizer either terminated
  its line search abnormally (returning `converged=False`) or stopped and
  reported success at a non-stationary point (because its absolute-step
  finite-difference gradient is a negligible *relative* step at large θ). On a
  near-perfect-ICC random-intercept fit (G=15, residual sd 0.03) this returned a
  between-group variance of 107.9 vs lme4's 130.3 (~17% low). The θ optimizer
  (new module `pystatistics/mixed/_optimizer.py`) now runs a bounded
  derivative-free Nelder-Mead fallback — engaged only when L-BFGS-B did not
  converge or a cheap scale-aware stationarity probe flags a premature stop, and
  adopted only when it strictly lowers the deviance — so it converges to the true
  global optimum across the extreme-ratio tail (now matches lme4 to ~1e-6
  relative on the variance component and ~1e-13 on the log-likelihood) while
  leaving all well-converged fits byte-for-behaviour identical.
