# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **Fixed `factor_analysis` default (varimax) rotation failing to converge on
  clean multi-factor data.** `varimax` (in `multivariate/_rotation.py`) used an
  *absolute* convergence test on the rotation criterion, whose scale depends on
  the loadings; on well-fitting multi-factor models the criterion's per-iteration
  gain plateaus and never reached the absolute threshold within 1000 iterations,
  so `factor_analysis(X, n_factors>=2)` (rotation defaults to `'varimax'`) raised
  `ConvergenceError` on textbook simple-structure data. `varimax` now uses the
  *relative* convergence test `d < d_prev * (1 + tol)` with `tol=1e-5`, matching
  R's `stats::varimax`. The genuine-non-convergence `ConvergenceError` is
  retained. Rotated loadings now match `stats::varimax`, and multi-factor
  `factor_analysis` agrees with R `factanal` (uniquenesses to ~1e-4, ML objective
  to ~1e-6, loadings up to rotation/sign to ~1e-3). The `promax` initial varimax
  step inherits the same relative tolerance.

- **Added a `lower=` uniqueness floor to `factor_analysis` (default `0.005`).**
  Previously uniquenesses were only clamped just above 0, allowing degenerate
  Heywood solutions (a variable's uniqueness collapsing to ~0) and a different
  optimum than R. The optimisation is now box-constrained so every uniqueness
  stays `>= lower`, matching R `factanal`'s default `lower=0.005`. On the iris
  1-factor model this floors Petal.Length's uniqueness at 0.005 and yields the
  same ML objective as R (~0.585 vs the previous unconstrained ~0.566). `lower`
  must satisfy `0 < lower < 1` (raises `ValidationError` otherwise) and is
  recorded on the result's `info` dict.
