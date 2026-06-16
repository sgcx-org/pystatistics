# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- `mlest` (MVN MLE) now detects rank-deficient input and fails loudly instead
  of reporting a meaningless fit. When two or more variables are (near-)collinear
  the observed-data covariance is singular and no interior maximum-likelihood
  estimate exists; previously the optimizer could terminate and return
  `converged=True` with a near-singular fitted Σ (a silent false success).
  `mlest` now inspects the fitted covariance via the minimum eigenvalue of its
  correlation matrix (a scale-invariant degeneracy measure, default threshold
  1e-5) and raises `SingularMatrixError` by default. New parameters:
  `force=False` (set `True` to return the degenerate result anyway, with
  `converged=False` and a warning attached) and `collinearity_tol=None`
  (override the detection threshold). The check is centralized in `mlest`, so it
  covers the direct, EM, and monotone algorithms on both CPU and GPU backends.
  Detection only — collinear columns are never silently dropped; column
  selection remains the caller's responsibility. New module
  `pystatistics/mvnmle/_degeneracy.py`.
