# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- New `mice` module: Multiple Imputation by Chained Equations for multivariate
  missing data. `pystatistics.mice.mice(data, m=5, maxit=5, method='pmm',
  seed=...)` iteratively imputes each incomplete numeric column from the others
  and returns `m` completed datasets (`MICESolution`). Numeric methods `pmm`
  (predictive mean matching, the R default) and `norm` (Bayesian linear
  regression) are included, with R-faithful defaults (`m=5`, `maxit=5`, 5 PMM
  donors, left-to-right visit sequence). Imputation is fully deterministic given
  `seed` (required), with independent per-chain RNG streams.
- New `pystatistics.mice.pool`: Rubin's-rules pooling of an analysis across the
  `m` completed datasets, with Barnard-Rubin degrees of freedom and the standard
  diagnostics (within/between/total variance, relative increase in variance,
  lambda, fraction of missing information) and confidence intervals.
- Validated distributionally against R's `mice` 3.19.0: imputed-value
  distributions (mean, spread, quantiles) and Rubin's-rules pooled regression
  output match R on a shared incomplete dataset. Reference fixtures and the
  pinned generation script live in `tests/mice/references/`.
- CPU implementation only in this release. The module is structured (method
  registry, single backend entrypoint, per-column type metadata) so GPU
  acceleration and categorical methods can be added without restructuring.
