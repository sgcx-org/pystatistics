# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

*(empty — no unreleased changes yet)*
- **gam: tensor-product and isotropic multivariate smooths — `te()`, `ti()`,
  and `s(x, z, ...)`** (VA-1). Previously the smooth constructor took a single
  variable; multivariate smooths were absent.
  - `te(x, z, ...)` fits a tensor-product smooth: the marginal bases are
    combined by a row-wise Kronecker product with one penalty (and one
    smoothing parameter) per margin, and a single sum-to-zero identifiability
    constraint. Each margin may use any implemented basis (`cr`/`tp`/`cc`/`ps`)
    at its own dimension, e.g. `te('x', 'z', bs=['cc', 'cr'], k=[6, 5])`.
  - `ti(x, z, ...)` fits the tensor-product interaction with the marginal main
    effects removed, for functional-ANOVA models
    `te('x') + te('z') + ti('x', 'z')`. A single-variable `te('x')`/`ti('x')`
    is a centred 1-D smooth, matching mgcv.
  - `s(x, z, ...)` (two or more variables) fits an isotropic multivariate
    thin-plate spline — one penalty shared across the covariates, for variables
    on a common scale.
  - The tensor / multivariate basis matrices and penalties match
    `mgcv::smoothCon` to ~1e-9; full fits match `mgcv::gam` on total EDF,
    scale, fitted values and the per-margin smoothing parameters, under both
    GCV and REML, for Gaussian and GLM families (validated on
    `te`/`ti`/isotropic `s`, mixed cyclic×cubic margins, and a Poisson tensor
    fit). Smoothing-parameter selection uses the exact analytic REML/GCV
    gradient extended to the several overlapping penalties a tensor smooth
    carries (the penalty log-determinant and its gradient are taken jointly
    over each smooth's margins, so ordinary smooths are numerically unchanged).
  - `solution.smooth_terms[i].lambdas` / `.s_scales` are now tuples (one entry
    per margin for a tensor smooth; length 1 for an ordinary smooth), and
    `sp=` takes one value per smoothing parameter (per margin). Tensor /
    multivariate smooths are CPU-only, like the rest of the gam module.
  - New modules `pystatistics/gam/_tensor_smooth.py` (the `te`/`ti` spec),
    `_basis_te.py` (tensor basis assembly), `_basis_md.py` (multivariate
    thin-plate basis) and `_penalty_group.py` (the joint penalty determinant).
