# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **gam: expose the negative-binomial dispersion `theta` on the fitted result.**
  A `gam(..., family='nb')` fit estimates the NB dispersion theta internally
  (matching `mgcv::getTheta` to ~1e-5) and it drives every reported quantity,
  but the fitted result surfaced it nowhere — `family_name` was the bare string
  `'negative.binomial'` and the estimated theta was discarded. Added a
  `theta` accessor: `GAMSolution.theta` and `GAMParams.theta` now return the
  estimated theta for an auto-fit `family='nb'`, the supplied value for a fixed
  `NegativeBinomial(theta=...)` family, and `None` for every other family. No
  change to the fit itself (completeness fix; `pystatistics/gam/_common.py`,
  `_gam.py`, `solution.py`).

- **gam: support factor `by=` smooths and reject a factor coded as a `by=`
  column instead of silently misfitting it.** `s(x, by=g)` previously treated
  the `by` column as a continuous varying coefficient in every case, so passing
  a categorical grouping variable (e.g. `factor(g)` coded `0,1,2`) silently fit
  one meaningless smooth scaled by the level codes rather than a smooth per
  group. Two changes:
  - New `by_type='factor'` on `s()` fits a separate smooth per level of an
    integer-coded `by`, adding the per-level group means automatically —
    mgcv's `s(x, by=factor(g))`. Total EDF and fitted values match
    `mgcv::gam` to floating-point arithmetic (~1e-8). Factor-`by` defaults to a
    thin-plate basis, matching `mgcv::s()`, and each level gets its own
    smoothing parameter.
  - `s(x, by=...)` with no `by_type` now raises a clear error when the `by`
    column looks categorical (integer-valued, low-cardinality, coded as a
    contiguous run like `0,1,2` or `1,2,3`), telling the user to choose
    `by_type='factor'` or `by_type='continuous'`. A binary `0/1` `by` and any
    genuinely continuous `by` are unaffected, and existing continuous-`by`
    fits are numerically unchanged. Use `by_type='continuous'` to keep the old
    behaviour on a low-cardinality integer column.

  New module `pystatistics/gam/_factor_by.py`; `_smooth.py`, `_basis.py`,
  `_gam.py`.
