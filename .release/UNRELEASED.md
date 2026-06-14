# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- Categorical imputation for `mice`. Columns can now be declared
  `binary` / `categorical` (unordered) / `ordered` via `column_kinds`, and are
  imputed with the matching R-mice method: `logreg` (Bayesian logistic),
  `polyreg` (multinomial logit, reusing the `multinomial` module), and `polr`
  (proportional odds, reusing the `ordinal` module). Each draws its parameters
  from the posterior normal approximation, predicts class probabilities for the
  missing rows, and samples a category.
- `method='auto'` (new default) picks the R-default method per column kind
  (numeric->pmm, binary->logreg, categorical->polyreg, ordered->polr). An
  explicit method name still applies to all incomplete columns. Categorical
  columns must be encoded as integer category codes.
- Categorical predictors are treatment-dummy-encoded in the sweep
  (`pystatistics/mice/_encode.py`); the numeric-only path keeps a fast slice so
  its performance is unchanged. The chain maps categorical targets to/from
  consecutive `0..K-1` class indices.
- Validated distributionally against R `mice` 3.19.0: imputed category marginal
  proportions for logreg/polyreg/polr match R within ~0.015 on a shared
  mixed-type dataset (`tests/mice/references/`).
- Robustness: if a categorical model fit fails to converge on an awkward
  intermediate sweep state, the method falls back to a marginal draw for that
  step with a visible `UserWarning` (matching how R wraps MASS::polr), retrying
  the full conditional model on the next iteration. Observed ~0.1% of fits.
- The GPU backend now explicitly refuses data with any categorical column
  (categorical imputation is CPU-only); previously only the target method was
  checked.
- Added a public `vcov` property to the multinomial solution
  (`pystatistics/multinomial/solution.py`) for parity with the ordinal module,
  so the imputation method reads only public surfaces.
