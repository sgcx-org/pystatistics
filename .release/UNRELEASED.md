# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **Categorical predictors and interaction terms in regression.** `fit()`
  (OLS and all GLM families) and `survival.coxph()` now accept a structured
  term spec via a new `terms=` argument, alongside the existing numeric-matrix
  path. A term is a bare column name (numeric main effect), `C(name, ref=...)`
  (categorical, treatment/dummy coded with a selectable reference level), or a
  tuple of those (interaction — numeric×numeric, numeric×categorical, and
  categorical×categorical all supported). Example:
  `Design.from_datasource(ds, y="response", terms=["age", C("sex", ref="F"), (C("treatment", ref="A"), C("sex", ref="F"))])`.
  Expanded columns are labeled `sex[M]`, `treatment[B]:sex[M]`, and the
  `coef`/`standard_errors`/statistic/`p_values` outputs stay index-aligned to
  those labels. Cox keeps its no-intercept rule (the term builder emits an
  intercept-free matrix for it). PyStatistics intentionally does not use
  R-style formula strings; the spec is plain Python data, so it is
  GUI-constructible and round-trips into a copy-runnable snippet. New public
  symbol: `pystatistics.regression.C`.
- **`DataSource.from_dataframe` preserves non-numeric columns.** Previously
  every column was force-cast to float64, which crashed on string/categorical
  columns. Numeric columns are still stored as float64; non-numeric columns are
  retained as-is so they can be encoded as categorical predictors via `C(...)`.
- **ETS convergence on near-perfectly-fit series.** `_ets_fit._neg_loglik`
  and the final log-likelihood now floor the Gaussian variance relative to the
  data scale (`1e-12 * var(y)`) instead of a fixed `1e-30`. A noiseless/near-
  noiseless series previously drove residual variance toward zero, making the
  additive/multiplicative likelihood unbounded and leaving L-BFGS-B with an
  exploding gradient (ABNORMAL line-search termination, `converged=False`).
  Fits on genuine (noisy) series are numerically unchanged. Test
  `TestHoltLinear::test_linear_trend_recovery` now passes.
- **Shared categorical-encoding engine.** The factor-encoding engine (dummy /
  deviation encoding, interaction columns, design-matrix construction) lives in
  `pystatistics.core.encoding` so regression and ANOVA share one R-validated
  implementation. ANOVA's import path is unchanged (re-export shim).
