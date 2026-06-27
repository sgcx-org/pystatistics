# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **New `.conf_int` accessor + `conf_level=` parameter across the coefficient
  models** (consistency feature; the constitution lists `.conf_int` as a uniform
  result accessor, but it was unimplemented on every coefficient model in 4.0).
  Purely additive — no existing accessor, result, or signature changes.
  - `.conf_int` returns a `(p, 2)` array of **Wald** intervals `estimate ± q·se`
    on the coefficient scale (`(J-1, p, 2)` for multinomial). The quantile `q`
    matches the model's reference distribution: Student-t at `df_residual` for
    OLS and at each coefficient's Satterthwaite df for LMM; normal (z) for GLM
    fixed-dispersion families, Cox, multinomial, ordinal, and GLMM; Student-t for
    estimated-dispersion GLM families (Gaussian/Gamma). `exp(.conf_int)` gives
    hazard-/odds-/rate-ratio intervals where applicable.
  - Added `conf_level: float = 0.95` to `regression.fit`, `survival.coxph`,
    `multinomial.multinom`, `ordinal.polr`, and `mixed.lmm`/`mixed.glmm`, with
    `(0, 1)` validation. Each Solution exposes `.conf_level`.
  - OLS `.conf_int` matches R's `confint.lm` to ~1e-15; Cox/GLM match R's Wald
    intervals (`summary.coxph` conf.int / `coef ± z·se`) to ~1e-15.
  - `CoxSolution.summary()` now derives its hazard-ratio interval from
    `conf_level` instead of a hardcoded 1.96.
  - Penalized (ridge) regression reports NaN standard errors, so its `.conf_int`
    is NaN — a biased estimator has no valid Wald interval.
