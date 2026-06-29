# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **Auto-estimated negative-binomial AIC/BIC now count θ as a parameter, matching
  `MASS::glm.nb`.** When `fit(family='negative.binomial')` estimates the
  dispersion θ (θ not supplied), θ is a free parameter; R's `glm.nb` penalizes it
  in the information criteria. Previously the auto-θ fit reported the *fixed*-θ
  family AIC (`-2·logL + 2·rank`), so its AIC was 2 too low and its BIC omitted
  the `log(n)` term for θ. The AIC now adds 2 and the BIC adds `log(n)` for the
  estimated θ (via a new `GLMParams.ic_param_count`), reproducing `glm.nb`'s AIC
  and BIC to round-off. `rank`, `df_residual`, standard errors and Wald
  statistics are unchanged (θ is not counted there, matching R). Fixed-θ
  negative-binomial fits — `fit(family=NegativeBinomial(theta=...))` — are
  unaffected.
- **The estimated θ is now exposed** on an auto-θ negative-binomial fit via
  `result.info['theta']` (with `result.info['theta_estimated'] = True`), matching
  what `glm.nb` reports. Validated against `MASS::glm.nb` (θ, coefficients,
  standard errors, deviance, AIC, BIC) with and without prior weights / an offset
  in `tests/regression/test_nb_autotheta_r_validation.py`.
