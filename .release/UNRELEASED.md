# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **Gamma GLM AIC now matches R's `glm.fit` AIC to round-off.** `GammaFamily.aic`
  (`pystatistics/regression/families.py`) previously evaluated the
  log-likelihood at dispersion `deviance/df_residual` and used the generic
  `-2*loglik + 2*rank`, both of which disagree with R. R's `Gamma()$aic`
  evaluates at the MLE dispersion `deviance/sum(weights)` and counts the
  estimated dispersion as a free parameter (`+2`). For a typical Gamma(log) fit
  this corrected AIC from ~585 to ~545 (R's value); the gap was ~2.5 AIC units
  in the weighted reference cases. The dispersion *reported* for standard errors
  (the moment estimate `deviance/df_residual`) is unchanged — only the AIC's
  internal dispersion differs. Fixed-θ negative-binomial AIC was already correct
  (matches `MASS::negative.binomial(theta)$aic`, no extra penalty for a known θ)
  and is unchanged. AIC parity for Gamma (log and inverse links) and
  negative-binomial is now asserted against R in
  `tests/regression/test_weights_offset_r_validation.py` (previously only
  finiteness was checked).

- **Prior weights (`weights=`) and offset (`offset=`) for `regression.fit`.**
  `fit()` now accepts per-observation prior `weights` and a linear-predictor
  `offset`, matching R's `lm(..., weights=)` and `glm(..., weights=, offset=)`.
  - `weights` are prior/observation weights: weighted least squares for OLS
    (Gaussian), and IRLS prior weights for every GLM family
    (binomial/poisson/gamma/negative-binomial, including the negative-binomial
    θ estimation). Must be non-negative and not all zero. As with R's
    `lm`/`glm`, they are precision weights — residual df is `n − p` (rows), so a
    case-weighted fit and the corresponding row-replicated fit share point
    estimates but not standard errors.
  - `offset` enters the linear predictor as `η = Xβ + offset` and is not
    estimated — e.g. `log(exposure)` for a Poisson rate model. The null
    deviance, fitted values, residuals, deviance and standard errors all account
    for it.
  - Validated against R (`glm.fit`/`lm` with `weights=`/`offset=`) to
    CPU-vs-R round-off on coefficients, standard errors, fitted values,
    deviance, null deviance and dispersion across Gaussian, binomial, Poisson,
    Gamma (log and inverse links) and fixed-θ negative binomial.
  - GPU (float32) OLS and GLM backends honor both inputs, reproducing the CPU
    fit to float32 tolerance.
  - Standard errors for a weighted OLS fit now use the weighted Gram `XᵀWX`
    (was the unweighted `XᵀX` on the GPU path). R² for a weighted fit uses the
    `mss/(mss+rss)` decomposition (matches R's `summary.lm` with no offset).
  - Prior versions raised `TypeError` when `weights=`/`offset=` were passed —
    this closes that documented gap.
  - Not yet supported together with a ridge penalty (`l2 > 0`): `weights=` /
    `offset=` raise `NotImplementedError` there (no R reference to validate a
    weighted/offset ridge against; `MASS::lm.ridge` takes neither).
