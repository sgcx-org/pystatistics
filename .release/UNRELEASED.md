# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

### 5.0 — pre-launch consistency sweep (BREAKING)

The single pre-launch breaking cut. Removals and renames land as their own
commits; this section accumulates them.

**Scheduled removals cleared (were deprecated with a `DeprecationWarning`):**

- Removed `mvnmle.mlest(backend='cpu-reference')`, the deprecated alias for
  `solver='reference'`. `backend=` encodes device+precision only; the R-exact
  numpy inverse-Cholesky path is a numerical-routine (`solver`) choice. Use
  `mlest(X, solver='reference')`. Dropped `'cpu-reference'` from `BackendChoice`,
  the alias branch in `mvnmle/solvers.mlest`, and the dead `_get_backend`
  branch; the PyTorch-missing fallback warning now points at `solver='reference'`.
- Removed the `gam` `SmoothInfo.lambda_` / `.s_scale` scalar property shims. A
  tensor `te()`/`ti()` smooth carries one smoothing parameter per margin, so the
  accessors are the tuples `SmoothInfo.lambdas` / `.s_scales` (deprecated in
  4.8.0).

**`hypothesis` — descriptive parameter names (no single-letter / no collisions):**

- `chisq_test`: `p=` → `expected_probs=`, `rescale_p=` → `rescale_probs=`,
  `B=` → `n_resamples=`.
- `prop_test`: `p=` → `null_value=` (the null proportion — a generic null),
  `n=` → `n_trials=`.
- `fisher_test`: `B=` → `n_resamples=`.
- `var_test`: `ratio=` → `null_value=` (the null variance ratio; default 1.0).
- `p_adjust`: `p=` → `p_values=`, `n=` → `n_comparisons=`.
  Resolves the previous collision where the bare name `p` meant three different
  things (expected proportions, a null proportion, and a p-value vector) and the
  bare `n`/`B` were single-letter public parameters.

**`mice` — uniform accessors and confidence-level convention:**

- `pool(dfcom=)` → `pool(df_complete=)`.
- `pool(alpha=)` → `pool(conf_level=)`. The value convention is now the
  library-wide confidence level (default `0.95`), not a significance level
  (was `0.05`); pooled interval bounds are unchanged for the equivalent request.
- `MICESolution.completed(i)` → `completed(index)`.
- `PooledSolution.se` → `.standard_errors`; `.ci_low`/`.ci_high` →
  `.ci_lower`/`.ci_upper`, and a new `.conf_int` accessor returns a `(k, 2)`
  array of `[lower, upper]` per pooled estimate.
- The metadata `.info` dict keys `"m"`/`"maxit"` are now `"n_imputations"`/
  `"max_iter"`, matching the public parameter names.

**`montecarlo` — result accessors and CI-type values:**

- `BootstrapSolution.R` / `PermutationSolution.R` → `.n_resamples` (the
  constructor parameter was already `n_resamples`; only the accessor lagged).
- `BootstrapSolution.se` → `.standard_errors`, `.ci` → `.conf_int`,
  `.sim` → `.method`, `.ci_conf_level` → `.conf_level`.
- `boot_ci(ci_type=...)` values `'perc'` → `'percentile'`, `'stud'` →
  `'studentized'` (the returned CI dict is keyed by these too); `'bca'` unchanged.
- The metadata `.info` dict keys `'sim'`/`'stype'` are now `'method'`/
  `'statistic_type'`.
- `boot_ci` now raises `ValidationError` on a multi-level `conf_level` sequence
  instead of silently using only the first level (a fail-loud fix).
- `BootstrapSolution.t` / `.t0` (R `boot`-object parity) are unchanged.

**`regression` / `anova` — family, link, and analysis-table naming:**

- The `regression.anova()` analysis-of-deviance function is renamed
  `regression.deviance_table()` (it collided with the `anova` module's
  `anova()`); its result class `AnovaTable` → `DevianceTable` (now with a
  Jupyter `_repr_html_`). Its `test=` values are lowercase `'chisq'`/`'lrt'`/`'f'`
  (the `Pr(>Chisq)` display headers are unchanged). `drop1()` is unchanged.
- GLM family string values are hyphenated: `'negative.binomial'` →
  `'negative-binomial'`, `'inverse.gaussian'` → `'inverse-gaussian'` (both the
  accepted `family=` value and the emitted `.family_name`). The `'nb'` shorthand
  still works.
- The Gamma family class is `Gamma` (was `GammaFamily`) and its `.family_name`
  is lowercase `'gamma'`, consistent with the other families.
- The inverse-squared link value/name is `'inverse-squared'` (the cryptic
  `'1/mu^2'` spelling is removed).
- `ridge(lam=)` → `ridge(l2=)` (the library-wide L2-penalty name).
- `anova_rm(correction=)` values `'gg'`/`'hf'` → `'greenhouse-geisser'`/
  `'huynh-feldt'`.
- Input-validation errors that were bare `TypeError` / `ValueError` /
  `NotImplementedError` on public paths now raise the corresponding
  `core.exceptions` types (`ValidationError` / `DimensionError` /
  `NotImplementedFeatureError`).

**`timeseries` — descriptive names and the forecast confidence convention:**

- `decompose(type=)` → `decompose(kind=)` (and the `DecompositionSolution.type`
  / `ACFSolution.type` result attributes → `.kind`); `type` shadowed a builtin.
- `forecast_ets(h=)` / `forecast_arima(h=)` → `n_ahead=` (the forecast horizon).
- `ndiffs(alpha=)` → `ndiffs(significance_level=)` (the ETS `alpha=` smoothing
  coefficient is a different concept and keeps its name).
- `auto_arima(allowdrift=)` → `allow_drift`; `arima`/`forecast_arima`
  `newxreg=` → `new_xreg` (`xreg` itself is unchanged).
- `arima`/`auto_arima` `method=` values are lowercase: `'css-ml'`/`'ml'`/`'css'`/
  `'whittle'` (were `'CSS-ML'`/`'ML'`/`'CSS'`/`'Whittle'`).
- `forecast_ets`/`forecast_arima` `levels=[95]` (whole percents) →
  `conf_level=0.95` (fractions), accepting a float or a sequence; a value `>= 1`
  now fails loud. Interval bounds are unchanged for the equivalent request.

**`mvnmle` — error taxonomy:**

- `little_mcar_test()` now raises `ConvergenceError` (carrying `iterations`/
  `reason`) on non-convergence and `NumericalError` on a failed ML computation,
  instead of a bare `RuntimeError` (which is now reserved for GPU-unavailable /
  environment failures). The GPU-unavailable error uses the canonical shared
  message (with the "use `backend='cpu'`" remedy).

**`gam` — uniform result accessors:**

- `GAMSolution.se` → `.standard_errors` (consistent with the rest of the
  library). Added `.coef` (labeled dict over the parametric coefficients),
  `.conf_int` (parametric coefficients), and `.backend_name`. Smooth-term
  significance stays on `.summary()` (approximate F/Chi-sq, not per-coefficient),
  so no `.z_values`/`.t_values`/`.p_values` arrays are added.

**`mixed` — descriptive parameter name and uniform accessors:**

- `grm_lmm(W=)` → `grm_lmm(random_factor=)` (the low-rank factor defining
  `K = WW'/M`); `W` was a single-letter public parameter.
- `LMMSolution` / `GLMMSolution` gain `.backend_name`, `.timing`, `.warnings`;
  `GRMSolution` gains `.timing` / `.warnings`. `LMMSolution` / `GLMMSolution`
  gain a `.coef` accessor aliasing `.fixef`.

**`ordinal` / `multinomial` — predict selector, link value, uniform accessors:**

- `OrdinalSolution.predict(type=)` and `MultinomialSolution.predict(type=)` →
  `predict(kind=)` (`type` shadowed a builtin); values `'class'`/`'probs'`
  unchanged.
- Ordinal logit link value `'logistic'` → `'logit'`, matching the GLM/GAM
  spelling of the same link.
- `OrdinalSolution` gains `.category_names`; `MultinomialSolution` gains
  `.backend_name` / `.info` / `.timing` and a `.coefficients` alias for
  `.coefficient_matrix`.
