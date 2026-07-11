# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

### Added

- **ordinal `polr`: `loglog` and `cauchit` links.** `polr(..., link='loglog')`
  and `polr(..., link='cauchit')` now fit, completing MASS::polr's full set of
  five links (previously only `logistic`/`probit`/`cloglog` were implemented and
  the other two failed loud). New `LogLogLink` (Gumbel-max CDF, `g^{-1}(η)=
  exp(-exp(-η))`) and `CauchitLink` (standard Cauchy CDF, `g^{-1}(η)=½+atan(η)/π`)
  in `ordinal/_likelihood.py`, wired through the CPU solver and the (autograd)
  GPU backend. Validated vs `MASS::polr` on frequency-expanded `MASS::housing`
  (n=1681): `loglog` matches coefficients/thresholds/log-likelihood to ~1e-3;
  `cauchit` matches coefficients to ~1e-2 (its heavy-tailed likelihood surface is
  flat near the optimum) and attains a log-likelihood at least as high as R's
  fitted-probability loglik. (Note for reference: MASS's own `logLik()`/`deviance`
  for a `cauchit` fit is internally inconsistent with its `fitted()` probabilities
  — it reports −1752.77 while `sum(log P(y))` at the same fit is −1742.17; the
  estimator-invariant fitted-probability loglik is the correct anchor, and we
  match it.)

- **mixed `glmm`: `GLMMSolution.is_singular` boundary-fit diagnostic.** A GLMM
  fit now surfaces the same `is_singular` flag `LMMSolution` already exposed,
  mirroring `lme4::glmer`'s `isSingular()` — True when a random-effects variance
  has collapsed to (near) zero or an implied correlation has reached ±1. The
  detector (`is_singular_fit`, a function of the fitted θ only) is shared with the
  LMM path; the solver now warns on a boundary fit and records the flag on
  `GLMMParams`. The estimates were already the correct boundary MLE — this closes
  a diagnostic asymmetry, not a numerical gap. Cross-checked against
  `glmer(isSingular=TRUE/FALSE)` on binary data (a collapsed-variance fit
  varRE≈3.6e-15 flags True; a genuine random-intercept fit flags False), with
  fixed effects matching glmer.

- **multivariate `factor_analysis`: `scores=` factor-score estimation.**
  `factor_analysis(..., scores='regression')` (Thomson) and `scores='bartlett'`
  now return an (n × n_factors) score matrix on `FactorSolution.scores`, matching
  R's `factanal(scores=)`; the default `scores='none'` leaves `.scores` as None.
  Validated vs `factanal` across all three rotations (none/varimax/promax) and
  both estimators to ~1e-4 on a 2-factor, 6-variable reference (n=300).

### Fixed

- **multivariate `factor_analysis` promax rotation: column normalisation now
  matches R's `stats::promax`.** The promax step normalised the oblique
  transform by the column norm `1/sqrt(diag(Q'Q))` instead of R's
  `sqrt(diag((Q'Q)⁻¹))` (equal only when `Q'Q` is diagonal), so promax loadings
  were off by a few percent from `factanal(rotation='promax')` — a fidelity gap
  against the R parity the docstring already claimed. Corrected to R's exact
  convention; promax loadings (and hence promax factor scores) now match
  `factanal` to machine precision. Orthogonal (varimax/none) results are
  unchanged. Found while validating the new `scores=` support.

### Added

- **anova: `omega_squared` / `partial_omega_squared` effect sizes.**
  `AnovaSolution` now exposes ω² and partial ω² per term (the less-biased
  companions to the existing η²/partial-η²), computed for `anova_oneway` and
  factorial `anova`. `ω² = (SS_term − df_term·MS_error)/(SS_total + MS_error)`;
  `partial ω² = (SS_term − df_term·MS_error)/(SS_term + (N − df_term)·MS_error)`.
  Validated vs R `effectsize::omega_squared` (one-way match to 1e-9; for a
  one-way design partial ω² coincides with ω², matching effectsize).

- **anova: Games-Howell post-hoc test.** `anova_posthoc(result,
  method='games-howell')` runs the unequal-variance / unequal-n all-pairs
  procedure — each pair uses its own group variances and a Welch-Satterthwaite
  df, with studentized-range inference — the appropriate post-hoc when a Levene
  test rejects homoscedasticity (Tukey pools a single MSE). Validated vs the
  canonical base-R `ptukey`/`qtukey` formula (diff/SE/df/p/CI to ~1e-4).

- **anova: pairwise Cohen's d on post-hoc comparisons.** Every
  `PostHocComparison` now carries a `cohens_d` field — the pairwise standardized
  mean difference with pooled SD, signed to match the reported `diff` — populated
  for the `tukey`, `bonferroni`, `dunnett`, and `games-howell` methods. Validated
  vs R `effectsize::cohens_d` (pooled SD) to ~1e-6.

- **regression GLM: additional links and families (VA-5).** New link functions
  `cloglog`, `cauchit`, `sqrt`, and `1/mu^2` (inverse-squared), and new families
  `inverse.gaussian`, `quasipoisson`, and `quasibinomial`. Non-canonical links are
  selected via the family instance, e.g. `fit(X, y, family=Binomial(link='cloglog'))`
  — the same nesting R uses (`binomial(link="cloglog")`). Quasi families fit
  identically to `poisson`/`binomial` but estimate the dispersion (so SEs are
  inflated by `sqrt(phi_hat)` and inference uses the t-distribution), and their AIC
  is NaN — matching R's `quasipoisson()`/`quasibinomial()`. Validated vs R `glm`
  (coefficients, SEs, dispersion, deviance, AIC) to machine precision. New link and
  family implementations live in `regression/_links_extra.py` and
  `regression/_families_extra.py` (keeping `families.py` under the LoC limit).

- **regression: diagnostics — hat values, Cook's distance, standardized
  residuals (VA-6a).** Both `LinearSolution` (OLS/WLS) and `GLMSolution` now
  expose `.hat_values` (leverage / hat-matrix diagonal), `.cooks_distance`, and
  `.residuals_standardized` (internally studentized; deviance-based for GLM),
  matching R's `hatvalues`, `cooks.distance`, and `rstandard`. The GLM leverage
  uses the final IRLS working weights (`hatvalues.glm`). Validated vs R to ~1e-6;
  leverage sums to the model rank. Leverage is computed once at fit time in the
  CPU/GPU backends (shared helper `regression/backends/_hat.py`).

- **regression: `anova` and `drop1` analysis-of-deviance tables (VA-6a).**
  `anova(model)` gives a sequential (Type I) analysis of deviance; `anova(m1, m2,
  …)` compares nested models; `drop1(model)` reports single-term deletions with
  the resulting deviance, AIC, and test. The test follows R's convention — a
  chi-square (LRT) for fixed-dispersion families, an F test for
  estimated-dispersion families and linear models — and multi-column factor terms
  are grouped and tested with the correct degrees of freedom. Validated vs R
  `anova.glm`/`anova.lm`/`drop1` (deviances, SS, F/χ² statistics, p-values, AIC)
  to machine precision, including a k-level factor as a single df=k−1 term. The
  terms machinery now tracks a term→column `assign` map (`build_terms_design`,
  `Design.assign`/`Design.term_names`) to support this.

- **regression GLM: profile-likelihood confidence intervals (VA-6b).**
  `GLMSolution.profile_conf_int(conf_level=…)` computes the profile-deviance
  intervals R's `confint(glm)` returns — more accurate than the Wald intervals in
  `.conf_int` when the log-likelihood is asymmetric in a coefficient. Each
  endpoint is found by profiling the coefficient (offset trick + refit) and
  root-finding the deviance-drop threshold (χ²₁ for fixed-dispersion families,
  the F-scaled threshold for estimated-dispersion families). Validated vs R
  `confint.glm` for binomial (~1e-5) and poisson (~1e-6).

- **mixed `glmm`: offset, prior weights, and aggregated-binomial response (A4).**
  `glmm(..., offset=…)` adds a per-observation offset to the linear predictor
  (`η = Xβ + Zb + offset`, e.g. `log(exposure)` for a Poisson rate model);
  `glmm(..., weights=…)` supplies IRLS prior weights; and a two-column response
  `y = [successes, failures]` is accepted as an aggregated binomial (R's
  `cbind(k, n-k)`), converted internally to proportions with the trial counts as
  weights. Threaded through the PIRLS core (`_glmm_pirls`, `StructuredContext`),
  which previously assumed a unit-weight, offset-free response and failed loud on
  these. Validated vs `lme4::glmer` on the `cbpp` aggregated-binomial example and
  a Poisson offset model (fixed effects and variance components agree at the
  module's Laplace two-tier tolerance, glmm ≈ glmer(nAGQ=1)).

- **mixed `glmm`: `correlated=` for uncorrelated (diagonal) random effects
  (VA-7).** `glmm(..., correlated=False)` (or a `{group: bool}` dict) fits a
  **diagonal** random-effects covariance — R's `(… || g)` — instead of the
  default full covariance. The relative-covariance Cholesky then carries only its
  `q` diagonal entries (no off-diagonals), so the estimated RE correlation is
  exactly 0. Threaded through a single shared θ→factor parameterisation
  (`theta_to_factor` / `_theta_positions`) now used everywhere θ is packed
  (`build_lambda`, the batched/sparse factor builders, the θ bounds/starts, the
  singularity test, and the variance-component extraction). Validated vs
  `glmer(y ~ x + (1 + x || g))`: correlation pinned to 0 and variances/fixed
  effects match glmer, while the correlated fit of the same data recovers a
  non-zero correlation.

- **montecarlo: BCa now uses regression `empinf` acceleration on balanced and
  stratified bootstrap (A7).** Previously only the ordinary bootstrap used R's
  default regression estimate of the empirical influence for the BCa acceleration
  parameter `a`; the balanced and stratified paths fell back to the delete-1
  jackknife, which can shift a BCa **tail** endpoint by several percent of the CI
  width for a strongly non-linear statistic (measured up to ~7% at small n).
  `regression_influence` now regenerates the balanced/stratified resample
  frequencies from the seed and computes the regression influence — centred
  within strata for a stratified bootstrap, matching R's `empinf` — with a
  seed-reproduction self-check that safely falls back to the jackknife if the RNG
  path ever drifts. The **parametric** bootstrap keeps the jackknife (it has no
  resample frequencies, so the regression estimate does not apply — R's `boot.ci`
  has the same limitation; documented). Validated: the acceleration matches R's
  `empinf(type="reg")` on the same data for ordinary/balanced/stratified to
  Monte-Carlo tolerance. The ordinary path is unchanged (a numerically cleaner
  truncating `rcond` in the influence solve leaves its acceleration identical to
  ~1e-5).

- **gam: cyclic-cubic (`bs='cc'`) and P-spline (`bs='ps'`) smooth bases (A3).**
  `s(x, bs='cc')` fits a cyclic cubic regression spline — the function and its
  first two derivatives match at the endpoints — for periodic/seasonal
  covariates; `s(x, bs='ps')` fits an Eilers-Marx P-spline (cubic B-spline basis
  with a discrete difference penalty). Both are mgcv-exact: the basis matrices,
  penalties, and `S.scale` match `mgcv::smoothCon` to ~1e-9 (ps to machine
  precision), and full REML fits match `mgcv::gam`'s total EDF, scale, and fitted
  values. `cr`/`tp`/`cc`/`ps` now cover the mainstream mgcv bases; the exotic
  `re`/`ds`/`gp`/`fs` bases remain a documented carve-out (see CONVENTIONS
  "Capability scope") and still fail loud.

- **gam: usable negative-binomial family with estimated dispersion (VA-3).**
  `gam(..., family='nb')` previously failed loud ("Cannot compute deviance
  without theta"); it now estimates the dispersion `theta` the way `mgcv::nb()`
  does under REML — minimising the profiled REML criterion over `theta`,
  re-selecting the smoothing parameters at each trial `theta` — and fits.
  `method='GCV'` with an *estimated* theta is refused loudly: profiling the
  UBRE score over theta is structurally degenerate (the NB deviance shrinks
  monotonically as theta → 0 with no counterweight in UBRE, so the profiled
  optimum collapses to theta ≈ 0 — verified on theta=3 data — where mgcv's
  GCV-era `nb()` switches to a different, unimplemented theta estimator);
  a *fixed* `NegativeBinomial(theta=…)` still fits fine under GCV/UBRE. The
  estimated `theta` matches mgcv's `nb()` to ~1% (available on
  `solution.info['nb_theta']`); the smooth fit itself carries gam's existing
  GLM-family tolerance vs mgcv. An explicit `NegativeBinomial(theta=…)` still
  fits directly at the fixed dispersion.

- **gam: continuous `by=` (varying-coefficient) smooths (VA-2).**
  `s(x, by='z')` fits the varying-coefficient term `z · f(x)` (mgcv's
  `s(x, by=z)` for a continuous `by`): the smooth keeps its full basis (no
  centering — the by-multiplication removes the constant confound) and each row
  is scaled by the by-variable. Validated vs `mgcv::gam` (total EDF within ~0.03,
  fitted within a few percent — gam's GLM/REML tolerance). A 0/1 indicator
  by-variable restricts the smooth to that level's observations, the building
  block for a factor-specific smooth. (Native factor-`by` auto-expansion — a
  separately-penalised smooth per factor level — is not yet provided.)

- **gam: analytic smoothing-parameter gradient for the GLM families (A2,
  finding H4).** The outer smoothing-parameter search for Poisson/binomial
  (UBRE), free-dispersion GCV (Gamma, gaussian non-identity, …) and
  fixed-dispersion Laplace REML is now driven by the exact Wood (2011)
  implicit-derivative criterion gradient (new `gam/_gradient_glm.py`; the
  Gaussian-identity closed form was already analytic at 4.6.1) — one inner
  P-IRLS fit per outer step instead of the previous finite-difference `2m+1`
  fits for `m` smooths. The implicit `d beta/d rho` solve uses full NEWTON
  weights: a Fisher-weight shortcut is exact only for canonical links and
  panel-verified up to several percent wrong on probit/Gamma-log — enough to
  silently shift the selected smoothness. Selection lands on the same optimum
  (selected lambdas match the finite-difference search to ≤ 8.5e-5 relative;
  free-selection EDF vs `mgcv::gam` improved from ~1e-4–2e-4 gaps to ≤ 4e-4
  worst-case across a 30-dataset sweep, most ≤ 2e-5) and `select_lambdas` is
  1.1–9.0× faster on multi-smooth GLM fits (n=2000, m=1..6: REML 2.0–9.0×,
  UBRE/GCV 1.1–3.3×; inner fits 19–163 → 10–19, including the two
  branch-resolution fits described below). The gradient is verified
  against central finite differences of the actual criterion to ~1e-7
  relative across all supported family/link/criterion combinations,
  including non-canonical links, rank-deficient (concurvity) designs and
  near-separation binomial fits.

  Two hardening fixes from the adversarial review of this change (both
  reproduced end-to-end before fixing):
  (1) *Branch resolution for multimodal inner fits.* At near-zero penalty
  the P-IRLS problem can have multiple fixed points; the warm-chained
  search tracks the deep branch (on the panel case, GCV 4.43 vs mgcv's
  4.84 on identical data) while the final fresh refit landed on a shallow
  branch (GCV 38.2) — silently reported with `converged=True`.
  `select_lambdas` now evaluates both branches at the accepted lambdas and
  hands the winner's converged mean back so `gam()`'s final fit continues
  that branch (mgcv never refits from scratch either); the reported
  criterion always belongs to the reported fit.
  (2) *Singular Newton system: warned Fisher fallback, not a crash.* On
  data whose small-lambda optimum drives the Newton system numerically
  singular (reachable: binomial-probit n=60, where mgcv's own optimum
  sits), the gradient now falls back to the always-defined Fisher-weight
  implicit solve for that evaluation with a warning, instead of raising an
  uncaught `ConvergenceError` out of public `gam()` on a fit mgcv
  completes. Relatedly, a diverging inner P-IRLS at a TRIAL lambda during
  the search is now a soft `+inf` barrier (the line search backtracks,
  as mgcv's newton does) rather than an aborted selection; divergence at
  the starting values or at the final fit still fails loud.

### Fixed

- **gam: non-canonical-link REML now uses mgcv's Newton-weight Laplace
  determinant.** `reml_score` computed the Laplace determinant
  `log|X'WX + S_λ|` with FISHER weights; mgcv uses the full-NEWTON Hessian
  weights. The two coincide at canonical links (hence the historical 6e-11
  poisson match), but at non-canonical links the criterion surface itself
  differed — proven exactly (probit: 0.5·(log|A_F|−log|A_N|)=0.0335846 vs
  the observed 0.0335879 score delta; nb-log: −0.0420104 vs −0.0420104) —
  so free REML selection sat ~1e-3–8e-3 EDF from mgcv on probit/nb.
  Fixed: fixed-dispersion non-canonical REML computes `log|X'W̃X + S_λ|`
  (Newton weights, Cholesky on the pivoted rank block), and
  `reml_gradient_glm` differentiates the SAME determinant
  (`dW̃/dη` by a deterministic second central difference). Canonical links
  keep the fit's exact QR-stable factor bit-for-bit. Post-fix: the REML
  score matches mgcv's reported value to ~1e-8 on probit and nb at fixed
  sp (was 0.03–0.04 off), free-selection EDF gaps collapse to ≤3e-5
  (probit +0.000000, nb 3e-5, θ̂ 3.2973 vs mgcv 3.29700), and the R10
  separation hard case matches mgcv's sp/EDF/score with the same R-style
  warning. A non-positive-definite Newton Hessian (PD-tested by Cholesky —
  a slogdet sign misses even-dimensional negative-definite matrices) falls
  back to the Fisher determinant with a warning, value and gradient making
  the same deterministic decision. `reml_score` now takes the design
  matrix `X` (internal API).

- **core: `ConvergenceError` no longer requires an iteration count — gam's
  fail-loud paths raise the documented exception again.** `ConvergenceError`
  demanded a positional `iterations` argument, but the 4.6.0 gam P-IRLS
  fail-loud sites (non-finite working response; step-halving divergence)
  construct it with a message only — so a genuinely diverging GAM fit raised
  `TypeError` instead of the documented `ConvergenceError`, breaking
  `except ConvergenceError` handlers (still loud, wrong type). `iterations`
  is now optional (`None` when the failure is not tied to an iteration
  count); all 27 existing call sites that pass it are unchanged.

- **regression GLM: estimated dispersion for quasi/inverse-Gaussian families now
  uses R's Pearson convention.** Added a `Family.dispersion_estimator` hook: the
  quasi-likelihood and inverse-Gaussian families report the Pearson-chi²/df
  dispersion `summary.glm` uses (the definitional quasi-likelihood estimate),
  rather than the deviance/df default. Existing families (Gaussian/Gamma/NB) keep
  their prior convention unchanged. Applied on both the CPU and GPU IRLS backends.

- **gam: refuse the newly-registered quasi/inverse-Gaussian families loudly.**
  With the quasi and inverse-Gaussian families now in the shared family registry
  (VA-5), `gam` would otherwise accept them via `resolve_family` despite never
  having been validated against `mgcv` for them. `gam` now whitelists its
  validated families (gaussian, binomial, poisson, Gamma, negative.binomial) and
  raises a clear `ValidationError` for anything else, preserving fail-loud
  behaviour.
