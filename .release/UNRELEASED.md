# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **survival: split `solution.py` into one module per Solution class** (internal
  refactor, no behavior change). `KMSolution`, `LogRankSolution`, `CoxSolution`
  and `DiscreteTimeSolution` now live in `_solution_km.py`, `_solution_logrank.py`,
  `_solution_cox.py` and `_solution_discrete.py`; `pystatistics.survival.solution`
  remains as a re-exporting facade, so all existing imports keep working. Class
  bodies are byte-identical to the pre-split file.
- **survival: Cox fitting is now at or below R's speed, and hardened against
  degenerate/ill-scaled inputs** (found while validating the feature cluster;
  applies to the whole Cox path, stratified and unstratified).
  - Performance: the concordance C-statistic count is now a numba-compiled
    Fenwick kernel and the Efron tie correction is fully vectorized (no
    per-tie-group Python loop). Cox fitting went from ~5-11x slower than R on
    tied data to R parity-or-faster across n=8k-50k (e.g. n=20000 integer-time
    ties: was ~11x slower, now ~0.93x). Results are bit-identical to the pure
    -Python reference (numba enters as a required dependency, same house pattern
    as the ARIMA/ETS kernels). New module
    `pystatistics/survival/_concordance_kernel.py`.
  - Covariates are mean-centered before fitting (as R does). The Cox
    likelihood/score/information are exactly invariant to this shift, but on
    large-magnitude covariates (epoch timestamps, genomic coordinates, large
    monetary sums) the previous uncentered information matrix lost precision to
    cancellation and could report `se=0` / `p=1` for a genuinely significant
    coefficient. Such fits now match R exactly.
  - Fail-loud input validation: non-finite `time`, non-finite / non-0/1
    `event`, and missing (NaN/None) `strata` labels are now rejected at the
    `SurvivalDesign` boundary instead of silently corrupting risk sets or
    dropping rows.
  - A zero-event (fully censored) Cox fit now reports `converged=False` and
    `concordance=NaN` (matching the discrete-time model and R's NA), instead of
    a fabricated `converged=True` / `concordance=0.5`. Harrell's C is likewise
    `NaN` when there are no comparable pairs.
  - `KMSolution.median_survival` now follows R survfit's `minmin` convention:
    when the curve touches exactly 0.5 it averages that time with the next time
    the curve drops below 0.5 (previously returned the first time only).
- **survival: robust / cluster-robust Cox standard errors — `coxph(robust=True,
  cluster=...)`** (VA-8). The Lin-Wei sandwich estimator (`robust=True`) reports
  Huber-White SEs; `cluster=` groups correlated rows of one subject into a
  single independent unit (implies `robust=True`). `.standard_errors` /
  `.z_values` / `.p_values` / `.conf_int` then reflect the robust SE, with the
  model-based SE kept on `.naive_standard_errors` and a `.robust` flag;
  `summary()` prints both columns like R. Built on exact dfbeta / score
  residuals (`residuals.coxph(type='dfbeta')`) including the Efron tie
  correction, and composes with `strata=` and counting-process `start=`.
  Validated vs `coxph(robust=TRUE)` / `coxph(cluster=id)` to machine precision.
  Naming per CONVENTIONS Amendment A9 (`robust=` is the misspecification-robust
  variant, distinct from `timeseries`'s LOESS `robust`; `cluster=` reserved for
  independent-unit grouping). New module `pystatistics/survival/_cox_robust.py`.
- **survival: counting-process / time-varying Cox — `coxph(start=...)`** (VA-8).
  Each row is at risk on `(start, time]`, so a subject may span several rows
  with different covariate values (time-dependent covariates) or enter late
  (left truncation). Matches `coxph(Surv(start, stop, event) ~ x)` with `time`
  as the stop: coefficients, SEs, log-likelihood, iteration counts, and
  concordance to machine precision on the reference fits, composed with
  `strata=`, heavy ties (Efron + Breslow), and `cox_zph`. Risk sets are
  computed with an entry-side reverse-cumsum correction (still one O(n)
  sweep); Harrell's concordance was generalized to an ascending
  activation/deactivation sweep (still O(n log n)) that counts each pair only
  while co-at-risk — non-overlapping spells of one subject never form a pair.
  Naming per CONVENTIONS Amendment A8 (`start=` on coxph, `entry=` on
  kaplan_meier; internally one validated `SurvivalDesign.entry` field). A row
  with `start >= time` is refused loudly (R NA-drops such rows with a warning
  — a documented, deliberately stricter deviation).
- **survival: left-truncated Kaplan-Meier — `kaplan_meier(entry=...)`** (VA-8).
  Delayed-entry risk sets `n_risk(t) = #{entry < t <= time}`, matching
  `survfit(Surv(entry, time, event) ~ 1)` (and `~ g` composed with `strata=`)
  to machine precision on curve, SE, and CI across conf types.
- **survival: new `cox_zph()` — test of the proportional-hazards assumption**
  (VA-8). Score test on scaled Schoenfeld residuals, implementing the modern
  survival >= 3.0 formulation of `survival::cox.zph` (not the pre-3.0
  correlation test). `cox_zph(fit, transform='km'|'rank'|'identity'|'log')`
  takes a fitted `CoxSolution` (stratified or not, Efron or Breslow ties) and
  returns a `CoxZphSolution` with per-covariate + GLOBAL chi-square/df/p, the
  scaled Schoenfeld residual matrix (`.residuals`), transformed event times
  (`.x`), and `.var` — all matching R to machine precision on the reference
  fits (incl. heavy-ties Efron/Breslow and stratified fits). `CoxSolution` now
  retains its fitted design (like `GLMSolution`) to support post-fit
  diagnostics. New module `pystatistics/survival/_cox_zph.py`.
- **survival: `kaplan_meier(strata=...)` now returns per-stratum survival curves**
  (A1). Previously `strata=` raised `NotImplementedFeatureError`. Returns a new
  `StratifiedKMSolution` holding one ordinary `KMSolution` per stratum (index by
  label: `sol["A"]`, or iterate `sol.curves`), matching
  `survfit(Surv(time, event) ~ g)`. Each curve is validated against R to machine
  precision (survival/times/n_risk, se, and CI on all conf types); a stratum with
  no events yields an empty curve, as R's `summary(survfit)` does. New module
  `pystatistics/survival/_km_strata.py`.
- **survival: `coxph(strata=...)` now fits a stratified Cox proportional hazards
  model** (A1). Previously `strata=` raised `NotImplementedFeatureError`.
  - Shared coefficient vector across strata, a separate baseline hazard (separate
    risk sets) per stratum — the stratified partial likelihood. Matches
    `survival::coxph(Surv(t, e) ~ x + strata(g))` for both `ties='efron'` and
    `ties='breslow'`: coefficients and log-likelihood to machine precision,
    standard errors ~1e-7 relative, on the R reference fits (R 4.5.2 /
    survival 3.8.3). Concordance aggregates comparable pairs *within* each stratum
    (R's `concordance.coxph` convention).
  - New `CoxSolution.n_strata`; `summary()` notes the stratum count. The
    stratified fit is CPU-only, consistent with the unstratified path.
  - New modules `pystatistics/survival/_cox_strata.py` (stratified fit) and
    `pystatistics/survival/_cox_newton.py` (the Newton-Raphson driver now shared
    by the single-stratum and stratified paths, so their convergence logic cannot
    drift apart).
- **survival: Cox fitting is more robust and now matches R's diagnostics** (found
  while validating stratified Cox; applies to the unstratified path too).
  - Added R's backtracking line search (`_cox_newton`): the Newton step is halved
    whenever it decreases the partial log-likelihood. Ill-conditioned fits that
    previously failed to converge (e.g. covariates on very different raw scales,
    such as flchain `age` + serum free-light-chain values) now converge to R's
    estimate in a comparable iteration count. Well-behaved fits are unchanged.
  - `coxph` now emits R's "coefficient may be infinite" warning when the fit
    plateaus while a coefficient runs to +/- infinity (monotone likelihood /
    separation), instead of silently returning the large value — matching
    `survival::coxph`. New `CoxParams.infinite_coefs`.
  - Fixed Harrell's concordance tie handling: a subject censored at an event time
    is now correctly counted as outliving the event (a comparable pair), matching
    R's convention. Previously these pairs were dropped, biasing the C-statistic
    on data with tied event/censoring times; concordance now matches R exactly
    rather than approximately.
- **timeseries: `arima` / `auto_arima` now support regression with ARIMA errors
  (`xreg`), drift (`include_drift`), and parameter masking (`fixed=`)** (VA-4 /
  VA-4b). Previously these R capabilities were absent (documented only as a TODO).
  - `arima(y, order, xreg=X)` fits `y = X @ beta + eta` where `eta` follows the
    ARIMA process, matching `stats::arima(xreg=)` / `forecast::Arima(xreg=)`. The
    regression coefficients are reported on the solution as `xreg_coef` (with a
    `'drift'` / `'intercept'` / `xreg1..xregk` naming in `xreg_names`), with joint
    standard errors in the trailing block of `vcov`. Validated vs `stats::arima`:
    coefficients ~5e-6, log-likelihood ~1e-9, AIC/sigma2 machine-precision,
    standard errors ~4e-5 on the reference fits.
  - `include_drift=True` adds a linear time-trend regressor (the models R reports
    "with drift"); it reproduces R's drift/`d` interaction exactly (an
    ARIMA(p,1,q) "with drift" is an ARMA-with-mean on the differenced series).
    Fails loud when total differencing `d + D >= 2` (the trend is unidentifiable).
  - `fixed=` holds coefficients fixed during estimation, as a Pythonic
    `{name: value}` mapping (e.g. `fixed={'ma1': 0}`) or R's positional nan-vector.
    Fixed coefficients carry zero variance in `vcov` and are excluded from the
    information criteria. Matches `stats::arima(fixed=)`.
  - `auto_arima(..., allowdrift=True)` (default) now SELECTS drift models when the
    total differencing order is 1 and drift lowers the information criterion —
    matching `forecast::auto.arima`'s "with drift". Non-drift and seasonal
    selections (e.g. AirPassengers `(0,1,1)(0,1,1)[12]`) are unchanged.
  - `forecast_arima(..., newxreg=)` forecasts regression-with-ARIMA-errors models:
    future regressor values are required for `xreg` models; drift/intercept future
    columns are synthesized. Point forecasts and prediction SEs match
    `predict.Arima` to ~1e-6. (Prediction intervals use the MLE `sigma2` that
    matches `stats::arima`/`predict.Arima`; `forecast::Arima` reports a
    df-adjusted `sigma2 = SSR/(n - ncoef)` and hence slightly wider intervals — a
    documented convention difference, consistent with the rest of the module.)
  - The exact-ML/CSS regression path is CPU-only (the Whittle / `arima_batch` GPU
    kernels do not carry a regression term); `xreg`/`include_drift`/`fixed` fail
    loud with `method='Whittle'`.
  - New module `pystatistics/timeseries/_arima_xreg.py`; the general numerical
    Hessian moved to `_arima_likelihood.compute_numerical_hessian`. The plain
    (no-regressor) ARIMA fit path is unchanged.
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
