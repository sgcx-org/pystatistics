# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **timeseries: seasonal ARIMA information criteria counted the wrong
  parameters (RIGOR R18 showstopper).** `arima(..., seasonal=...)` in
  `_arima_fit.py` computed `aic`/`aicc`/`bic` with `k = len(opt_params) + 1`,
  where `opt_params` is the EXPANDED multiplicative polynomial — the airline
  model `(0,1,1)(0,1,1)[12]` was counted as k=15 (expanded MA order 13 + mean
  + sigma2) instead of the free-parameter k=3 (ma1, sma1, sigma2), inflating
  its AIC from 1021.00 to 1044.90; `(1,0,1)(1,0,1)[12]` was counted k=28
  instead of 6. `ARIMASolution` was internally inconsistent: `.n_params`
  (already the free count) disagreed with the k inside `.aic`. Now
  `k = p + q + P + Q + (1 if mean estimated) + 1`, matching R `stats::arima`'s
  `length(coef) + 1` convention, and `.aic == -2*loglik + 2*n_params` holds
  for every model (seasonal, non-seasonal, differenced, degenerate).
  Verified on AirPassengers vs R 4.5.2: airline AIC 1021.0030 vs R 1021.0029,
  AICc 1021.1919 (exact), `(2,1,1)(0,1,0)[12]` AICc 1018.1655 vs R 1018.1652.

- **timeseries: `arima()` estimated a mean for differenced models; R drops
  it.** `stats::arima` ignores `include.mean` when d + D > 0; we estimated
  one anyway (e.g. `(2,1,1)` on AirPassengers got mean=2.669, acting as an
  implicit drift term), which both mis-counted k and changed the fit
  (loglik −675.85 with spurious mean vs R's −685.17 without). `arima()` now
  forces `include_mean=False` when d + D > 0. After the change the differenced
  fits match R exactly: `(2,1,1)` loglik −685.1690 (R −685.1690), airline
  −507.5015 (R −507.5014). Consequence for forecasts: differenced models no
  longer carry an implicit drift (matches R `stats::arima`; explicit drift à
  la `forecast::Arima(include.drift=)` remains unsupported).

- **timeseries: degenerate `(0,d,0)` AICc correction fixed** in
  `_arima_fit.py`: was `AIC + 2/(n-2)`, correct k=1 formula is
  `AIC + 2k(k+1)/(n-k-1) = AIC + 4/(n-2)`. Matters now that `(0,1,0)`-type
  candidates (mean-free) hit this path during `auto_arima` search.

- **timeseries: `auto_arima` seasonal search never varied (P, Q)** —
  `_stepwise_search`/`_grid_search` in `_arima_order.py` pinned the seasonal
  order at `(1, D, 1)`, so R's AirPassengers pick `(2,1,1)(0,1,0)[12]` was
  unreachable. Stepwise now follows Hyndman–Khandakar: initial candidates
  `(2,d,2)(1,D,1)/(0,d,0)(0,D,0)/(1,d,0)(1,D,0)/(0,d,1)(0,D,1)` and ±1 moves
  on P/Q (alone and jointly) next to the existing p/q moves; grid search
  iterates the full P/Q grid. `AutoARIMAParams.best_seasonal` now reports the
  SEARCHED seasonal order; seasonal `search_results` entries are
  `((p,d,q), (P,D,Q,m))` pairs. Combined effect of the three ARIMA fixes:
  `auto_arima(AirPassengers, period=12, ic='aicc')` selects
  `(2,1,1)(0,1,0)[12]` AICc 1018.17 — identical to
  `forecast::auto.arima` — where it previously selected
  `(2,1,2)(1,1,1)[12]` (reported AICc 1088.4; R-correct AICc 1025.7, i.e.
  7.5 worse than R's pick). Non-seasonal AirPassengers now selects `(2,1,3)`
  AICc 1350.22 (R's own `arima()` reproduces our fit and AICc to 4 decimals);
  R's greedy no-drift stepwise stops at `(4,1,2)` AICc 1374.39, so we beat it
  by R's own criterion — we do NOT add a drift feature to chase R's
  with-drift pick `(4,1,2)+drift` (AICc 1357.22).

- **timeseries: ADF p-value was materially wrong in the fail-to-reject
  region.** `_adf_pvalue` in `_stationarity.py` linearly interpolated between
  the 1%/5%/10% critical values and extrapolated above the 10% point: a
  near-unit-root series (stat −1.1426, ct) got p=0.4433 where MacKinnon says
  0.9216 (tseries: 0.9144). Replaced with the MacKinnon (1994) response
  surface in new module `_adf_mackinnon.py` (same surface statsmodels
  `adfuller` uses), valid across the whole range — no 0.01 floor. Verified vs
  statsmodels 0.14.6: max |p diff| 3e-16 on a dense τ grid across nc/c/ct;
  36 series/lag/regression configs max diff 4e-13. Statistic untouched
  (already exact vs tseries AND statsmodels). Reported critical values
  upgraded from the MacKinnon 1996 two-term adjustment to the 2010
  finite-sample response surface (matches statsmodels `mackinnoncrit`).

- **timeseries: `adf_test` default `regression` changed 'c' → 'ct'.** The
  docstring claimed "'c' matches R" — false: `tseries::adf.test` always uses
  constant + trend. With the 'ct' default the statistic reproduces tseries
  exactly (verified 4 series, d < 1e-6). 'nc'/'c'/'ct' all remain available.
  `ndiffs` in `_differencing.py` pins its internal call to `regression='c'`
  (the drift variant forecast::ndiffs uses), so the new default does not
  leak into ndiffs; the MacKinnon p-value surface itself flips no
  ndiffs(test='adf') decision on an 18-series battery (see the ndiffs
  bullet below).

- **timeseries: `kpss_test` default bandwidth aligned to
  `tseries::kpss.test`.** Was `floor(3*sqrt(n)/13)` (lag 3 at n=200, 2 at
  n=100); tseries uses `trunc(4*(n/100)^(1/4))` (`lshort=TRUE`; 4 at both).
  New `lshort: bool = True` parameter mirrors tseries (`lshort=False` →
  `trunc(12*(n/100)^(1/4))`); explicit `n_lags` overrides it. At matched
  bandwidth statistic and interpolated p-value reproduce tseries exactly
  (adversarially verified: 36/36 cases across 9 series × Level/Trend ×
  lshort TRUE/FALSE, lag equal to tseries' in every case, stat diff
  ≤ 2e-13, p diff ≤ 4e-15). Removed a dead duplicate `critical_values`
  construction in `kpss_test`.

- **timeseries: `ndiffs` KPSS bandwidth pinned to forecast's rule.**
  `forecast::ndiffs` does NOT use the tseries/urca `lags="short"`
  bandwidth — it passes `use.lag = trunc(3*sqrt(n)/13)` (the OLD
  pystatistics default). Cross-verification caught `ndiffs(WWWusage)`
  flipping 1 → 0 (away from forecast) when kpss_test's new default
  leaked in; `ndiffs` now passes `n_lags=trunc(3*sqrt(n)/13)`
  explicitly. Verified against `forecast::ndiffs` on 18 series
  (including the WWWusage borderline case, now pinned in the fixture).
  Note the ADF-path p-value change flips NO ndiffs decision on that
  battery (old vs new identical on all 18).

- **timeseries: `kpss_test` fails loud on degenerate input.** An exactly
  constant series (or exactly linear series with `regression='ct'`)
  produced a 0/0 statistic reported as confident rounding noise
  (stat 4.4, p=0.01 — "rejects stationarity" for a constant); now raises
  `ValidationError`. The unreachable `return 0.05` interpolation
  fallback in `_kpss_pvalue` now raises instead of silently reporting
  p=0.05.

- **timeseries: seasonal-AR exact-ML log-likelihood was ~80 units below R
  (silent diffuse-init fallback).** `_stationary_init` in `_arima_kalman.py`
  solved the Lyapunov equation `P = TPT' + RR'` by linear fixed-point
  iteration with max_iter=200 and ABSOLUTE tol 1e-12; a moderately
  persistent seasonal AR (sar1=-0.47 at lag 12 → spectral radius 0.94,
  rate 0.88/iter) converged to within 1.4e-11 of the exact solution but
  missed the tolerance, so the near-perfect P was DISCARDED for the
  diffuse kappa=1e6 init — shifting loglik by ~80 units on
  `(1,1,1)(1,1,0)[12]` models (log-AirPassengers: py 160.5 vs R 241.7;
  nottem: −614.5 vs −536.9) and corrupting IC ranking of seasonal-AR
  candidates. Replaced with a doubling iteration (S_{j+1} = S_j + A_j S_j
  A_j', A_{j+1} = A_j²; quadratic in horizon) with RELATIVE tolerance;
  diffuse fallback now only for genuinely non-stationary points, matching
  R (which solves Q0 exactly and never falls back for stationary models).
  Both models now match R: loglik 241.7298 vs 241.7332 / −536.8834 vs
  −536.8839, coefficients to ~5 decimals. Removed the now-dead
  `_sparse_T_times_M_times_TT` helper and scipy `solve_discrete_lyapunov`
  import/fallback (`_initial_covariance`).

- **timeseries: seasonal ARIMA forecasts dropped the seasonal
  coefficients; SEs ignored differencing and state uncertainty.**
  `forecast_arima` in `_arima_forecast.py` fed `fitted.ar/.ma` — the
  factored NON-seasonal coefficients — into the point-forecast recursion
  and psi weights (airline forecasts up to ~5.4 off R `predict()`), and
  computed SEs as `sigma*sqrt(cumsum(psi²))` of the differenced-scale ARMA
  only (a random walk reported flat se=sigma at every horizon instead of
  sigma*sqrt(h)). Rewritten: point forecasts come from the exact Kalman
  filtered state via new `kalman_arma_forecast` in `_arima_kalman.py`
  (R's `KalmanForecast` approach; the CSS-residual recursion carries
  conditioning error at the sample end when an MA root is near the unit
  circle — (2,1,1)(0,1,0)[12] with ma1=-0.98 was ~1.4 off R even with
  the right polynomials), using the multiplied-out effective polynomials;
  SEs aggregate the full h×h Kalman forecast-error covariance through the
  integration operator `1/((1-B)^d (1-B^m)^D)`. Verified vs R
  `predict.Arima` on airline, (2,1,1)(0,1,0)[12], (2,1,1), (0,1,0), and
  both (1,1,1)(1,1,0)[12] fits: max |mean diff| 0.0014, max se rel diff
  0.002%. The dead `_forecast_differenced` recursion was removed.

- **timeseries: `sigma2` now reports the Kalman profile estimate for
  ML-family fits** in `_arima_fit.py` (was CSS-residual SSE/n for all
  methods, 2.9% high on the near-unit-MA model: 133.09 vs R's 129.31 —
  the residual source of forecast-SE error). Matches R `stats::arima`
  to 4 decimals on the reference fits; pure-CSS fits keep SSE/n (R's CSS
  convention).

- **timeseries: seasonal `vcov`/`summary()` standard errors were read
  from the wrong matrix.** The Hessian was computed over the EXPANDED
  polynomial parameters, so `summary()` printed seasonal s.e. from
  structurally-zero expanded lags (airline sma1: 0.38 printed vs R's
  0.0828). `_compute_hessian` now takes a callable and seasonal fits
  differentiate in the FACTORED parameterization (ar, ma, sar, sma,
  mean) using the exact-ML objective; airline and (1,1,1)(1,1,0)[12]
  s.e. now match R to 4 decimals. Non-seasonal fits keep the existing
  CSS-Hessian convention. `vcov` docstring ordering is now actually true.

- **timeseries: ML stage of CSS-ML is now better-of-two-starts for
  mean-carrying fits.** The exact stationary init exposed a flat canyon
  in the (ar, mean) surface toward the AR unit root: the CSS stage can
  hand ML a drifted-mean basin L-BFGS-B cannot leave (AirPassengers
  (1,0,1): mean 115.7, loglik 1.84 below R). `_optimize_arima` and
  `optimize_arima_factored` now also run the ML stage from the original
  Yule-Walker/sample-mean start and keep the better optimum — same
  pattern as the 4.6.3 damped-ETS fix; fits improve or stay identical.
  (1,0,1) now matches R: loglik −700.8744 vs −700.8741 at the same
  coefficients. Only `include_mean` fits pay the extra ML run.

- **timeseries: `auto_arima` now applies forecast::auto.arima's
  near-unit-root candidate veto and replicates its stepwise walk.** With
  the corrected seasonal-AR likelihood, boundary models (AR/MA root
  pile-up at the unit circle chasing the differencing operator) started
  winning raw AICc — AirPassengers briefly selected (3,1,3)(1,1,2)[12]
  at 1004.8. forecast deliberately excludes candidates whose expanded
  AR or MA polynomial has a root with modulus < 1.01 (`myarima` sets
  ic=Inf); `_try_fit` now does the same (`_has_near_unit_roots`), and
  the stepwise search now replicates forecast's exact move priority
  (seasonal P/Q moves first, singles before joint moves) with
  first-improvement restart — the walk's PATH determines which local
  optimum a greedy search reaches, and an all-neighbours sweep stopped
  at (0,1,1)(2,1,0)[12] AICc 1019.5 where R's walk reaches
  (2,1,1)(0,1,0)[12] at 1018.2. AirPassengers seasonal selection again
  matches R exactly. `arima()` itself still fits whatever is requested
  (veto applies to automatic selection only, exactly like R).

- **timeseries: point forecasts for d ≥ 2 (or seasonal D ≥ 2) were
  catastrophically wrong.** `_undifference` in `_arima_forecast.py`
  walked the integration ladder in reverse (first cumsum seeded with the
  tail of the RAW series instead of the most-differenced one) — invisible
  for d + D ≤ 1, but an ARIMA(1,2,1) 12-step mean was off by ~13,600
  while its (fixed) SEs matched R to 7.7e-5. Pre-existing, caught by the
  adversarial review of the forecast rewrite. Integration now proceeds
  from the most-differenced scale outward; verified vs R `predict()`:
  (1,2,1) mean diff 0.0006, (0,2,0) exact, (0,1,1)(0,2,1)[12] 0.006.

- **timeseries: `auto_arima` now determines d with the KPSS test,
  matching `forecast::auto.arima`'s default.** `_determine_d` hardcoded
  `ndiffs(test='adf')`; on series where ADF and KPSS disagree the search
  ran at the wrong d and could not reach R's model class (wineind: adf
  d=0 vs R/KPSS d=1 → pick 41 AICc worse by R's own accounting;
  WWWusage: adf d=2 → over-differenced (2,2,0) vs R's (1,1,1)). Cross-
  verification proved forcing R's d reproduces R's exact picks on both.
  With KPSS-based d, both now match; the other six benchmark series
  (already KPSS-consistent) are unchanged.

- **timeseries: spurious `ConvergenceError` when the CSS warm-start
  aborts but ML refinement converges.** In `_optimize_arima` the
  `converged` flag kept the CSS stage's failure even after a successful
  ML stage (co2 (2,1,1): ML at −466.830 = R's −466.830, yet arima()
  raised). Pre-existing bookkeeping, newly exposed by the mean-drop
  change; `converged` now reflects the accepted ML result, and the
  factored two-start prefers a converged second optimum over a failed
  first one.

- **timeseries: `sigma2` failure sentinel no longer silent.**
  `kalman_arma_loglik` signals failure with a placeholder sigma2 of 1.0;
  `arima()` would have reported it as a real variance. Now raises
  `ConvergenceError` (unreachable in practice — the ML stage just
  evaluated those parameters — but the silent default violated the
  fail-loud rule).

- **timeseries: removed the dead innovations-algorithm ML path**
  (`_exact_loglik_innovations`, `_innovations_algorithm`,
  `_arma_autocovariance` in `_arima_likelihood.py`; the scipy
  `solve_discrete_lyapunov` fallback in `_arima_kalman.py`). No callers
  since the Kalman path landed, and review found `_arma_autocovariance`
  computes a wrong gamma(0) for mixed ARMA (2.1915 vs true 2.1353 for
  ar=(0.6,−0.3), ma=0.4) — its docstring claimed it was a 1e-6-accurate
  second-source reference. Dead AND wrong code removed rather than
  fixed. `_stationary_init` also symmetrizes its result (float matmul
  drift ~1e-12 on near-unit systems), and stale "innovations algorithm"
  docstrings now say Kalman filter.

- **timeseries: exact-ML fits now report the invertible (canonical) MA
  representation.** The exact-ML likelihood is invariant under
  reflecting MA roots across the unit circle (theta → 1/theta with a
  matching sigma2 rescale), so the optimizer landed on the
  non-invertible mirror on ~10% of boundary-ish fits (AirPassengers
  (1,1,1)(1,1,1)[12]: sma1=1.19, sigma2 30% below R; co2 (1,0,1)+mean:
  ma1=1.46, sigma2 53% below). Statistically the invertible
  representative is the identified/fundamental (Wold) one: only its
  sigma2 is the one-step prediction-error variance, and the
  non-invertible representation made the CSS residual recursion
  diverge. R does exactly this: `stats::arima`'s internal `maInvert`
  normalizes the fitted MA blocks of ML-family fits post-fit and
  recomputes the Hessian/sigma2 at the inverted coefficients — so this
  is required R parity, not an extra nicety (verified by seeding R's
  optimizer in the mirror basin via `init=`: R still reports the
  invertible fit). New `normalize_ma_coefficients` in `_arima_factored.py`
  reflects inside-circle roots per MA FACTOR (preserving the
  multiplicative seasonal structure); roots on the circle are left
  alone (the MA unit-root pile-up under over-differencing is a genuine
  boundary optimum where both representatives coincide); a
  likelihood-invariance guard reverts loudly if the flip is not
  numerically neutral. NOT applied to pure-CSS fits — the CSS criterion
  is not reflection-invariant, so the flipped parameters would not be
  the optimum of the fitted criterion (R's split too). All downstream
  artifacts (sigma2, residuals, vcov, the auto_arima root veto) are
  computed at the normalized point; sigma2 on the previously-mirror
  fits now matches R (0.09% worst), and mirror candidates now survive
  the selection veto exactly when R keeps them. Likelihood, AICc, and
  forecasts are unchanged (equivalence class).

- **timeseries: ML-family `residuals` are now the standardized Kalman
  innovations `v_t/sqrt(F_t)`,** matching R `stats::arima`'s
  `residuals()` (arima.c scales by `sqrt(gain)`). These are the
  model's actual innovations with CONSTANT variance sigma2 at every t —
  the homoscedastic white noise Ljung-Box/ACF/normality diagnostics
  assume — and satisfy `mean(residuals**2) == sigma2` identically
  (verified to machine precision on every reference fit). CSS
  residuals approximate them only up to a conditioning transient
  decaying like the largest MA root modulus^t (~7% still alive at
  t=131 for ma1=−0.98), and diverged outright on the mirror fits.
  Verified against R `residuals()` on the reference battery (aligned
  after the d+D·m observations differencing consumes): max diff ≤6e-4
  at matched fits. `sigma2` for ML fits is now computed as
  `mean(residuals**2)` directly (algebraically identical to the Kalman
  profile estimator; removes a redundant filter pass and the silent
  failure sentinel — `kalman_arma_innovations` raises loudly instead).
  Pure-CSS fits keep the CSS recursion and sigma2 = SSE/n (R's CSS
  convention). New `kalman_arma_innovations` in `_arima_kalman.py`;
  the normalization orchestration lives in
  `_arima_factored.normalize_to_invertible` (keeps `_arima_fit.py`
  under the LOC limit and dedupes the seasonal/non-seasonal guard).
  `kalman_arma_forecast`/`kalman_arma_innovations` raise
  `ConvergenceError` (not `ValidationError`) on internal numerical
  failure — these are not user-input errors.

- **timeseries: `arima()` gains R's `init=` parameter** (new module
  `_arima_init.py`). Layout follows R `coef()` order
  `[ar, ma, sar, sma, mean?]` (mean slot only when a mean is actually
  estimated, i.e. d + D = 0 and include_mean); `numpy.nan` entries are
  filled with defaults (zeros for coefficients, sample mean of the
  differenced series for the mean); non-stationary AR/seasonal-AR
  inits raise ValidationError (R errors "non-stationary AR part");
  non-invertible MA inits are normalized to the invertible
  representative before optimization — R's documented maInvert-on-init
  intent; empirically R's own implementation ERRORS on such inits
  (its CSS/optim stage diverges first), so accepting them is a strict
  improvement over the reference. Rejected loudly for
  method='Whittle'. R-parity pinned in the fixture: the airline warm
  start lands at R's optimum (ma1/sma1 to 1e-5, loglik to 1e-4).

- **timeseries: `arima()` docstring no longer overclaims interface
  parity.** The old text said "Matches the interface and numerical
  approach of R's stats::arima()" while the interface lacked xreg,
  transform.pars, fixed, init, n.cond, SSinit, optim.method/control,
  kappa, with no disclosed reason — a violation of the "do it or give
  a principled reason" promise. The docstring now states the claim
  precisely: numerical-behaviour parity (verified), a documented
  supported subset (order/seasonal/include_mean/method/init),
  not-yet-implemented parameters (fixed, xreg incl. drift), and the
  by-design exclusions with reasons (transform.pars/SSinit/kappa/
  n.cond/optim.* are knobs over R's optimizer and state-space
  internals; we guarantee results parity, not knob parity). The
  timeseries module index claim was likewise corrected. Also
  documents the pure-CSS conditioning difference vs R (we
  zero-initialize over all observations; R conditions on and excludes
  the first n.cond — measured: airline seasonal CSS identical to R,
  (2,1,1) coefficients ~1e-3 apart with sigma2 ~1.3%, weakly
  identified fits may reach different CSS optima; CSS-ML/ML results
  are unaffected and remain covered by the parity guarantee).

- **timeseries: remaining convention notes:** `arima_batch` (Whittle)
  still reports a documented per-series sample mean under d > 0;
  drift/intercept terms for d + D = 1 models remain unsupported
  (`fixed=` and `xreg=` are the tracked interface gaps, disclosed in
  the `arima()` docstring).

- **tests:** new `tests/timeseries/test_arima_kalman_r_parity.py`
  (SAR loglik/coef/IC parity on both failing models, stationary-init
  doubling normal/edge/failure incl. Lyapunov residual and scipy
  agreement, forecast mean+se parity on NINE models vs R
  `predict.Arima` incl. d=2 and seasonal D=2, analytic random-walk se,
  loud forecast failure on non-finite params, sigma2 profile parity,
  factored-vcov s.e. parity and dimensions, near-unit-root veto
  normal/boundary/empty cases, converged-flag guard on the co2 (2,1,1)
  CSS-abort case) with fixture `arima_kalman_r_reference.json` +
  generator (R 4.5.2); `test_arima_ic.py` gains the WWWusage
  KPSS-d-selection regression test. Full timeseries suite incl.
  slow-marked: 521 passed, 0 failures.

- **tests:** new `tests/timeseries/test_arima_ic.py` (IC self-consistency,
  R free-k parity, mean-under-differencing, degenerate AICc, auto_arima
  seasonal selection = R, validation failures) and
  `tests/timeseries/test_stationarity_mackinnon.py` (ADF default/statistic/
  p-value vs statsmodels fixtures incl. near-unit-root ≈0.92, saturation,
  monotonicity, CV failure cases; KPSS bandwidth rules, tseries parity,
  lshort validation). Reference fixtures + generator scripts under
  `tests/fixtures/` (R 4.5.2, tseries 0.10-58, statsmodels 0.14.6). Full
  timeseries suite: 459 passed, 0 failures.
