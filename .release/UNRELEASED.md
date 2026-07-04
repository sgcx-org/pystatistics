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

- **tests:** new `tests/timeseries/test_arima_ic.py` (IC self-consistency,
  R free-k parity, mean-under-differencing, degenerate AICc, auto_arima
  seasonal selection = R, validation failures) and
  `tests/timeseries/test_stationarity_mackinnon.py` (ADF default/statistic/
  p-value vs statsmodels fixtures incl. near-unit-root ≈0.92, saturation,
  monotonicity, CV failure cases; KPSS bandwidth rules, tseries parity,
  lshort validation). Reference fixtures + generator scripts under
  `tests/fixtures/` (R 4.5.2, tseries 0.10-58, statsmodels 0.14.6). Full
  timeseries suite: 459 passed, 0 failures.
