# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **timeseries: `stl` rewritten as a faithful clean-room replica of R
  `stats::stl` (showstopper fix).** The previous "simplified" implementation
  returned a materially wrong decomposition on strongly-trending series while
  claiming R parity: on `stl(co2, period=12)` it produced a seasonal range of
  82.8 (true: ~6.2) and a *decreasing* end-of-series trend ([344.7, 342.2,
  339.5]) on a monotonically rising series (R: [364.2, 364.3, 364.5]) — the
  trend error was silently absorbed into a U-shaped "seasonal" artifact. Root
  causes: the cycle-subseries smoother was a fixed-bandwidth degree-1 kernel
  (R: q-nearest-neighbour tricube loess, default degree 0) that never extended
  the subseries by one period at each end, and the low-pass filter used
  zero-padded `mode="same"` convolutions instead of the exact
  MA(p)→MA(p)→MA(3)→loess cascade on the extended series. The new
  implementation (new modules `timeseries/_loess.py`, `_stl.py`,
  `_stl_robust.py`; `_decomposition.py` now hosts only classical
  `decompose()` and the shared result types) follows Cleveland, Cleveland,
  McRae & Terpenning (1990) and replicates R's algorithm exactly: R's loess
  bandwidth/endpoint rules, the jump-stride evaluation grid with linear
  interpolation (including the trailing-window quirk), running-sum moving
  averages in R's summation order, the outer bisquare robustness loop, and —
  required for robust parity on even-length series — R's partial-sort
  behaviour in the 6·MAD scale estimate, whose descending position requests
  make it return a non-order-statistic element on some inputs (roughly 10% of random
  even-length vectors, growing with length; R inherits this from the netlib
  `psort`). Verified
  against R 4.5.2 on 15 reference cases (co2, AirPassengers, sunspots, lynx,
  Nile; periodic/robust/matched-window/jump=1/partial-period variants, stored
  in `tests/fixtures/stl_r_reference.json`): worst absolute component
  difference 2.8e-11 (per-case worsts range 6e-13 to 2.8e-11).
- **timeseries: `stl` API extended and defaults aligned to R (behaviour
  changes).** New parameters `seasonal_degree` (default 0, R's `s.degree` —
  the old code hardcoded degree 1), `trend_degree` (1), `lowpass_window`
  (`nextodd(period)`), `lowpass_degree` (= `trend_degree`), and
  `seasonal_jump`/`trend_jump`/`lowpass_jump` (R's `ceiling(window/10)`
  defaults; pass 1 to evaluate the loess at every point). Existing kwargs
  `seasonal_window`, `trend_window`, `robust`, `n_inner`, `n_outer` keep their
  meaning. Deliberate behaviour changes: (1) `seasonal_window` now defaults to
  `"periodic"` (R requires `s.window` explicitly; the old default was a fixed
  window that produced the wrong decomposition above); (2) `robust=True` now
  defaults to R's `n_inner=1, n_outer=15` (was effectively `n_inner=2,
  n_outer=1`); (3) series length must exceed `2*period` (R's rule; exactly
  `2*period` was previously accepted); (4) window spans must be odd integers
  >= 3 — R silently rounds even spans up, PyStatistics raises
  `ValidationError` (fail-loud); the old `>= 7` floor for `seasonal_window`
  is relaxed to R's `>= 3`. `solution.info` now discloses the resolved
  windows/degrees/jumps, iteration counts, and final robustness weights.
- **timeseries: `stl` CPU performance vs R — known gap, disclosed.** The
  rewrite is vectorised (grouped cycle-subseries batches, batched loess
  evaluation, exact vectorised interpolation) but remains slower than R's
  Fortran on like-for-like calls: co2 periodic 1.3 ms vs R 0.43 ms (~3x),
  sunspots (n=2820) 5.8 ms vs 1.4 ms (~4x), robust AirPassengers 5.4 ms vs
  0.35 ms (~15x). Remaining cost is Python/numpy per-call overhead on small
  windows; closing it fully needs a native (Cython/C) kernel — flagged for a
  follow-up decision rather than shipped silently.
- **timeseries: `ets` gains R-style `"Z"` wildcard automatic model selection;
  default changed to `model="ZZZ"` (deliberate default change, matching
  `forecast::ets`).** Each component of the model string may now be `Z`
  (`"ZZZ"`, `"ZZN"`, `"MZZ"`, ...); candidates consistent with the fixed
  letters are enumerated exactly as `forecast::ets` does with its defaults
  (`restrict=TRUE`, `allow.multiplicative.trend=FALSE`): error Z → {A,M},
  trend Z → {N,A,Ad}, season Z → {N,A,M}; additive-error×multiplicative-season
  excluded; multiplicative candidates dropped on non-strictly-positive data
  (explicit requests raise instead); seasonal candidates dropped for
  period 1, n <= period, or period > 24 (explicit seasonal letters raise, as
  R stops). Each candidate is fitted with the existing ETS engine and the
  model minimising `ic` (new parameter: `"aicc"` default / `"aic"` / `"bic"`)
  is returned with the full candidate table and skip reasons disclosed in
  `solution.info["selection"]`. New module `timeseries/_ets_select.py` hosts
  the public `ets()`; the fitting engine in `_ets_fit.py` is now
  `fit_ets_model()` (private path — public imports via
  `pystatistics.timeseries.ets` are unchanged). Verified against R
  forecast 9.0.0 on 10 reference selections
  (`tests/fixtures/ets_r_reference.json`): candidate sets match R exactly on
  all 10; the selected model matches R on USAccDeaths, Nile, WWWusage and
  diff(Nile), and differs on near-tied tables (AirPassengers, co2, lynx)
  because the engine optimises a wider parameter space than R's defaults
  (independent (0,1) beta/gamma bounds vs R's beta<alpha, gamma<1-alpha;
  m free initial seasonal states vs R's m-1 normalised; phi <= 0.999 vs
  0.98) and reaches slightly better likelihoods for trended candidates —
  documented in `_ets_select.py`; full selection parity would require
  aligning the engine's parameter space (follow-up decision). Two further
  documented divergences: a fully-specified model with `damped=None` is
  fitted exactly as written (R would also try the damped twin), and very
  short series raise instead of falling back to R's unoptimised Holt-Winters
  path.
- **timeseries: `ets` wildcard-path hardening (adversarial-review fixes).**
  `period` is validated as a non-bool integer at the public entry (floats/
  strings previously crashed with a raw `TypeError` from deep in the engine
  once seasonal candidates were enumerated); empty and too-short series raise
  `ValidationError` instead of a raw numpy reduction error; 2-D input now
  raises instead of being silently flattened (both the wildcard and the
  fully-specified path); `damped=True` with a trend fixed to `'N'` (e.g.
  `model="ZNN"`) raises like R's 'Forbidden model combination' instead of
  being silently ignored; the `period <= 24` seasonal limit is enforced for
  fully-specified strings too; candidate enumeration now follows R's exact
  loop order (damped variant innermost, per season); and the
  no-selectable-model errors are accurate — series too short for finite AICc
  (n < 5 for the default `ets(y)`) raises `ValidationError` naming the
  remedy (`ic='aic'`/`'bic'` or a concrete model) rather than a false 'no
  candidate could be fitted' `ConvergenceError` (note: the pre-4.6.2 default
  `model="ANN"` fitted n=3-4; the new selection default requires n >= 5).
- **timeseries: `stl(x, p, seasonal_window=None)` selects the default**
  (`"periodic"`), consistent with every other window parameter, instead of
  raising.
- **tests/fixtures:** R generator scripts committed alongside the fixtures
  (`tests/fixtures/generate_stl_r_reference.R`,
  `tests/fixtures/generate_ets_r_reference.R`) for provenance; ndiffs
  default-flip tests made non-vacuous (signature default + a series where
  KPSS and ADF disagree); ets small-n failure path and previously-unused
  fixture entries wired into tests.
- **timeseries: `ndiffs` default changed from `test="adf"` to `test="kpss"`
  to match R `forecast::ndiffs` (deliberate default change).** The two tests
  have opposite null hypotheses and can recommend different `d` on borderline
  series; both remain available and numerically unchanged — only the default
  flipped.
- **timeseries: ETS log-likelihood/AIC reporting convention documented (no
  numeric change).** `ets` reports the full Gaussian log-likelihood;
  R `forecast::ets` reports Hyndman's concentrated pseudo-log-likelihood
  `-0.5*n*log(SSE)`. The two differ by the exact constant
  `0.5*n*[log(n/(2*pi))-1]` (e.g. +88.36 at n=100; verified against stored
  forecast::ets fits to <0.001), the parameter count `k` is identical, so AIC
  differences, rankings, and `"ZZZ"` selection are unaffected — only the
  printed numbers differ. Documented on `ETSParams`/`ETSSolution`
  (`log_likelihood`, `aic`, `aicc`, `bic`), in the `_ets_fit.py` module
  docstring, and in the `timeseries` package docstring.
- **tests:** new R-parity suites `tests/timeseries/test_stl_r_parity.py`
  (15 fixture cases × components/weights/resolved-parameters + headline co2
  assertions), `test_loess.py` (kernel invariants: polynomial reproduction,
  jump interpolation, weight handling, fallbacks), `test_stl_robust.py`
  (partial-sort semantics and bisquare weights); ETS ZZZ candidate-set/
  selection/failure tests and log-lik-convention tests appended to
  `test_ets.py`; `ndiffs` default tests in `test_acf_stationarity.py`;
  existing STL/ETS tests updated for the deliberate behaviour changes above.
  New fixtures: `tests/fixtures/stl_r_reference.json` (R 4.5.2),
  `tests/fixtures/ets_r_reference.json` (forecast 9.0.0).
