# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **timeseries: `stl` CPU speed now matches or beats R `stats::stl` — the
  loess/robustness core is compiled with numba (no algorithm or result
  change).** The 4.6.x rewrite matched R numerically but was 3-15x slower
  because the loess kernel, moving-average cascade, cycle-subseries
  smoothing, and the netlib partial-sort robustness step were vectorised
  Python/numpy called many times per decomposition. The entire STL inner +
  outer loop is now a single `@njit(cache=True, fastmath=False)` driver
  (`timeseries/_stl_core.py`), composing numba loess kernels
  (`timeseries/_loess.py`) and robustness kernels
  (`timeseries/_stl_robust.py`) — mirroring R's single Fortran routine.
  Same-machine steady-state timings vs R 4.5.2 (best-of-5, ms/call):
  co2 periodic 1.17→**0.094** (R 0.31); monthly sunspots n=2820 5.63→**0.48**
  (R 1.32); AirPassengers 0.54→**0.041** (R 0.069); co2 periodic robust
  11.6→**0.85** (R 0.90); AirPassengers robust 5.2→**0.29** (R 0.24). Non-robust
  decompositions are now 1.7-3.3x faster than R; robust decompositions match
  R (within tens of microseconds on the smallest series, where fixed
  per-call overhead dominates). Numerical parity is unchanged — all 15
  `stl_r_reference.json` cases pass the 1e-9 gate (worst 2.4e-11, including
  the robustness weights). No packaging change: numba is already a required
  dependency (the ARIMA Kalman filter). Cost: the kernels compile on first
  use (~4 s one-time) and are then loaded from numba's on-disk cache
  (~0.1 s per fresh process), the same trade-off the ARIMA path already
  carries. The clean-room implementation (from the algorithm as documented,
  not R's GPL Fortran) is preserved.
- **timeseries: STL internals refactored for the numba core (no public API
  change).** `timeseries/_loess.py` and `_stl_robust.py` now hold numba
  kernels behind unchanged Python wrappers (`loess_smooth`,
  `loess_subseries_smooth`, `_psort_pair`, `_robustness_weights`); the new
  `timeseries/_stl_core.py` holds the fused driver; the Python
  `_moving_average`/`_low_pass`/`_cycle_subseries`/`_inner_loop` helpers in
  `_stl.py` are removed (folded into the driver). `stl()`'s signature,
  defaults, `info` payload, and results are unchanged.
- **timeseries: ETS engine parameter space aligned with R `forecast::ets`
  defaults (behaviour change — fitted results, `n_params`, and AIC/AICc/BIC
  values change for most models).** Three alignments in
  `timeseries/_ets_fit.py`: (a) smoothing parameters are now optimised over
  R's "usual" region — `beta < alpha` and `gamma < 1 - alpha` enforced at
  every optimiser iterate via alpha-dependent logit bounds — instead of
  independent `(1e-4, 1 - 1e-4)` boxes; (b) seasonal models estimate `m - 1`
  free initial seasonal states with the remaining one (the index used at the
  first observation) determined by the normalisation `sum(s) = 0` (additive)
  / `sum(s) = m` (multiplicative), exactly as R — which reduces `n_params`
  by 1 for seasonal models to match R's count (e.g. AAA period-4: k = 9,
  was 10), makes AICc differences directly comparable to R's, and makes the
  normalisation hold exactly on the fitted `init_season` (was approximate);
  (c) `phi` is bounded to R's `(0.8, 0.98)` (was `(0.8, 0.999)`).
  Free-parameter starting values now follow R's `initparam` (e.g.
  `alpha0 = 1e-4 + 0.2*(1 - 2e-4)/m`), and user-fixed parameters outside the
  usual region raise `ValidationError` (R errors "Parameters out of range";
  previously they were silently coerced into bounds by the logit transform).
  R's `bounds="both"` admissibility intersection is NOT implemented
  (documented in the module docstring). Validated against R 4.5.2 +
  forecast 9.0.0 per-candidate: where both engines find the same optimum the
  log-likelihoods now agree to ~1e-6 and AICc (after the documented
  convention constant) matches R exactly.
- **timeseries: ETS default optimiser tolerance tightened `tol=1e-8` →
  `1e-10`** (`ets()` and `fit_ets_model()`): at `1e-8` L-BFGS-B stopped
  ~0.03 log-lik short of R's optimum in flat likelihood valleys (Nile ANN
  alpha 0.248 vs R's 0.2455); at `1e-10` the Nile ANN optimum matches R to
  4 decimal places. Roughly doubles per-fit time (a full 15-candidate ZZZ
  selection on AirPassengers remains ~3 s).
- **timeseries: ETS `"ZZZ"` selection parity vs R after the alignment —
  outcome documented.** Candidate sets and parameter spaces now match R
  exactly, but 6 of the 10 reference selections still differ (AirPassengers,
  co2, lynx and variants), for a new, better-understood reason: R's
  Nelder-Mead frequently stalls far short of the optimum on seasonal
  candidates (up to 371 log-lik units on co2 MNM; 47 on AirPassengers AAA)
  while PyStatistics' L-BFGS-B does not, so near-tied tables resolve
  differently. In every divergent case the PyStatistics selection has a
  *better* value of R's own criterion than R's selection (new regression
  test `test_divergent_selection_dominates_r_choice`). Known remaining
  engine weakness, pre-existing and disclosed: on co2's trended
  non-seasonal candidates (A/M,A/Ad,N) our optimiser stalls ~120-150
  log-lik short of R's optimum (does not affect any reference selection —
  those candidates trail the seasonal ones by >1000 AICc).
  `_ets_select.py`'s parity documentation rewritten accordingly.
- **timeseries: `ETSParams`/`ETSSolution` moved to new module
  `timeseries/_ets_result.py`** (`_ets_fit.py` was at the 500-LoC hard
  limit after the parameter-space work). Pure move; both classes remain
  importable from `pystatistics.timeseries` and
  `pystatistics.timeseries._ets_fit` (re-export), so no public import
  changes.
- **timeseries: benign `RuntimeWarning: overflow encountered in exp` from the
  ETS optimiser silenced (no numeric change).** `_inv_logit` in
  `timeseries/_ets_fit.py` computed `np.exp(-z)` on unclipped logit-scale
  parameters, so L-BFGS-B line-search probes at large negative `z` overflowed
  to `inf` — mathematically the sigmoid correctly saturates to the lower
  bound, but the warning fired on every affected fit. The logit argument is
  now clipped to [-500, 500] before exponentiation; beyond that range the
  sigmoid is saturated far below one ulp, so all fitted results are
  bit-identical (verified: identical alpha/log-likelihood/fitted-value sums
  on all 10 `ets_r_reference.json` selection cases pre/post change, and the
  full timeseries suite fits with `RuntimeWarning` promoted to error).
