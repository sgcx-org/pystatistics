# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **Performance: Cox PH (`coxph`) is now O(n log n) instead of O(n²).** Rewrote
  the three hot paths in `survival/_cox.py` with no change to results
  (coefficients, hazard ratios, standard errors, z-values, p-values, partial
  log-likelihood, and concordance are bit-for-bit unchanged; all R-verified
  survival tests pass):
  - `_partial_loglik` and `_score_and_information` previously rebuilt the risk
    mask `time >= t_j` and re-sliced the design for every unique event time, on
    every Newton iteration (~O(n_events·n) per iteration). They now compute the
    risk-set sums S0/S1/S2 with a single reverse cumulative sum over time-sorted
    data, indexed per event time via `searchsorted` — one O(n) sweep per
    iteration. Efron's tie correction is vectorized for single-death times
    (the common case) with a bounded inner loop only over genuine multi-death
    ties.
  - `_concordance` (Harrell's C) was an O(n²) all-pairs Python double loop; it is
    now an O(n log n) Fenwick-tree count, reproducing the existing definition
    exactly.
  - Measured (3-covariate proportional-hazards data): a full `coxph` fit
    including concordance drops from ~10.0 s to ~31 ms at n=8000 (~318×) and from
    ~38 s to ~66 ms at n=16000 (~577×); wall-clock now grows ~linearly with n,
    competitive with R's `survival::coxph`. Memory is O(n·p²) (a transient
    (n, p, p) cumulative-sum array), which is negligible for the small p typical
    of Cox models.

- **GPU GLM (`regression.fit(..., backend='gpu')`) no longer falsely rejects
  correct float32 fits.** The unpenalized float32 IRLS path held the strict
  float64 convergence tolerance and declared non-convergence — failing loud —
  whenever the deviance plateaued at the float32 round-off floor, even when the
  fit was correct and well-conditioned. This misfired most on Apple Silicon
  (MPS, noisier float32) and made `survival.discrete_time(backend='gpu')`
  intermittently unusable at moderate n. Acceptance is now decided by the
  **relative Newton decrement** at the final coefficients (a stationarity test,
  evaluated in float64): a float32 fit that has reached a genuine optimum at the
  float32 floor is accepted and matches the CPU fit to the float32 tier; a
  non-stationary fit, or one whose float32 inner solve breaks down, still fails
  loud. No silent CPU fallback or precision downgrade — the error names the
  explicit options (`backend='cpu'`, `backend='gpu_fp64'`, ridge, or
  `force=True`). Verified on MPS and CUDA across binomial/Poisson/Gamma.

- **`survival.discrete_time` now accepts `backend='gpu_fp64'`** (CUDA-only
  exact double-precision GPU path), in addition to `'cpu'`/`'gpu'`/`'auto'`.
  Forwarded to the person-period logistic regression; on Apple Silicon it raises
  the canonical CUDA-required error (no silent fallback).

- **`KMSolution` gains the uniform `.standard_errors` and `.conf_int`
  accessors** (additive). `.standard_errors` is the Greenwood SE (alias of the
  existing `.se`); `.conf_int` is the `(m, 2)` `[lower, upper]` survival band
  (the column-stacked form of the existing `.ci_lower` / `.ci_upper`). The legacy
  accessors remain. This brings Kaplan-Meier in line with the library-wide
  result-accessor names without removing anything.

- **New `NotImplementedFeatureError` exception** for recognized-but-unimplemented
  features. It subclasses both `PyStatisticsError` (so `except PyStatisticsError`
  catches it) and the builtin `NotImplementedError` (so existing
  `except NotImplementedError` keeps working) — the same dual-inheritance pattern
  as `ValidationError`/`ValueError`. Stratified Kaplan-Meier and stratified Cox PH
  now raise it instead of a bare `NotImplementedError`, routing every survival
  error through the `core.exceptions` hierarchy.

- **Constitution (CONVENTIONS.md): added amendment A6** — "PyStatistics does not
  'help'." An explicit request is honored exactly or fails loud with the
  reason; it is never silently substituted with a different device, precision,
  algorithm, or estimator. The rule has two symmetric halves: do not silently
  help (no silent CPU fallback), and do not falsely refuse (a correct float32
  fit at the float32 floor must be accepted — correctness checks are calibrated
  to the requested precision). The GPU GLM acceptance fix above resolves a
  latent violation of the second half.
