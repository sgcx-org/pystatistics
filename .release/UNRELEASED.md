# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

### 4.0 — the consistency release (in progress)

A library-wide interface-consistency pass. No statistical/numerical behavior
changes; the math is unchanged. Public API names and backend handling are being
unified across every module under a single constitution
(`pystatistics/CONVENTIONS.md`). This release contains breaking interface changes
and removes deprecated spellings outright (no shim).

Landed so far:

- **New: `pystatistics/CONVENTIONS.md`** — the binding API constitution
  (semantic-consistency rule "one name, one meaning", the backend/precision
  convention, the selector taxonomy `backend`/`method`/`solver`/`link`/`family`,
  the result-object and exception conventions, and the full migration table).
- **New: `pystatistics.core.compute.backend`** — a single canonical resolver
  (`resolve_backend`) for the public `backend=` argument. It maps the backend
  string to a concrete `(device, precision)` target with one phrasing per
  failure mode (unknown backend → `ValidationError`; GPU unavailable →
  `RuntimeError`; `gpu_fp64` on MPS → the canonical "requires CUDA" message).
  Every module will route `backend=` through it so semantics are identical
  library-wide.
- **regression: `backend=` no longer encodes the algorithm.** The
  algorithm-suffixed strings `'cpu_qr'`, `'cpu_svd'`, `'gpu_qr'` are removed.
  `backend=` is now strictly `(device, precision)` — `'cpu'`/`'gpu'`/`'gpu_fp64'`/`'auto'`
  — and the numerical routine is selected with a new `solver=` argument
  (`'qr'` (default) or `'svd'`) on the linear-model path. `solver=` is rejected
  for GLMs and on the GPU backend (which uses Cholesky on the normal equations).
- **regression** now resolves `backend=` through the shared resolver, so an
  unknown backend raises `ValidationError` (was bare `ValueError`) and the GPU
  float64 / MPS messaging matches the rest of the library.
- **descriptive, hypothesis, montecarlo**: now resolve `backend=` through the
  shared resolver. These have no GPU float64 path, so `backend='gpu_fp64'` is
  rejected with a clear message (the "honest subset" — they expose only
  `{'cpu','gpu','auto'}`). Unknown backends now raise `ValidationError`
  uniformly. `backend='auto'` no longer selects Apple-Silicon MPS (matching the
  library-wide policy that `auto` picks a GPU only on CUDA); hypothesis keeps
  its documented exception that `auto` stays on CPU (its GPU only accelerates
  Monte-Carlo p-values).
- **`use_fp64` is removed; GPU float64 is now `backend='gpu_fp64'`.** The
  separate `use_fp64=` keyword is gone from every public fit function
  (multivariate `pca`, multinomial `multinom`, ordinal `polr`, `mice`, `gam`,
  timeseries `arima`/`arima_batch`). To run the GPU in double precision, pass
  `backend='gpu_fp64'` (CUDA only — raises a clear "requires CUDA" error on
  Apple-Silicon MPS). Precision now lives entirely in the backend string:
  `cpu`=float64, `gpu`=float32, `gpu_fp64`=float64. This is a breaking change.
- **mvnmle `mlest` gains `backend='gpu_fp64'`** (CUDA float64 direct MLE); its
  `backend='cpu-reference'` and `algorithm=` options are unchanged in this
  release. All seven GPU-float64-capable modules now resolve `backend=` through
  the shared resolver, so `gpu_fp64`-on-MPS, GPU-unavailable, and unknown-backend
  errors are identical everywhere.
- **`backend='auto'` no longer selects Apple-Silicon MPS in any module**
  (previously a few modules' `auto` could). `auto` picks a GPU only on CUDA,
  else CPU — the library-wide policy. MPS is still available via an explicit
  `backend='gpu'`.

#### Parameter renames (breaking)

Public parameters are unified under the constitution's naming law. Old names are
removed (no alias). Statistical behavior is unchanged.

- **mice**: `maxit` → `max_iter`, `m` → `n_imputations` (also on the result:
  `MICESolution.maxit`/`.m` → `.max_iter`/`.n_imputations`; `PooledResult.m`
  → `.n_imputations`).
- **ordinal `polr`**: `method` → `link` (the link function), `ridge` → `l2`,
  `level_names` → `category_names`.
- **multinomial `multinom`**: `class_names` → `category_names`.
- **multivariate `pca`**: `method` → `solver` (numerical routine; `method` is
  reserved for statistical choices).
- **mvnmle `mlest`**: `algorithm` → `method` (estimation method), inner
  `method` → `solver` (numerical optimizer). `mvnmle.little_mcar_test`:
  `algorithm` → `method`.
- **anova `levene_test`**: `center` → `location`.
- **hypothesis `t_test`**: `mu` → `pop_mean`, `var_equal` → `equal_var`.
  `wilcox_test`: `mu` → `null_value`. The `alternative` value `"two.sided"` →
  `"two-sided"` everywhere (hypothesis, montecarlo permutation, timeseries
  stationarity).
- **descriptive** (`cor`/`describe`/`cov`/`var`/`quantile`/`summary`): `use` →
  `na_action`, with values `"complete.obs"`/`"pairwise.complete.obs"` →
  `"complete"`/`"pairwise"`. `quantile`'s `type` → `quantile_type` (no longer
  shadows the builtin, and matches `describe`).
- **montecarlo `boot`**: `R` → `n_resamples`, `sim` → `method`, `stype` →
  `statistic_type` with descriptive values `"index"`/`"frequency"`/`"weight"`
  (were `"i"`/`"f"`/`"w"`). `boot_ci`: `conf` → `conf_level`, `type` →
  `ci_type`. `permutation_test`: `R` → `n_resamples`.

#### Constitution amendments

`CONVENTIONS.md` gained an **Amendments** section (the document is now
self-amending): **A1** — option *values* obey the naming law (descriptive
lowercase words, no dots, no single-letter codes); **A2** — a test's null value
is named for the quantity it constrains (`pop_mean` for a population mean, else
`null_value`); **A3** — the Wald statistic accessor is `.t_values` (t-reference:
linear models, LMM) or `.z_values` (normal reference: GLM, GLMM, ordinal,
multinomial, Cox), never `.test_statistics`/`.z_statistics`/`.t_statistics`.

#### Result/Solution object consistency (breaking)

- **Every public fit result is now a `…Solution`.** Renamed: `PCAResult`→
  `PCASolution`, `FactorResult`→`FactorSolution`, `ARIMAResult`→`ARIMASolution`,
  `ETSResult`→`ETSSolution`, `ACFResult`→`ACFSolution`, `StationarityResult`→
  `StationaritySolution`, `ARMABatchResult`→`ARMABatchSolution`,
  `AutoARIMAResult`→`AutoARIMASolution`, `DecompositionResult`→
  `DecompositionSolution`, `PooledResult`→`PooledSolution`, `MCARTestResult`→
  `MCARTestSolution`.
- **Wald statistic accessor unified** (amendment A3): linear models'
  `.t_statistics`→`.t_values`; GLM's `.test_statistics`→`.z_values`; Cox /
  discrete-time `.z_statistics`→`.z_values`.
- **Mixed models**: `.se`→`.standard_errors` (LMM and GLMM).
- **Result accessors aligned with their renamed parameters**:
  `OrdinalSolution.method`→`.link`, `MultinomialSolution.class_names`→
  `.category_names`, `LeveneSolution.center`→`.location`.
- **multivariate `PCASolution`/`FactorSolution` now follow the standard
  "Solution wraps `Result[Params]`" envelope** used by the rest of the library.
  The computed data fields moved into new frozen `PCAParams`/`FactorParams`
  payloads (`pystatistics/multivariate/_common.py`); each Solution wraps a
  `core.result.Result` and exposes the same public attributes via properties.
  No public attribute/method changed (`.sdev`, `.rotation`, `.x`, `.device`,
  `.to_numpy()`, `.to()`, `.explained_variance_ratio`,
  `.cumulative_variance_ratio`, `.summary()`, and all factor fields are
  preserved). New on both: `.info`, `.timing`, `.backend_name`, `.warnings`
  metadata accessors and an HTML repr (`_repr_html_`). `backend_name` is
  `"cpu_svd"` (CPU PCA), `"gpu_pca (cuda)"` (GPU PCA), and `"cpu_factanal"`
  (factor analysis). No numerical behavior change.
- **timeseries result objects now follow the standard "Solution wraps
  `Result[Params]`" envelope** used by the rest of the library. The computed
  data fields moved into new frozen `*Params` payloads — `ARIMAParams`
  (`pystatistics/timeseries/_arima_solution.py`, a new module so `_arima_fit.py`
  stays under the line limit), `ETSParams`, `ACFParams`, `StationarityParams`,
  `ARMABatchParams`, `AutoARIMAParams`, `DecompositionParams`. Each of
  `ARIMASolution`, `ETSSolution`, `ACFSolution`, `StationaritySolution`,
  `ARMABatchSolution`, `AutoARIMASolution`, `DecompositionSolution` now wraps a
  `core.result.Result` and exposes the same public attributes via properties.
  No public attribute/method changed (e.g. `ARIMASolution.n_params`,
  `.summary()`, `._collect_coef_names_values`; `ETSSolution.summary()`; the
  ACF/stationarity/decomposition `.summary()` methods are all preserved). New on
  every class: `.info`, `.timing`, `.backend_name`, `.warnings` metadata
  accessors and an HTML repr (`_repr_html_`). `backend_name` is `"cpu"` for the
  CPU paths and the device-tagged string for the GPU Whittle paths
  (`"whittle_gpu (cuda, fp32)"`, `"whittle_batch_gpu (cuda, fp64)"`, etc.). The
  new `*Params` classes are exported from `pystatistics.timeseries`. No
  numerical behavior change.
- **Notebook HTML repr is now uniform library-wide.** A single shared mixin
  `SolutionReprMixin` (`pystatistics/core/result.py`) provides `_repr_html_`,
  which renders the object's `summary()` inside a `<pre>` block. Every public
  `*Solution` with a `summary()` now inherits it, so Jupyter rendering is
  identical across modules with no per-class duplication. The previously inline
  `_repr_html_` methods on the multivariate (`PCASolution`, `FactorSolution`)
  and timeseries (`ACFSolution`, `StationaritySolution`, `DecompositionSolution`,
  `AutoARIMASolution`, `ETSSolution`, `ARIMASolution`) solutions were removed in
  favor of the shared implementation. `PooledSolution` and `ARMABatchSolution`
  are unaffected (no `summary()`). No behavior change to the rendered output.
- **`PooledSolution` (`pool()`) and `ARMABatchSolution` (`arima_batch()`) gained
  a `summary()`** and now also inherit `SolutionReprMixin`, so every public
  Solution has both `summary()` and the HTML repr.

#### Exceptions & validation (breaking)

- **`ValidationError` now subclasses both `PyStatisticsError` and the builtin
  `ValueError`** (amendment A4). `except PyStatisticsError` catches all library
  errors; `except ValueError` (the numpy/scipy habit) still catches every
  validation failure — an invalid argument genuinely *is* a value error.
- **No module raises a bare `ValueError` anymore.** Every input-validation
  `raise ValueError(...)` across the library became `raise ValidationError(...)`
  — bringing `mixed`, `survival`, `montecarlo`, `anova` onto the core exception
  hierarchy and tidying stragglers in `regression`, `mvnmle`, `mice`, `core`
  (incl. the shared `core.validation` helpers), `hypothesis`, `gam`, and
  `timeseries`. Existing `except ValueError` code keeps working (subclassing).
  The internal PIRLS failure in `mixed` now raises `NumericalError` (was
  `RuntimeError`); GPU-unavailable correctly stays `RuntimeError`.
- Amendment **A4** records the exception taxonomy and the `ValueError` subclassing.

#### Documentation consolidation

- **`pystatistics/GPU_BACKEND_CONVENTION.md` folded into `CONVENTIONS.md` and
  removed.** Its unique material — the "when to add a GPU backend (and when not
  to)" algorithmic-fit guidance and the GPU-backend testing checklist — now lives
  in the backend/precision section of `CONVENTIONS.md`, the single binding doc.
  The in-code reference in `pystatistics/ordinal/_solver.py` was repointed to
  `CONVENTIONS.md`.
- **`docs/` GPU notes consolidated into one non-binding `docs/GPU_NOTES.md`.**
  Merged `docs/GPU_BACKEND_NOTES.md` (CUDA-vs-MPS hard-won knowledge) with the
  durable MPS small-n dispatch-floor findings from `docs/MPS_LINALG_TODO.md`;
  both originals plus the disposable `docs/MICE_GPU_WIP.md` working list were
  deleted. A header notes the file is non-binding; the binding rules live in
  `CONVENTIONS.md`.
- **Version stamps refreshed** in `docs/DESIGN.md` and `docs/ROADMAP.md` (now
  4.0 / June 2026) with a one-line pointer noting 4.0 is the API-consistency
  release governed by `CONVENTIONS.md`.

#### CUDA verification

The full suite was run on real CUDA hardware (Forge, RTX 5070 Ti / Blackwell
sm_120, torch cu128): **2802 passed, 0 failed** — exercising every
`backend='gpu_fp64'` path and the GPU backend-name strings that MPS/CPU cannot
reach. The run caught one straggler an MPS-only machine could never see: a
CUDA-gated test in `tests/core/test_datasource.py` still called
`pca(..., method=)` (the pre-4.0 spelling, now `solver=`); fixed.
