# PyStatistics Conventions (the Constitution)

This document is the single source of truth for the PyStatistics **public API**:
how parameters are named, what they mean, how backends and precision are
selected, what fit functions return, and which errors they raise.

It is **binding**. New modules and new code conform to it. When this document
and any other doc, habit, or upstream reference (including R) disagree, **this
document wins**. It supersedes and absorbs the former `GPU_BACKEND_CONVENTION.md`.

Authors of new modules: read this first, then mirror the reference modules
(`regression`, `multivariate`, `multinomial`).

---

## The Prime Directive — why PyStatistics exists

This is the reason the library exists. It **predates and governs every rule
below**, including §0 and the numbered/lettered rules. When any rule, habit, or
convenience is in tension with this directive, the directive wins. The other
rules are the *means*; this is the *end*.

> **A working statistician can reach for PyStatistics _instead of R_ and receive a
> clear, justified deal — equivalent statistical results, stronger performance where
> the hardware allows, explicit and justified limitations, and never a hidden change
> to the question they asked.**

(This is the honest, precise form of the original *"give up nothing."* A literal "give
up **nothing**" is not an engineering law — every serious library makes tradeoffs; a
*justified, transparent* deal is what we actually promise, and PyStatistics already asks
users to accept transparent tradeoffs: condition gates, fail-loud refusals, explicitly
labelled randomized solvers, GPU optionality.)

Operationally the directive is **three guarantees, in strict priority order** — the same
three standing obligations below, named and ranked:

1. **Correctness.** Never return a result outside the documented correctness contract; if
   correctness cannot be guaranteed, fail loudly rather than return a result.
2. **Fidelity.** Never satisfy Correctness by silently solving a *different* problem — no
   silent change of estimator, backend, precision, solver, or approximation (A6).
3. **Performance.** Subject to Correctness and Fidelity, make every reasonable effort to
   match or exceed the reference on **equivalent hardware** using the **same statistical
   estimator**.

(The validation program states these verbatim — see `RIGOR.md` → the three foundational
guarantees.)

Two tests decide whether we are meeting it:

1. **Methods-section credibility.** A researcher can write "analysis performed in
   PyStatistics" in a paper and not feel they need to apologize for, hedge, or
   hide it. The numbers withstand a reviewer who checks them against R.
2. **No regret.** Having used PyStatistics, the statistician never has cause to
   think *"I would have been better off had I just used R."* Not in correctness,
   not in speed, not in the credibility of the result.

This directive has three standing obligations:

- **Correctness parity (non-negotiable).** Results match a trusted reference (R,
  and where relevant scipy) to the appropriate numerical tolerance. This is the
  subject of §0 below and of the validation program.
- **Performance: on the CPU path, speed parity with R is a _requirement_, not a
  goal.** The CPU path is where PyStatistics stands in as the R/SAS replacement in
  Python, so a statistician must never wait longer for the same answer than R
  would take. If a CPU path is slower than R, we make every effort to close the
  gap; "slower than R on CPU" is a **defect**, tracked as a regression by the
  validation suite (`ratio_pystat_over_r` / `speedup_vs_r`), never reported as
  neutral information. Faster than R is a welcome bonus; **parity is the line.**
  - **There is no escape hatch on the CPU path.** We do not get to hand the user a
    tradeoff ("it's slower, but…") — because R hands them none. (Contrast the GPU
    path, which is *itself* a compensating benefit R cannot offer; there a faster
    method with a **loud** escape hatch is acceptable — e.g. OLS/GLM on GPU default
    to Cholesky and route genuine ill-conditioning to ridge / QR / `force=True` /
    `gpu_fp64`. That hatch is a GPU affordance, not a CPU one.)
- **Coherence & trust.** The library reads as one coherent work and fails loud
  rather than returning a quietly-wrong answer (the naming law, the result/error
  conventions, and A6). A statistician trusts a tool that is consistent and that
  refuses rather than misleads.

**How a CPU speed gap is closed — in this order.** When a CPU path is slower than
R, we investigate and exhaust these, in order, before reaching for the next:

1. **Fix the algorithm** — choose a lighter but equally-sound method (e.g.
   non-pivoted Householder QR with the factor applied implicitly, rather than
   full column-pivoting QR). Note: optimize the *sound* method; do not downgrade
   to a weaker one (QR is the least-squares default precisely because it does not
   square the condition number — genuine ill-conditioning is answered by a
   better-posed *estimator* like ridge, not a less-stable *solver*).
2. **Exploit mathematical structure** — sparsity, symmetry, the problem's special
   form.
3. **Vectorize** — remove Python-level loops; one BLAS call instead of many.
4. **Use optimized BLAS/LAPACK** — call the right compiled primitive; avoid
   materializing temporaries.
5. **Improve the implementation** — allocation, memory traffic, dtype, buffer
   reuse.
6. **Only then**, if the gap remains because the underlying native routine itself
   is the bottleneck — R wins only because it ships a compiled routine we lack —
   **write our own native implementation** (Cython/C), reimplemented **clean-room
   from the algorithm**. R was chosen as the reference partly because its sources
   are inspectable: read them to understand the *math*, then write it fresh. Never
   transliterate R's (GPL) source line-for-line — that would make our code a GPL
   derivative, and is a craftsmanship failure besides.

Reaching step 6 is neither a failure nor optional: if a native implementation is
the only thing that brings the CPU path to parity, that is what we build. *"That
is a lot of engineering effort"* is **never** a reason to leave a CPU path slower
than R. The earlier steps exist so we don't reach for C prematurely — not so we
can stop short of parity. (Switching the *method or parameterization itself* —
as opposed to optimizing it — is justified only by necessity, not convenience:
MVNMLE adopted the forward-Cholesky parameterization because it was *required* to
make the estimator viable on a GPU, not merely because it was faster.)

Every numbered rule that follows exists to earn one of these three. If a rule
ever works against the directive, fix the rule.

---

## 0. Governing philosophy

PyStatistics guarantees **statistical and numerical parity with R** (and, where
relevant, scipy) — its results are correct against a trusted reference. It does
**not** guarantee, or aim for, **API parity**. R is a reference implementation
for the *math*, not a constitution for the *interface*.

The goal is **not** to make R users feel at home. The goal is for PyStatistics
to feel like one coherent library written by one author, that happens to produce
R-equivalent numbers. The API should read as PyStatistics — not as a collection
of R package dialects translated into Python.

R argument names are prior art, not authority:

1. A concept defined library-wide uses its **constitutional name** everywhere.
2. If R happens to use the same name and it does not conflict, keeping it is fine.
3. If R uses a different name for a concept PyStatistics has standardized, the
   PyStatistics name wins — rename it.
4. A name genuinely local to one procedure may keep its conventional spelling
   **only if** it is clear and does not collide (see Rule S0).

---

## Naming law

### S0 — Semantic Consistency (the primary rule)

**One name, one meaning.** If two parameters share a name, they mean the same
thing. If two parameters mean the same thing, they share a name. A name binds to
a concept **library-wide** — never "this spelling, locally redefined."

S0 outranks every other rule. A name can be constitutional, scipy-aligned, and
R-conventional and still be **illegal** if it collides semantically with the same
name used elsewhere. When a collision exists, at least one side must be renamed
so the shared name keeps exactly one meaning.

Consequences baked into this document: `backend=` is *always* an execution
target, never an algorithm; `method=` is *always* a statistical choice, never a
numerical or device choice; `center` is the data-centering flag, so Levene's
location estimator is `location`, not `center`.

### S1 — Descriptive `snake_case`; no single-letter public parameters

Public parameters are descriptive `snake_case`. Single-letter parameters are
banned (`m`, `R` are out). **Sole exception:** the canonical data arrays `x`,
`y`, and the design matrix `X`, which are universally conventional.

### S2 — No dotted string values

String-valued options use no `.` separators (R lexical artifact). Use a hyphen
or a plain word: `"two-sided"`, not `"two.sided"`; `"complete"`, not
`"complete.obs"`.

### S3 — Never shadow Python builtins

No parameter may be named `type`, `id`, `input`, `list`, etc. Use a qualified
name (`ci_type`).

### S4 — One constitutional name per library-wide concept

See the registry below. These names are reserved and mean exactly one thing.

### S5 — `method` vs `solver` vs `link` are distinct

The selector taxonomy (below) assigns each kind of choice its own name. `method`
is the **statistical** choice; `solver` is the **numerical** routine; `link` is
the link function; `backend` is the execution target; `family` is the
distribution.

### S6 — Tie-breaker: prefer the Python ecosystem

When de-R-ifying a name that has no PyStatistics prior art and no constitutional
entry, prefer the dominant Python spelling (scipy/numpy/sklearn) over R. S6 is
only a tie-breaker; it never overrides S0–S4. (Example: `equal_var` follows
scipy. Counter-example: the one-sample null mean is `pop_mean`, not scipy's
`popmean`, because S1 snake_case outranks the S6 tie-break.)

---

## Selector taxonomy (each name = exactly one concept)

| Name | Means | Never means |
|---|---|---|
| `backend` | execution target: device **and** precision | an algorithm |
| `family` | error / response distribution | anything else |
| `link` | link function (a model specification) | a numerical routine |
| `method` | the **statistical** choice — which estimator / statistic / principle (changes the statistical meaning of the result) | a numerical or device choice |
| `solver` | the **numerical** routine producing the same statistical result (differs only in speed / stability) | a statistical choice |
| `na_action` | missing-data policy | anything else |

A user who learns this table can predict every module: `method` = which
statistic, `solver` = which numerical routine, `backend` = where it runs,
`link` / `family` = the model.

---

## Constitutional concept registry

| Concept | Canonical name | Notes |
|---|---|---|
| convergence tolerance | `tol` | per-procedure default allowed |
| max iterations | `max_iter` | |
| compute backend | `backend` | values: `cpu` · `gpu` · `gpu_fp64` · `auto` |
| numerical escape hatch | `force` | "proceed despite a numerical-safety guard" |
| predictor / column labels | `names` | |
| response-category labels | `category_names` | replaces `level_names` / `class_names` |
| GLM family | `family` | |
| link function | `link` | when exposed outside `family` |
| prior / observation weights | `weights` | per-observation weight vector `(n,)`; the model's prior weights (WLS weights for OLS, IRLS prior weights for GLM). Never a coefficient vector or a resampling probability. |
| linear-predictor offset | `offset` | additive term in the linear predictor, η = Xβ + `offset`; a per-observation float vector `(n,)`, **not estimated**. Reserved library-wide: the token `offset` **never** means an array index, position cursor, or displacement — use `start` / `*_start` / `base_*` for those. |
| L2 / ridge penalty | `l2` | |
| RNG seed | `seed` | |
| confidence level | `conf_level` | |
| count of stochastic replicates | `n_<thing>` | `n_resamples`, `n_imputations` |
| statistical variant | `method` | see taxonomy |
| numerical routine | `solver` | see taxonomy |
| missing-data policy | `na_action` | values: `everything` · `complete` · `pairwise` |
| test alternative | `alternative` | values: `two-sided` · `less` · `greater` |

Anything not listed and genuinely local-and-clear (`order`, `seasonal`,
`include_mean`, `center`, `scale`, `paired`, `strata`, `ties`, `reml`,
`ss_type`, `quantile_type`, `correction`, `conf_type`, …) keeps its conventional
name.

**Reserved names govern _our_ identifiers, not external libraries' parameter
spellings.** A reserved concept name (e.g. `offset`) binds every identifier we
author. It does **not** reach into a third-party API we merely call: passing
`torch.tril_indices(n, n, offset=-1)` uses PyTorch's diagonal-`offset` argument,
which is upstream's name, not ours — those keep their original spelling.

---

## Backend & precision convention

`backend=` jointly encodes **device and precision**. The combinations are a
small, closed set — invalid states (CPU-fp32, MPS-fp64) are simply
unrepresentable rather than guarded at runtime.

| `backend=` | Device | Precision | Role |
|---|---|---|---|
| `'cpu'` | CPU | float64 | the reference path (R-parity, regulated-industry default) |
| `'gpu'` | CUDA *or* MPS (auto) | float32 | the speed default |
| `'gpu_fp64'` | **CUDA only** | float64 | correctness path; numerically exact (~1e-15 vs CPU) |
| `'auto'` | CUDA-fp32 if present, else CPU | — | never auto-selects MPS |

Rules:

- **Default resolution.** `backend=None` resolves from input: a numpy array / CPU
  source → `'cpu'`; a `torch.Tensor` already on a GPU → `'gpu'`. numpy defaults
  to `cpu` because "unspecified" must mean the reference path.
- **No public device strings.** `'cuda'` / `'mps'` are internal `device_type`
  values, never public `backend=` values. The device under `'gpu'` is
  auto-resolved.
- **`gpu_fp64` is CUDA-only.** On MPS it raises (Metal has no float64) with the
  one canonical message:

  > `backend='gpu_fp64' requires CUDA: Apple Silicon (Metal/MPS) has no float64. Use backend='gpu' (float32) on Apple Silicon, or backend='cpu' for a double-precision fit.`

- **Precision is in the string, not a flag.** There is **no** `use_fp64`
  parameter. (Removed in 4.0.)
- **Honest subsets.** A module exposes only the `backend` values it can honor.
  Modules with no algorithmically warranted float64 GPU path
  (`hypothesis`, `descriptive`, `montecarlo`) list `{'cpu','gpu','auto'}` and
  raise on `'gpu_fp64'`. A module with no GPU path exposes no `backend=` at all.
- **When to add a GPU backend at all.** Only when the computation maps to a GPU
  win (large dense linear algebra, big-N likelihoods, batched fits,
  frequency-domain transforms) — measured CPU time > ~100 ms with > 80% in a
  cuBLAS/cuSOLVER/cuFFT-mappable kernel. Do not add a GPU backend just because a
  module is public; offering a slower path that can't be honored violates S0 and
  fail-fast.
- **FP32 tolerance floor.** When running float32, floor `tol` at 1e-5 (FP32
  gradient precision is ~1e-7; tighter tolerances stall on the noise floor).
- **Test tolerances** against CPU use the published `GPU_FP32` tier from
  `core.compute.tolerances`, not ad-hoc numbers.

### When to add a GPU backend — and when not to

A GPU backend exists because the underlying computation is a good fit for GPU
hardware, **not** because a module is public and therefore "deserves" one.
Adding a backend that doesn't make algorithmic sense is worse than having none:
it invites users onto a slower path, adds a maintenance surface, and forces
convention compliance for no benefit.

**Reasonable targets:** large dense linear algebra (QR / SVD / Cholesky),
per-iteration gradient evaluations on large design matrices, big-N likelihood
evaluations, batched independent fits, frequency-domain transforms.

**Not reasonable targets — do not add a GPU backend for these:**

- Many-small-fits workflows where per-call launch overhead dominates compute
  (e.g. ANOVA SS, which fits a chain of tiny regressions — the CPU LAPACK call
  is microseconds, the GPU kernel launch is milliseconds).
- Inherently sequential algorithms without exploitable parallelism (most EM
  variants below a certain dataset size, Cox PH partial likelihood).
- Algorithms already bounded by PCIe / device-to-host transfer, where compute
  is cheap enough that copying costs more than the kernel.
- Algorithms where the existing scipy / LAPACK implementation is already within
  an order of magnitude of hardware peak — a 2× speedup is not worth the
  convention-compliance + test + review cost.

**Signal that GPU makes sense:** measured CPU time > ~100 ms on a representative
input, with > 80 % of that time in a compute kernel that maps to
cuBLAS / cuSOLVER / cuFFT. Below that threshold, launch + transfer overhead eats
the win.

If a public module has **no** `backends/gpu*.py`, that is a deliberate statement
— like `anova`, `timeseries.ets`, `survival.coxph`,
`multivariate.factor_analysis`, and the acf / stationarity families. The public
API correspondingly does not expose a `backend=` parameter, which is the correct
state: offering a `backend='gpu'` you can't honor violates S0 and fail-fast.

### Testing GPU backends

Every GPU backend ships a `Test<Module>GPU` class mirroring the reference
modules, covering at least:

- `test_invalid_backend_raises` (bad backend string → `ValidationError`);
- `test_gpu_unavailable_raises_explicitly` (monkeypatched `detect_gpu`,
  `backend='gpu'` must raise `RuntimeError`);
- `test_auto_backend_falls_back_to_cpu_when_no_gpu` (monkeypatched,
  `backend='auto'` must succeed on the CPU path);
- `test_gpu_fp64_matches_cpu_<property>` (CUDA only, skip on MPS);
- `test_gpu_fp32_matches_cpu_at_tier` (asserts the `GPU_FP32` tolerance tier
  from `core.compute.tolerances`, never ad-hoc numbers);
- `test_gpu_tensor_with_cpu_backend_raises` (explicit `backend='cpu'` on a GPU
  tensor must raise — no silent migration).

The FP32 tolerance floor (`tol` floored at 1e-5) applies to assertions too: a
fit that explicitly ran in float32 cannot be held to float64 convergence.

---

## Result objects

- **Suffix.** Every public fit returns a `…Solution` (not `…Result`).
- **Envelope.** Every Solution wraps the core `Result[Params]`, so
  `.backend_name`, `.timing`, `.warnings`, and `.info` exist library-wide
  (including timeseries and multivariate, which previously returned bare
  dataclasses).
- **Uniform accessors** (present whenever meaningful):

  | Concept | Accessor |
  |---|---|
  | coefficients (array) | `.coefficients` |
  | coefficients (labeled dict) | `.coef` |
  | standard errors | `.standard_errors` |
  | fitted values | `.fitted_values` |
  | residuals | `.residuals` (GLM-style variants: `.residuals_deviance`, `.residuals_pearson`, `.residuals_working`, `.residuals_response`) |
  | test statistics | `.z_values` (z-test) / `.t_values` (t-test) — never `.test_statistics` |
  | p-values | `.p_values` |
  | confidence interval | `.conf_int` |
  | convergence | `.converged`, `.n_iter` (on every iterative fit) |
  | backend used | `.backend_name` |

- **Display.** Every Solution implements `.summary()` (R-style printed report),
  `__repr__()`, and `_repr_html_()` (Jupyter).

---

## Errors & validation

- **Input validation** goes through `core.validation` helpers (`check_array`,
  `check_finite`, `check_2d`, `check_consistent_length`, …). No module rolls its
  own.
- **Exception types** come from `core.exceptions`, never bare
  `ValueError`/`RuntimeError`:

  | Condition | Exception | Canonical message shape |
  |---|---|---|
  | unknown / unsupported `backend` | `ValidationError` | `Unknown backend {value!r}. Valid options: …` |
  | GPU requested, none available | `RuntimeError` | `No GPU available (need CUDA or MPS). Use backend='cpu', or install PyTorch with CUDA/MPS support.` |
  | `gpu_fp64` on MPS | `RuntimeError` | the canonical CUDA-required message above |
  | iterative non-convergence | `ConvergenceError` | include `iterations`, `final_change`, `reason` |
  | invalid input (shape / non-finite / bad arg) | `ValidationError` (or `DimensionError`) | descriptive, fail-loud |

- **Fail loud, no silent defaults.** An explicit request that cannot be honored
  (e.g. `backend='cpu'` with a GPU tensor) raises; it never silently migrates.

---

## Determinism

- Stochastic procedures take `seed: int`. The RNG is constructed from it
  explicitly (`numpy.random.default_rng(seed)` on CPU; `torch.manual_seed` on
  GPU) — no global RNG state.
- GPU RNG is not bit-identical to CPU; replicates are statistically equivalent,
  not bitwise-equal. Document this per function.

---

## 4.0 migration table (old → new → reason)

| Old | Module | → 4.0 | Reason |
|---|---|---|---|
| `maxit` | mice | `max_iter` | S4: library-wide max-iterations name. |
| `m` | mice | `n_imputations` | S1: no single-letter params. |
| `R` | boot | `n_resamples` | S1: no single-letter params. |
| `sim` | boot | `method` | S5: selects the statistical bootstrap variant. |
| `stype` | boot | `statistic_type` (values `index`/`frequency`/`weight`) | S1: de-abbreviate cryptic name + values. |
| `use` | cor, describe | `na_action` (values `everything`/`complete`/`pairwise`) | S4 + S2: missing-data policy; de-dot values. |
| `mu` | t_test | `pop_mean` | S1 + S6 (snake_case over scipy `popmean`). |
| `var_equal` | t_test, var_test | `equal_var` | S6: scipy spelling. |
| `conf` | boot_ci | `conf_level` | S4: library-wide confidence level. |
| `type` | boot_ci | `ci_type` | S3: shadows a builtin. |
| `ridge` | polr | `l2` | S4: library-wide L2 penalty. |
| `method` (link) | polr | `link` | S0/S5: link function is a model spec, not a `method`. |
| `method` (svd/gram) | pca | `solver` | S0/S5: numerical routine, not a statistical choice. |
| `algorithm` | mvnmle | `method` | S5: estimation method (direct/em/monotone). |
| `method` (optimizer) | mvnmle | `solver` | S0/S5: inner numerical optimizer. |
| `center` (median/mean) | levene_test | `location` | S0: collides with pca `center: bool`. |
| `level_names` / `class_names` | ordinal / multinomial | `category_names` | S4: one name for response-category labels. |
| values `"two.sided"` | hypothesis, timeseries | `"two-sided"` | S2: de-dot. |
| `use_fp64=True` | mice, ordinal, multinomial, multivariate, mvnmle, gam, timeseries | `backend='gpu_fp64'` | precision lives in the backend string. |
| `…Result` | multivariate, timeseries | `…Solution` | uniform result suffix. |
| `.se` | mixed | `.standard_errors` | uniform accessor. |

These are breaking interface changes. They ship together as **4.0**, the
consistency release. No deprecation shim — the old names/flags are removed.
Statistical results are unchanged.

## 5.0 migration table (old → new → reason)

The single pre-launch consistency sweep (see the versioning policy below). Like
4.0: hard renames, no shims, statistical results unchanged.

| Old | Module | → 5.0 | Reason |
|---|---|---|---|
| `p` (expected proportions) | chisq_test | `expected_probs` | S0/S1: bare `p` named three concepts. |
| `rescale_p` | chisq_test | `rescale_probs` | follows `expected_probs`. |
| `B` | chisq_test, fisher_test | `n_resamples` | S1 + reserved `n_<thing>`. |
| `p` (null proportion) | prop_test | `null_value` | S0/S1 + A2 (generic null). |
| `n` | prop_test | `n_trials` | S1: no single-letter params. |
| `p` (p-value vector) | p_adjust | `p_values` | S0/S1. |
| `n` | p_adjust | `n_comparisons` | S1. |
| `ratio` | var_test | `null_value` | A2: generic null (variance ratio). |
| `dfcom` | pool | `df_complete` | S1: de-abbreviate. |
| `alpha` (0.05) | pool | `conf_level` (0.95) | S0 + reserved confidence-level name. |
| `i` | MICESolution.completed | `index` | S1. |
| `.se` | PooledSolution, GAMSolution | `.standard_errors` | uniform accessor. |
| `.ci_low` / `.ci_high` | PooledSolution | `.ci_lower` / `.ci_upper` (+ `.conf_int`) | uniform accessor. |
| `.R` | Bootstrap/PermutationSolution | `.n_resamples` | S1 accessor leftover (param already migrated in 4.0). |
| `.se` / `.ci` / `.sim` / `.ci_conf_level` | BootstrapSolution | `.standard_errors` / `.conf_int` / `.method` / `.conf_level` | uniform accessors. |
| values `perc` / `stud` | boot_ci `ci_type` | `percentile` / `studentized` | A1: descriptive option values. |
| `anova` | regression | `deviance_table` | S0: collided with the `anova` module. |
| `AnovaTable` | regression | `DevianceTable` | follows the function rename. |
| values `Chisq` / `LRT` / `F` | deviance_table `test=` | `chisq` / `lrt` / `f` | A1: lowercase option values. |
| values `negative.binomial` / `inverse.gaussian` | regression `family=` | `negative-binomial` / `inverse-gaussian` | S2/A1 (value + emitted `.family_name`). |
| `GammaFamily` / `.family_name='Gamma'` | regression | `Gamma` / `'gamma'` | S0 consistency with sibling families. |
| value `1/mu^2` | regression `link=` | `inverse-squared` | A1: descriptive option value. |
| `lam` | ridge | `l2` | S4: reserved L2-penalty name. |
| values `gg` / `hf` | anova_rm `correction=` | `greenhouse-geisser` / `huynh-feldt` | A1: descriptive option values. |
| `type` | decompose | `kind` | S3: shadows a builtin. |
| `.type` | ACF/DecompositionSolution | `.kind` | S3. |
| `h` | forecast_ets, forecast_arima | `n_ahead` | S1. |
| `alpha` (significance) | ndiffs | `significance_level` | S0: collided with the ETS smoothing `alpha`. |
| `allowdrift` | auto_arima | `allow_drift` | S1: snake_case. |
| `newxreg` | arima, forecast_arima | `new_xreg` | S1: snake_case. |
| values `CSS-ML`/`ML`/`CSS`/`Whittle` | arima, auto_arima `method=` | `css-ml`/`ml`/`css`/`whittle` | A1: lowercase option values. |
| `levels` (whole percents) | forecast_ets, forecast_arima | `conf_level` (fractions) | S0 + reserved confidence-level name. |
| `backend='cpu-reference'` | mvnmle | `solver='reference'` | scheduled removal; `backend` ≠ a `solver` choice. |
| bare `RuntimeError` | little_mcar_test | `ConvergenceError` / `NumericalError` | A4: error taxonomy. |
| `SmoothInfo.lambda_` / `.s_scale` | gam | `.lambdas` / `.s_scales` | scheduled removal (per-margin tuples). |
| `W` | grm_lmm | `random_factor` | S1: no single-letter matrix param. |
| `type` | ordinal/multinomial `predict()` | `kind` | S3: shadows a builtin. |
| value `logistic` | ordinal `link=` | `logit` | S0: one spelling for the logit link (A13). |
| `…Params` exports | 7 modules | removed from public API | internal payloads; the public surface is `…Solution` (A12). |
| bare `TypeError`/`ValueError`/`NotImplementedError` | regression, core, ordinal, mvnmle | `ValidationError`/`DimensionError`/`NotImplementedFeatureError` | A4: `core.exceptions` on public paths. |

Additive (non-breaking) companions shipped in the same cut: uniform envelope
accessors (`.backend_name`/`.timing`/`.warnings`) on the mixed/multinomial
Solutions; `.coef` on LMM/GLMM; `.category_names` on OrdinalSolution;
`.coefficients` alias on MultinomialSolution; `.coef`/`.conf_int` on GAMSolution;
`boot_ci` now fails loud on a multi-level `conf_level`.

### Versioning policy — what a major means, and when the next one happens

A **major** version means exactly one thing: **a breaking change to the public
API**. Size is irrelevant — semver majors are cheap, not ceremonial. Do not treat
"major" as "big coordinated event"; that framing is what lets deprecations rot and
breaking debt pile up until a mega-release (the pain 4.0 was paying down).

**`5.0` is the pre-launch consistency sweep** — cut **once `pystatistics` is
feature-complete and about to launch publicly**, not for any single deprecation.
Rationale: breaking changes are *free* before there are real users and downstream
consumers pinning us, and expensive forever after — so the one deliberate,
comprehensive break happens at the last free moment. The 5.0 cut cleared every
scheduled removal in the ROADMAP "Deprecations & scheduled removals" table and
applied the v1-regret renames an in-depth consistency pass surfaced; the complete
list is the **5.0 migration table above**. After 5.0, any new deprecation is
parked in that ROADMAP table with its removal version so "deprecated" never
silently becomes "eternal".

**Sequencing (locked, 2026-07-08):** `pystatistics` is finished *completely* — the
whole-library close-out plus the 5.0 pre-launch sweep — **before** any downstream
`pystats*` vertical (PyStatsBio, etc.) is validated. Downstream packages must not be
validated against an API that will still change under them; they pin and validate
against the settled post-5.0 surface.

**After launch:** real semver — a major whenever something breaks, however small,
batched only for convenience, never deferred for ceremony.

---

## Capability scope — deliberate carve-outs from R parity

The Prime Directive requires that where PyStatistics omits a capability its R
reference offers, the omission is **deliberate, documented, and carries a GOOD
reason** — absence by oversight is not permitted. Most such gaps are *closed by
implementation* (the library grows the capability). The items below are the
opposite decision: capabilities we have **examined and deliberately do not
implement**, each because it is genuinely specialist, interface-only (the math is
identical), or already covered by a better path we offer. Every one **fails loud**
when requested — none silently substitutes a different computation.

This list is exhaustive for the current surface: a capability an R user reaches
for that is *not* here and *not* on this list is a gap to close, not an accepted
carve-out.

- **gam — exotic smooth bases `re` / `ds` / `gp` / `fs`.** The mainstream mgcv
  bases are implemented (`tp` thin-plate, `cr` cubic-regression, `cc` cyclic,
  `ps` P-splines); the random-effect (`re`), Duchon-spline (`ds`), Gaussian-
  process (`gp`), and factor-smooth-interaction (`fs`) bases are specialist
  constructs a working smoother rarely reaches for, and `re` in particular is
  better served by the `mixed` module's explicit random effects. Any unknown
  `bs=` raises `ValidationError` (fail-loud proof holds).

- **regression — non-treatment factor contrasts (helmert / sum / poly).** Only
  treatment (dummy) coding is offered — R's own default. The choice of contrast
  coding is a **pure interface decision with no effect on the math**: the fitted
  values, deviance, predictions, and every estimator-invariant quantity are
  identical across coding schemes; only the individual coefficients' *meaning*
  (and hence their labels) change. A user needing a specific contrast can recode
  the design column directly. Because nothing statistical differs, this is a
  labeling convenience, not a capability gap.

- **regression — the fully-general `quasi(link, variance)` constructor.** The
  mainstream overdispersion families — `quasipoisson` and `quasibinomial` — are
  implemented as first-class families. R's fully-general `quasi()` constructor,
  which lets the user pair an *arbitrary* variance function with an arbitrary
  link, is a specialist tool for bespoke mean-variance relationships outside the
  standard exponential family; it fails loud rather than being silently absent.

- **survival — `exact` (exact partial-likelihood) ties.** Cox tie handling offers
  `efron` (the default, and R's recommended method) and `breslow`, which together
  cover standard biostatistical practice. The `exact` method — the exact marginal
  partial likelihood — is rarely needed (relevant only with heavy tie structure
  where Efron already approximates it well) and is computationally costly; it is
  refused loudly rather than approximated.

- **montecarlo — variance-stabilizing transforms (`h` / `hinv`), antithetic
  sampling, and stratified/blocked permutation.** All five bootstrap CI types
  (normal / basic / percentile / BCa / studentized) and both ordinary and
  stratified resampling are provided. The `h`/`hinv` monotone-transform hooks of
  `boot.ci` are an advanced variance-reduction refinement the caller can apply
  externally (transform the statistic, invert the interval); antithetic and
  blocked-permutation schemes are specialist variance-reduction devices outside
  the ordinary-and-stratified core. Each unsupported option fails loud.

- **multivariate — `prcomp`'s `tol` rank truncation.** PCA exposes
  `n_components=`, which selects the retained rank directly and is the equivalent
  (and more explicit) control; R's `tol=` — "drop components whose standard
  deviation is below `tol × sd[0]`" — is a different spelling of the same rank
  decision, reachable by inspecting the returned standard deviations and setting
  `n_components`. No statistical capability is missing.

---

## Amendments

This constitution is **self-amending**. When a naming or API question arises
that the rules above do not unambiguously answer, the resolution is recorded
here as a numbered amendment so that every future occurrence has a single,
binding answer rather than a fresh one-off judgement. Only a genuinely unique
situation (one that cannot recur) is exempt from being generalised into an
amendment. Amendments have the same force as the rules above.

### A1 — Option *values* obey the naming law, not just parameter names

S1 (descriptive `snake_case`, no single-letter identifiers) and S2 (no dotted
string values) apply to the **string values** an option accepts, not only to the
parameter name. Option values are descriptive lowercase words:

- No cryptic abbreviations or single-letter codes: `statistic_type` takes
  `'index'` / `'frequency'` / `'weight'`, never `'i'` / `'f'` / `'w'`.
- No dotted separators: `'two-sided'`, `'complete'`, not `'two.sided'`,
  `'complete.obs'`.
- Multi-word values use a hyphen (`'two-sided'`, `'log-log'`); prefer the
  dominant Python-ecosystem spelling on a tie (S6).

### A2 — Name a test's null value for the quantity it constrains

The value a hypothesis test compares the data against is named for *what that
parameter is*, never reusing the Greek-letter shorthand `mu`:

- `pop_mean` when it is specifically a **population mean** (e.g. one-sample
  `t_test`).
- `null_value` for a **generic or non-mean null** — a location shift
  (`wilcox_test`), a proportion, a ratio, etc. This matches the library-wide
  `.null_value` result accessor.

Two tests share `pop_mean` only if both constrain a population mean; otherwise
the generic case is `null_value`. (S0 holds: the names differ because the
quantities differ.)

### A3 — Wald statistic accessor: `.t_values` vs `.z_values`

The coefficient Wald statistic (estimate / standard error) on a result object
is exposed under exactly one of two names, chosen by the reference distribution
the fit uses for its p-values:

- `.t_values` when the reference is Student-t (finite-sample): linear models
  and mixed-model LMM.
- `.z_values` when the reference is normal / asymptotic: GLM, GLMM, ordinal,
  multinomial, and Cox proportional hazards.

`.test_statistics`, `.z_statistics`, `.t_statistics` are not used. A class
exposes only the one name matching its distribution; it never exposes both. GLM
standardises on `.z_values` for all families (the Wald statistic value is
identical regardless; the t-vs-z choice for gaussian/gamma p-values is handled
inside the p-value computation, not the statistic accessor).

### A4 — Exception types, and `ValidationError` is also a `ValueError`

Every error a module raises comes from the `core.exceptions` hierarchy (all
rooted at `PyStatisticsError`), never a bare builtin. The mapping:

- invalid argument / shape / non-finite input → `ValidationError` (or its
  subclass `DimensionError` for shape/dimension mismatches);
- iterative non-convergence → `ConvergenceError`;
- GPU requested but unavailable / `gpu_fp64` on MPS → `RuntimeError` (an
  environment failure, not a value error — this is the one builtin that stays).

`ValidationError` subclasses **both** `PyStatisticsError` and the builtin
`ValueError`. So `except PyStatisticsError` catches all library errors, and
`except ValueError` (the numpy/scipy habit) still catches every validation
failure. A module must not raise a bare `ValueError` for input validation —
raise `ValidationError`, which *is* a `ValueError`.

### A5 — Scope of the naming law: the public surface, plus reserved names everywhere

The naming law (S0–S6) and the 4.0 migration table govern the **public API
surface**: public parameter names, public result-accessor names, public option
*values*, and any string a user can see (including a `__repr__` label). Purely
internal identifiers — module-private dataclass fields in `_common.py`, solver
locals, backend-class attributes — are implementation details and are **not**
required to track the migration table. (This is why `BackendTarget.use_fp64`
is legal even though `use_fp64` is banned as a *public* parameter: precision
still needs an internal representation; only the public spelling is constrained.)

**Exception — reserved names are protected library-wide, because S0 is
absolute.** The taxonomy selectors (`backend`, `family`, `link`, `method`,
`solver`, `na_action`) and the reserved Wald-statistic accessors (`t_values`,
`z_values`, per A3) bind to exactly one concept *everywhere*, internal code
included. A private field must not repurpose a reserved name for a different
concept: an internal field named `method` that actually holds a *link*, or a
field named `t_values` that actually holds a Wald *z*-statistic, is a latent S0
collision and is renamed to the concept it holds. (Resolved in 4.0:
`OrdinalParams.method` → `link`; `GLMMParams.t_values` → `z_values`.
`LMMParams.t_values` is unchanged — it genuinely holds Student-t statistics.)

Non-reserved legacy spellings on private fields (e.g.
`MultinomialParams.class_names`, `OrdinalParams.level_names`, which the public
accessor already exposes as `category_names`) collide with nothing and are
tolerated as internal detail; migrating them is encouraged for tidiness but not
mandatory, and is not done where it would only churn private plumbing.

### A6 — PyStatistics does not "help": the exact request, or a loud failure

PyStatistics never silently substitutes a different device, precision,
algorithm, or estimator to "get an answer out." An explicit request is either
honored as asked, or it fails loud with a specific reason and the available
remedies. This is the generalization of the "fail loud, no silent defaults"
rule in the Errors section, made binding library-wide.

This principle has **two symmetric obligations**, and violating either is the
same mistake — substituting the library's judgement for the user's explicit
choice:

1. **Do not silently help.** A request that cannot be honored fails loud; it
   does **not** quietly degrade to something that happens to work. If
   `backend='gpu'` cannot produce a reliable fit, it raises and names the
   options — it does **not** silently fall back to CPU, silently downgrade
   precision, or silently swap estimators. "I asked for discrete-time survival
   on MPS" must yield exactly that, or a clear explanation of why it cannot be
   done — never a CPU result handed back as if it were what was asked for.

2. **Do not falsely refuse.** A request that *can* be honored must be honored.
   Rejecting a result that is actually correct is as much a violation as
   silently helping — both override the user's explicit choice. In particular,
   correctness checks are calibrated to the **requested precision**: a float32
   fit is judged against the float32 round-off floor, not an unreachable float64
   tolerance. A genuinely-converged float32 fit (a stationary point at the
   float32 floor) is accepted; only a genuinely-unreliable fit (non-stationary,
   diverged, or a broken solve) fails. Calibrating an fp32 path's acceptance to
   an fp64 tolerance — so that correct fp32 fits are rejected — is a latent
   violation of this rule and is fixed where found. (Resolved on the GPU GLM
   IRLS path: the unpenalized float32 convergence gate held the strict float64
   tolerance and rejected correct, well-conditioned fits that had reached the
   float32 floor; it now accepts a stationary fp32 optimum and fails loud only
   on a non-stationary one.)

### A7 — Dependency tiering: slim numpy/scipy core, torch optional

The hard runtime dependencies of every published `pystats*` package are **numpy and
scipy** only. **torch is optional** (`[gpu]` / `[accel]` extras), never required for a
default install. The packaging fact that forces this: as of PyTorch 2.11 (2026), PyPI
ships CPU-only torch wheels for Windows/macOS but **CUDA wheels for Linux**, and the
`+cpu` builds live only on `download.pytorch.org` — so a published package cannot pull
CPU-only torch on Linux through dependency metadata. Requiring torch would force a
multi-GB CUDA stack onto every Linux install (servers, regulated, air-gapped), violating
the Prime Directive for exactly the users who can least afford it.

Consequences:

- **The implementation reference is the simplest, fastest, most numerically-trustworthy
  *validated* path — which need NOT be numpy.** The *mathematical* reference is R;
  "numpy is canonical" was historical, not scientific. A torch path (a GPU backend, or a
  CPU accelerator such as an autodiff gradient) may be the default and **auto-preferred
  when torch is present**, provided: (a) torch is never a hard dependency; (b) the package
  runs **correctly** torch-free — degradation is *performance-only*, never function or
  correctness; (c) the choice is disclosed (`backend_name` / solution metadata), so
  default-selection is not a Fidelity (A6) violation; and (d) the torch path is validated
  numerically equivalent to a torch-free reference.
- **Where a torch accelerator delivers a speedup numpy cannot match** (e.g. MVNMLE's
  forward-Cholesky, `mixed`'s autodiff θ-gradient), the module documents the performance
  tradeoff; it does **not** obligate a numpy reimplementation of that mechanism (we do not
  hand-roll autodiff to honor slimness). The slim install is correct-but-slower there; the
  `[accel]` extra — or a provisioned CPU-torch, e.g. baked into the SGCX Docker image —
  buys the fast path.
- **Heavyweight dependencies are permitted when they are part of the validated
  computational engine and materially benefit correctness, stability, performance, or
  maintainability** — disclosed plainly, pinned compatibly, auditable/mirrorable, justified
  in the validation record, and packaged to avoid dragging an unrequested GPU/CUDA stack
  where the platform allows. **Slim installs are desirable, never at the cost of shipping
  an inferior default algorithm.**
- **Anti-rot:** CI must exercise a torch-free install of every package (full suite), so the
  torch-free path stays correct and functional rather than rotting into a second-class
  citizen.

This resolves the long-standing drift (torch creeping from optional toward de-facto-required
on CPU paths) deliberately, in favor of a slim public core plus a best-implementation
reference. Ruling recorded 2026-06-30; capability-first Option B. (SGCX ships as a Docker
image that bakes CPU-only torch, so managed clients get the fully accelerated build with no
CUDA bloat — the packaging limitation never touches the product.)

### A8 — Survival risk-set entry: KM `entry=` vs counting-process Cox `start=`

**Question.** Left-truncated Kaplan-Meier needs a delayed-entry time
(`survfit(Surv(entry, time, event) ~ 1)`); counting-process Cox needs an
interval per row (`coxph(Surv(start, stop, event) ~ x)`) so a subject can span
several rows with time-varying covariate values. Is "the time a row enters the
risk set" one concept (one name, S0) or two?

**Ruling (user, 2026-07-11): two concepts at the API surface, two names.**

- `kaplan_meier(..., entry=)` — a **per-subject delayed-entry time** (left
  truncation). Rows are subjects; the subject is at risk on `(entry, time]`.
  Python-ecosystem precedent: statsmodels `PHReg(..., entry=)`, lifelines
  `KaplanMeierFitter(entry=)` (S6).
- `coxph(..., start=)` — the **counting-process interval start** per ROW; the
  row is at risk on `(start, time]`, with `time` (the existing exit parameter)
  playing R's `stop`. Rows are spells: one subject may occupy many rows with
  different covariate values. Python precedent: lifelines
  `CoxTimeVaryingFitter(start_col=)`. Constitutional support: the registry's
  `offset` entry already steers displacement concepts to `start`/`*_start`.
  (No `stop=` kwarg exists — `time` already IS the exit name library-wide,
  and a second spelling for it would violate S0.)

**Disclosure.** Internally the two share one validated `SurvivalDesign.entry`
field — the estimators only ever see rows carrying a risk interval. The
concept distinction ruled here lives at the API surface (rows-are-subjects vs
rows-are-spells), matching how the Python ecosystem itself splits the
spelling. S0 is satisfied by that recorded distinction; no future surface
re-decides it.

### A9 — `robust=` on coxph vs `robust=` on the STL/LOESS smoother

**Question.** `coxph(..., robust=True)` requests the Lin-Wei sandwich VARIANCE
estimator (the point estimate is unchanged). `timeseries`'s STL already uses
`robust: bool` for LOESS robustness ITERATIONS (outlier downweighting, which
DOES change the fitted smooth). Two `robust=` bools, two different operations —
does S0 forbid the reuse?

**Ruling (2026-07-11): both keep `robust=`; the shared meaning is "use the
outlier/misspecification-robust variant of this procedure."**

S0 binds a name to a *concept*, and "robust" here names one concept at the
right altitude: *guard the result against departures from the model's
assumptions*. That the mechanism differs by method (a sandwich variance for a
regression, downweighting iterations for a smoother) is exactly analogous to
`method=`/`ties=` taking method-specific values — the concept is shared, the
realization is local. Both also match their R reference verbatim
(`coxph(robust=TRUE)`, `stats::loess(family="symmetric")` / STL `robust=TRUE`),
so a user transferring from R meets no surprise. A more specific spelling
(`robust_se=`) was considered and rejected: it would make `coxph` the only
place the robustness concept is not spelled `robust`, a worse S0 outcome.

**Companion — `cluster=`.** The grouped-robust grouping vector is `cluster=`
(matches `coxph(cluster=)`). Per S0's consequence rule, `cluster` is now
reserved for "independent-unit grouping for robust variance" library-wide and
must NOT later be reused for k-means-style cluster assignment without a rename.

### A10 — whole-data matrix `data=` vs regression design `X=`

**Question.** mvnmle (`mlest`) and mice take `data` / `data_or_design`; the
regression-family modules take `X` (design matrix) and `y` (response). Is "the
input matrix" one concept (one name, S0) or two?

**Ruling (2026-07-12): two concepts, two names.** A full data matrix carrying
missingness — the input to MVN MLE or multiple imputation — has no
response/predictor split; the estimator consumes the entire matrix. That is a
genuinely different concept from a regression *design* matrix `X` paired with a
response `y`. So `data` (mvnmle, mice) and `X`/`y` (regression, GLM, ordinal,
multinomial, mixed, gam) name different things and S0 is satisfied by keeping
both. Python-ecosystem precedent agrees (statsmodels/lifelines take `data` for
whole-frame estimators, `exog`/`X` for design matrices). No rename.

### A11 — single-letter `k` for the smooth basis dimension (mgcv exemption)

`gam`'s `s()`, `te()`, `ti()` take `k` (the basis dimension). S1 bans
single-letter public parameters, but `k` is a **documented exemption**, joining
the `x`/`y`/`X` carve-out: it is the entrenched mgcv name that every smoother
user already knows (`s(x, k=10)`), it is unambiguous in context, and renaming it
to `basis_dim` would alienate the reference audience for no semantic gain. The
exemption is specific to the smooth-basis dimension; it does not license
single-letter params elsewhere.

### A12 — internal `…Params` payloads are not public API

Every public fit returns a `…Solution` (the result surface). The frozen
backend-payload dataclass it wraps (`LinearParams`, `GLMParams`, `MVNParams`,
`HTestParams`, …) is **implementation detail**: it is NOT part of the public
API — not in any module's `__all__`, not re-exported from the package namespace.
It remains importable from its private submodule (e.g.
`pystatistics.regression._glm.GLMParams`) for backend authors. (Resolved in 5.0:
seven modules had leaked their `…Params` into the public surface while seven kept
them private; the leak was removed.)

### A13 — one spelling for the logit link: `'logit'`

The logit link value is `'logit'` everywhere it is selectable (regression/GLM,
gam, ordinal, multinomial). S0 binds the one concept — the logit transform
`g(mu)=log(mu/(1-mu))` — to one spelling. ordinal's former `'logistic'` value is
renamed to `'logit'` (resolved in 5.0). Rationale beyond S0: `'logistic'` names a
distribution/family, not the link transform, so it was also mildly misleading.

### A14 — reference-package vocabulary exemptions

A small, closed set of identifiers keep their entrenched reference-package
spelling as **documented exemptions** to S1/A1, because each one *is* the
canonical vocabulary a user of that reference already knows, is unambiguous in
context, and matches the reference verbatim — so the reference spelling is
clearer than a "corrected" one:

- **mice mipo diagnostics** — `PooledSolution.riv`, `.lambda_`, `.fmi`, `.df`
  (the `mice::mipo` column names: relative increase in variance, fraction of
  missing information, degrees of freedom).
- **ETS component codes** — `'A'`/`'M'`/`'N'`/`'Z'` and `model='ZZZ'` (the
  `forecast::ets` error/trend/seasonal taxonomy).
- **montecarlo bootstrap-object accessors** — `BootstrapSolution.t` / `.t0` (the
  R `boot` object's observed statistic and replicate matrix).
- **the `'nb'` family shorthand** — the mgcv-familiar alias for
  `negative-binomial`.

This list is exhaustive; a new single-letter or abbreviated public identifier
that is not on it is an S1/A1 violation to fix, not an exemption to assume.
