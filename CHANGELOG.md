# Changelog

## 2.3.0

- **New subpackage `pystatistics.nonparametric_mcar`** for distribution-
  free MCAR tests, motivated by the Lacuna ablation finding that the
  MVN-based Little's MCAR feature (MLE or MoM plug-in) does not help
  mechanism classification on heavy-tailed / categorical tabular data.

  - Added `propensity_mcar_test(data, *, model='rf'|'gbm', cv_folds=5,
    n_permutations=199, seed=0, alpha=0.05)`. Fits a sklearn
    `RandomForestClassifier` or `GradientBoostingClassifier` to predict
    each column's missingness indicator from the other columns
    (mean-imputed + per-column missing-indicator features), computes
    out-of-fold AUC, and calibrates against a permutation null. Returns
    a `NonparametricMCARResult` with `statistic = mean_auc - 0.5` and
    permutation-smoothed `p_value`. scikit-learn is an optional extra:
    `pip install pystatistics[nonparametric_mcar]`.

  - Added `NonparametricMCARResult` dataclass (statistic, p_value,
    rejected, alpha, method, n_observations, n_variables,
    n_missing_cells, extra). Intentionally narrower than the MVN-based
    `MCARTestResult` — no df / ml_mean / ml_cov / patterns, because
    nonparametric tests don't produce those.

  - Tests: 11 in `tests/nonparametric_mcar/test_propensity.py` covering
    normal cases (MCAR non-rejection, MAR rejection, reproducibility
    under fixed seed, GBM option), edge cases (fully-observed column
    ignored), and failure cases (1D input, too few rows/columns, no
    missingness, invalid hyperparameters).

  - Added `hsic_mcar_test(data, *, alpha=0.05, n_permutations=199,
    seed=0)`. Hilbert-Schmidt Independence Criterion (Gretton et al.
    2005/2008) between stochastically-imputed observed values and the
    missingness-indicator matrix, with Gaussian RBF kernel and
    median-heuristic bandwidth. Biased HSIC estimator, permutation null
    for calibration. Uses **stochastic** (column-mean + column-std
    noise) imputation rather than plain mean imputation — pure
    mean-imputation pulls heavy-missing rows toward the column
    centroid, which creates a systematic X-R coupling and rejects MCAR
    spuriously on MCAR-generated data. Pure numpy; no sklearn dep.

  - Added `missmech_mcar_test(data, *, alpha=0.05, n_permutations=199,
    n_neighbors=5, min_pattern_size=6, seed=0)`. Jamshidian-Jalal-style
    test of homogeneity of means across missingness-pattern groups,
    after k-NN imputation (via `sklearn.impute.KNNImputer`). Statistic
    is the between-pattern weighted sum of squared mean differences
    (Σ_p n_p ||μ_p − μ||²), calibrated against a pattern-label
    permutation null — equivalent in hypothesis to Jamshidian & Jalal
    (2010)'s bootstrap, but faster for the cached-scalar use case.
    Requires the same `nonparametric_mcar` extra for sklearn.

  - Tests: 9 in `test_hsic.py`, 10 in `test_missmech.py` — both
    covering MCAR non-rejection, MAR rejection, reproducibility under
    seed, and the same failure-case matrix as propensity.

  - Suite: 156/156 across `tests/nonparametric_mcar/` and `tests/mvnmle/`
    pass; no mvnmle regressions from the new subpackage.

- **Split `pystatistics/mvnmle/backends/_em_batched.py`** (501 SLOC →
  over the Rule 4 hard limit of 500) into three focused files plus a
  compatibility shim:
    - `_em_batched_patterns.py` (63 SLOC) — `_BatchedPatternIndex`
      dataclass, `_pattern_n`, `build_pattern_index`.
    - `_em_batched_np.py` (203 SLOC) — NumPy CPU backend
      (`compute_conditional_parameters_np`, `e_step_full_batched_np`,
      `compute_loglik_batched_np`, `chi_square_mcar_batched_np`).
    - `_em_batched_torch.py` (239 SLOC) — Torch GPU backend
      (`_e_step_full_torch`, `_loglik_full_torch`,
      `chi_square_mcar_batched_torch`, `compute_conditional_parameters_torch`).
    - `_em_batched.py` (30 SLOC) — thin shim re-exporting every
      symbol so existing importers in `em.py`, `solvers.py`, and
      `mcar_test.py` need no changes.
  The 157-test suite across `tests/test_code_quality.py`,
  `tests/mvnmle/`, and `tests/nonparametric_mcar/` passes after the
  split; `test_no_file_exceeds_500_code_lines` now passes.


## 2.2.0

- Fixed a `torch._C._LinAlgError` crash in `chi_square_mcar_batched_torch`
  (`pystatistics/mvnmle/backends/_em_batched.py`) on GPU FP32. The batched
  MoM fast path selected `cholesky_solve` when the SVD-based condition
  number was below threshold, but Cholesky requires positive-definiteness,
  which is strictly stronger than good conditioning. On GPU FP32, real
  tabular data (e.g. `lacuna_tabular_110` applied to UCI/OpenML datasets)
  can produce `sigma_oo` with good cond number but tiny negative eigenvalues
  from roundoff, making Cholesky fail. Fix: wrap the fast path in
  `try/except torch._C._LinAlgError`; on failure, fall back to
  `torch.linalg.pinv` for the batch (honouring the `regularize` flag —
  `regularize=False` still raises). Surfaced by dogfooding via Project
  Lacuna on 3,080 (dataset × generator) pairs across the 110-generator
  tabular registry; previously `mom_mcar_test` crashed on the first batch
  containing breast_cancer / wine / credit_card_default.

- Fixed an exception-type leak in `little_mcar_test`
  (`pystatistics/mvnmle/mcar_test.py:~250`). The ML-estimation try/except
  wrapped *every* exception — including `PyStatisticsError` subclasses
  like `NumericalError` — as a bare `RuntimeError`, breaking the
  documented `except PyStatisticsError:` pattern downstream and losing
  the original exception chain. Fix: explicitly re-raise
  `PyStatisticsError`, and use `raise ... from e` for anything else so
  the chain is preserved. Surfaced by Project Lacuna's cache builder,
  which catches `PyStatisticsError` to fall back to a sentinel entry
  when Little's test is numerically unfit for a particular
  (dataset, generator) pair; MLE failures were leaking past the catch
  and killing the whole build.

- Added ridge-fallback to the batched Cholesky sites inside the EM
  E-step / log-likelihood
  (`pystatistics/mvnmle/backends/_em_batched.py`):
  `e_step_full_batched_np` (line ~361), `_e_step_full_torch` (~680), and
  `_loglik_full_batched_torch` (~797). These compute per-pattern
  Cholesky of sigma_oo sub-blocks; real tabular data can produce
  individual sub-blocks that are numerically indefinite even when the
  global sigma is PD (integer-encoded categoricals with heavy
  collinearity in the intersection of a given missingness pattern's
  observed variables). Fix: wrap each site in `try/except LinAlgError`
  with a `ridge·I` retry (ridge = 1e-10 at pattern scale; statistically
  invisible). Also removed a dead Cholesky call in `e_step_batched_np`
  whose result was never used — it was only a crash liability.
  `np.linalg.solve` at that same site now has a pinv fallback for
  singular sub-blocks.

- Added `regularize: bool = True` to `mlest`, `_solve_em`, and
  `EMBackend.solve`, mirroring the existing convention on
  `mom_mcar_test` / `little_mcar_test`. When True (new default),
  `EMBackend._ensure_pd` applies a small diagonal ridge
  (`max(0, 1e-10 - min_eig) + 1e-12`) to the M-step sigma whenever its
  smallest eigenvalue falls below the PD threshold, rather than raising
  `NumericalError` outright. The ridge is vanishingly small relative to
  any real data scale — the typical case the old path rejected had
  min_eig ≈ 1e-13 from pure FP64 roundoff — and a UserWarning makes the
  event visible in logs. Call sites that need strict bit-for-bit
  behaviour pass `regularize=False`. Motivated by Project Lacuna
  dogfooding: applying real missingness generators to real UCI datasets
  (credit_card_default × MNAR-NonLinSocial produced min_eig ≈ -0.66) was
  hard-raising and killing the full cache build; the ridge fallback
  keeps the test numerically well-defined with negligible statistical
  impact, and the build proceeds.


## 2.1.0

- **`mom_mcar_test`: new method-of-moments MCAR test**
  (``pystatistics/mvnmle/mcar_test.py``). A separate function — not
  a new mode on ``little_mcar_test`` — because the method-of-moments
  variant **is not Little's test**. Little (1988) specifically calls
  for MLE plug-in estimators; swapping in pairwise-deletion sample
  moments gives a statistic of the same shape with different
  asymptotic properties, and calling it Little's test would be a
  polite but concrete lie. The separate function preserves the
  ``little_mcar_test`` contract exactly (matches R ``mvnmle`` bit-
  for-bit as before) while giving users a documented fast alternative.

  End-to-end timings at 15 % MCAR:

  | dataset        | shape     | little_mcar_test | mom_mcar_test |
  |----------------|-----------|------------------|---------------|
  | iris           | 150 × 4   | 2.9 ms           | 0.31 ms       |
  | wine           | 178 × 13  | 60.9 ms          | 2.17 ms       |
  | breast_cancer  | 569 × 30  | 1491 ms          | 28.7 ms       |

  For repeated-diagnostic workflows (e.g. an MCAR sweep over 3410
  datasets), this is **1.6 minutes vs ~50 minutes** end-to-end. The
  statistical trade-off is asymptotic efficiency: MoM is consistent
  under the MCAR null but not asymptotically efficient, and the
  finite-sample distribution deviates more from chi-square than
  Little's does. The docstring spells out when to use which:
  diagnostic screens → MoM; regulated submissions or anywhere the
  exact asymptotic distribution matters → Little's.

  Implementation details:
    - ``_pairwise_deletion_moments``: O(n v^2) pairwise mean and
      covariance via a single matmul. No per-column loop.
    - ``chi_square_mcar_batched_np`` / ``_torch``: fully batched
      chi-square assembly (batched SVD for conditioning,
      well-conditioned patterns through batched solve, ill-conditioned
      patterns through batched pinv as separate groups — no
      per-pattern Python loop).
    - ``backend`` parameter with same size-heuristic + visible-warning
      discipline as the EM path. GPU is supported but does not
      out-perform CPU on any tested shape — MoM's compute is small
      enough that transfer + launch overhead loses to CPU numpy.
      Auto-dispatch warns when this is the case.
    - Honesty: ``MCARTestResult`` gained a ``method`` field so
      downstream code knows which test produced a given result.
      ``little_mcar_test`` reports ``"Little (MLE plug-in)"``;
      ``mom_mcar_test`` reports ``"Method-of-moments
      (pairwise-deletion plug-in)"``.

  New tests (``tests/mvnmle/test_mom_mcar.py``, 10 tests):
  name-honesty, MLE-vs-MoM agreement on MCAR data, correct rejection
  on non-MCAR data, all-missing-row handling, speed guard of
  ≥ 10× over MLE on breast_cancer.

- **Fully-batched device-resident EM on GPU** (``_em_batched.py``
  / ``_run_em_loop_gpu``). Pre-2.1.0 the "GPU EM" path set up a
  torch device in the constructor but none of the per-iteration
  work actually ran on-device — the numpy E-step ran for every
  backend, which is why pre-2.1.0 benchmarks showed identical CPU
  and GPU timings. This release implements the real thing: one
  batched Cholesky + one batched solve over patterns for the
  regression betas, one batched gather + bmm over all N
  observations for the filled data, two dense gemms for the
  sufficient-statistic accumulators, all on-device. SQUAREM also
  runs fully on-device.

  EM-only timings (without the MCAR-assembly wrapper):

  | dataset        | shape     | CPU EM   | GPU EM   | speedup |
  |----------------|-----------|----------|----------|---------|
  | wine           | 178 × 13  | 38 ms    | 24 ms    | 1.6×    |
  | breast_cancer  | 569 × 30  | 2142 ms  | 147 ms   | 14.6×   |

  Small-data cases (apple, iris, missvals) lose on GPU because
  transfer + launch overhead exceeds the per-iteration work.
  Empirically calibrated heuristic: GPU is worth it when
  ``n_obs * n_vars > 1500``.

- **Size-heuristic dispatch with Rule-1 visibility** for both EM and
  MoM backends. When ``backend='auto'`` makes a non-obvious choice
  (e.g. picking CPU despite GPU availability because the data are
  too small), a ``UserWarning`` explains the decision and tells
  users how to override. When ``backend='gpu'`` is explicitly
  requested on small data, the request is honored (user knows best)
  but a warning notes that CPU would likely be faster. No silent
  fallbacks anywhere. New tests pin these behaviours.

- **Monotone-missingness closed-form MLE** (Anderson 1957; new
  ``pystatistics.mvnmle._monotone``). When the missingness pattern
  is monotone — when variables can be ordered such that each
  observation's missing entries form a contiguous suffix — the MVN
  MLE has a closed form via a chain of OLS regressions, with no
  iteration. Common on longitudinal data with attrition, panel
  surveys with dropout, and most sequentially-administered
  instruments. New public helpers:

    - ``pystatistics.mvnmle.is_monotone(data) -> bool``
    - ``pystatistics.mvnmle.monotone_permutation(data) -> ndarray | None``
    - ``pystatistics.mvnmle.mlest_monotone_closed_form(data) -> (mu, sigma, n)``
    - ``mlest(data, algorithm='monotone')`` routes through the
      closed-form; raises ``ValidationError`` if the data are not
      monotone (Rule 1: no silent dispatch). Users who want
      "use the closed form when applicable, fall back otherwise"
      should call ``is_monotone`` first and branch explicitly.

  The closed-form is the exact MLE (no tolerance-bounded
  approximation), matches R ``mvnmle`` reference output on both
  ``apple`` and ``missvals`` to machine precision, and is
  dramatically faster than iterative algorithms at larger v
  (a 1500 × 20 monotone dataset completes in ~2 ms vs EM's
  ~40 ms). For non-monotone random MCAR data (the common case
  in MCAR diagnostic use), detection is cheap (~O(v² n)) and
  correctly returns False so iterative algorithms run.

  New tests (``tests/mvnmle/test_monotone.py``, 12 tests):
  detection true-positive / true-negative on several canonical
  shapes; closed-form vs EM agreement; permutation invariance;
  non-monotone data raises; performance guard at v=20.

- **EM MLE: substantial real-data speedup via batched per-pattern
  linear algebra + SQUAREM acceleration** (Project Lacuna-driven).

  End-to-end ``little_mcar_test`` wall-clock at 15 % MCAR, seed 0:

  | dataset        | shape     | 2.0.1    | 2.1.0    | speedup |
  |----------------|-----------|----------|----------|---------|
  | apple          | 18 × 2    |  1.9 ms  |  2.0 ms  |  flat   |
  | missvals       | 13 × 5    | 19.9 ms  |  9.5 ms  |  2.1×   |
  | iris           | 150 × 4   |  2.8 ms  |  2.8 ms  |  flat   |
  | wine           | 178 × 13  | 79.4 ms  | 41.5 ms  |  1.9×   |
  | breast_cancer  | 569 × 30  | 3278 ms  | 2089 ms  |  1.6×   |

  For workloads that run MCAR repeatedly over many datasets
  (e.g. a 3410-entry MCAR sweep), this is roughly a 1-hour reduction
  per full pass at Lacuna's current scale.

  Three changes stack:

  1. **Batched per-pattern conditional parameters** (new
     ``pystatistics.mvnmle.backends._em_batched``). The E-step used
     to loop in Python over missingness patterns, issuing a scalar
     Cholesky + triangular solve per pattern. It now stacks all P
     pattern-sigma submatrices into a single
     ``(P, v_max, v_max)`` tensor (identity-padded in the unused
     slots so the Cholesky stays well-defined) and runs one batched
     Cholesky + one batched solve for the whole iteration. The
     accumulator loop over patterns remains in Python because
     ``n_k`` varies and full observation-level padding hurt more
     than it helped on the representative shapes we benchmarked.

  2. **SQUAREM acceleration** (Varadhan & Roland 2008; new
     ``pystatistics.mvnmle.backends._squarem``). EM's linear
     convergence is sped up by a Steffensen-style extrapolation of
     three consecutive EM iterates, safeguarded by a monotonicity
     check on the observed-data log-likelihood. Typical effect on
     well-behaved EM problems: 2–4× reduction in underlying
     EM-step-equivalents. Preserves the MLE — the convergence
     point is unchanged, only the path is shorter. On by default
     via a new ``accelerate=True`` kwarg on ``EMBackend.solve``;
     pass ``accelerate=False`` for the plain-EM reference path.

  3. **Fully batched observed-data log-likelihood**
     (``compute_loglik_batched_np``). The SQUAREM monotonicity
     safeguard calls the log-likelihood often, so that path
     needed to be cheap. The implementation now does one batched
     Cholesky over all patterns for log-determinants and one
     batched solve across all N observations for the quadratic-
     form contribution — no per-pattern Python loop.

- **Benchmark harness** (``benchmarks/mvnmle_bench.py``). Runs the
  five reference shapes (apple, missvals, iris, wine,
  breast_cancer) across the (algorithm, backend) matrix and
  prints wall-clock / iteration counts per case. Use
  ``--quick`` to skip the BFGS cases that don't converge on
  high-$v$ data; ``--tag`` labels a run for diff against prior
  baselines.

- **Documented why direct-BFGS is not always the right default.**
  Internal notes and the 2.0.0 / 2.0.1 release narrative already
  covered why ``algorithm='em'`` became the ``little_mcar_test``
  default; this release adds the story of why batching helps EM
  significantly but does *not* rescue direct-BFGS on realistic
  high-$v$ data (layer-3 Hessian conditioning is parameterization-
  invariant; batching only addresses layer-1 launch overhead).
  See ``GPU_BACKEND_CONVENTION.md`` Section 0 for the "when to
  add a GPU backend and when not" rule that drove the 2.0.1
  cleanup; this release extends that logic with
  "accelerating the algorithm by reducing iteration count
  (SQUAREM) is cheaper than accelerating each iteration."

- **Finding: the "GPU EM" backend was never actually running on
  GPU.** The ``device='cuda'`` / ``'mps'`` constructor flag set
  up ``self._torch`` but none of ``_e_step`` / ``_m_step`` /
  ``_compute_loglik`` used it — the numpy path ran for every
  backend. That's why pre-2.1.0 benchmarks showed identical
  CPU and GPU EM timings. We attempted to implement a real
  device-resident EM loop and found it was slower than CPU for
  all the shapes we care about (per-pattern kernel-launch
  overhead dominates the small per-pattern matrix work). The
  honest answer for now is that GPU EM stays CPU-equivalent
  by design; a future release may revisit with fully N-parallel
  observation-level batching if a workload appears where the
  GPU can actually win. This behaviour is unchanged from prior
  releases — we're just documenting what was already true.

- **SQUAREM test coverage** (new ``tests/mvnmle/test_squarem.py``).
  Four tests pinning the invariants: same MLE as plain EM on
  apple; substantially fewer EM-equivalent steps on missvals;
  monotonicity of log-likelihood preserved across iteration
  caps; same MLE as plain EM on a realistic shape (sklearn
  wine with 15 % MCAR).


## 2.0.1

- **GPU Backend Convention: codified when NOT to add a GPU backend**
  (`pystatistics/GPU_BACKEND_CONVENTION.md`, new Section 0). Formalises
  the Oprah-test rule: don't give every module a GPU backend — only
  those where the algorithm is actually a good fit for GPU hardware
  (large dense linear algebra, big-N likelihoods, batched independent
  fits). Explicitly calls out the non-targets (many-small-fits,
  sequential algorithms, transfer-bound cheap compute) and the ~100 ms
  CPU-time threshold below which launch + PCIe overhead eats any
  speedup. Names the deliberate CPU-only modules (`anova`,
  `timeseries.ets`, `survival.coxph`, `multivariate.factor_analysis`,
  acf / stationarity) so future contributors know these omissions are
  intentional, not oversights.
- **`auto_arima` now accepts `backend=` and `method=`**
  (`pystatistics/timeseries/_arima_order.py`). Parameters are threaded
  through `_stepwise_search` / `_grid_search` / `_try_fit` to every
  candidate `arima()` call. Previously there was no way to route the
  per-candidate fits through the GPU Whittle path, so on a
  GPU-equipped box the search was stuck on CPU even if the user
  wanted frequency-domain speedups. Default `backend=None` → CPU
  (R-reference); pass `method='Whittle', backend='gpu'` to run each
  candidate on GPU.
- **`little_mcar_test` now accepts `backend=` and `algorithm=`**
  (`pystatistics/mvnmle/mcar_test.py`). Previously the function had no
  backend knob and always ran `mlest()` with the default — which after
  2.0.0's CPU-default sweep meant there was no way to route Little's
  MCAR test through the GPU path, defeating the purpose of having a
  GPU MLE backend at all. The new `backend` parameter is forwarded
  straight to `mlest`; default None → CPU (consistent with the 2.0.0
  convention), `'gpu'` / `'auto'` opt in to GPU. The per-pattern
  test-statistic accumulation still runs on CPU — it's O(P × v³) for
  v = observed vars per pattern (typically < 50) and was never the
  bottleneck; the ML estimation step is where GPU matters. Verified
  GPU/CPU results agree within FP32 tolerance (Δ stat ≈ 1.4e-4 on
  the apple dataset).


## 2.0.0

- **BREAKING: Default backend is now CPU across all public solvers**
  (requires next version be a major bump per semver). Every public
  fit/solver entry point that previously defaulted to `backend='auto'`
  now defaults to `backend=None`, which resolves to `'cpu'` for numpy
  input — the R-reference path. GPU is never the default; callers
  must opt in explicitly with `backend='gpu'` (fail-loud if
  unavailable) or `backend='auto'` (prefer GPU, fall back to CPU).
  Rationale: GPU behavior is not guaranteed across installs, and
  regulated-industry users need "unspecified backend" to mean the
  validated path. Affected entry points:
    - `regression.fit` (covers OLS + all GLM families)
    - `mvnmle.mlest`
    - `survival.discrete_time` and internal `discrete_time_fit`
    - `montecarlo.boot`, `montecarlo.permutation_test`
    - `descriptive.describe`, `.cor`, `.cov`, `.var`, `.quantile`,
      `.summary`
    - `hypothesis.*` (signatures normalized — behavior unchanged;
      CPU was already the effective default)
  The convention already documented in
  `pystatistics/GPU_BACKEND_CONVENTION.md` (numpy input → CPU) is now
  uniformly enforced. Previously-compliant modules (`multivariate.pca`,
  `multinomial`, `ordinal`, `timeseries.arima` / `.arima_batch`,
  `gam`) are unchanged. Migration: if you were relying on implicit
  GPU selection on a GPU-equipped box, add `backend='auto'`
  (best-effort GPU) or `backend='gpu'` (require GPU) to the
  affected calls.


## 1.9.1

### README — catch-up for 1.8.0 and 1.9.0

- **README's "What's New" section updated** to reflect the 1.8.0
  and 1.9.0 releases (the full GPU backend sweep, Whittle ARIMA,
  GPU-resident PCAResult, `arima_batch`). Both prior releases
  shipped with README still describing 1.7.0 as current. Doc-only
  change — no code is touched.

- **`.release/CHECKLIST.md` already states** that the feature-
  summary prose in README is manual work that `release.py` does
  not auto-generate; the issue was the step being skipped, not the
  checklist being missing.


## 1.9.0

### Batched ARMA fit (arima_batch)

- **`arima_batch(Y, order=(p, d, q), method='Whittle')`** — new
  top-level function for fitting K independent ARMA models on the
  rows of a `(K, n)` matrix simultaneously. Non-seasonal,
  Whittle-method only in 1.9.0. Files:
  `pystatistics/timeseries/_arima_batch.py` (dispatcher + result),
  `backends/whittle_batch_gpu.py` (`BatchedWhittleGPU`).

- **Algorithm (GPU path):** one batched `torch.fft.rfft` computes
  the full `(K, m)` periodogram in a single call. Per-iteration the
  batched Whittle NLL evaluates `log|MA|² − log|AR|²` via a shared
  `(m, k)` cos/sin table and a K-wise einsum — one kernel. The
  sum-of-K-NLLs scalar's backward pass yields the full `(K, p+q)`
  gradient thanks to per-series loss independence. scipy L-BFGS-B
  can't wrap a batched objective cleanly, so optimization runs as
  batched Adam with per-row gradient-norm convergence: rows that
  reach the L∞ tolerance are frozen so their Adam state stops
  updating while the harder series finish converging.

- **Starting values:** Yule-Walker AR starts per series (always
  stationary) + mild negative MA heuristic — same convention as the
  single-series `method='Whittle'` path, where zero AR start can
  drift across the unit circle into the non-stationary mirror basin.

- **CPU fallback:** `backend='cpu'` (default) is a Python loop over
  `arima(method='Whittle')` per row. No speed win — this is the
  API-shape packaging for users with K series. The GPU path is
  where the actual benefit is.

- **Measured wall time (RTX 5070 Ti, ARMA(2, 0, 1), random stationary
  coefficients):**

    | (K, n)           | serial CPU Whittle loop | batched GPU | speedup |
    |------------------|------------------------:|------------:|--------:|
    | K=50    n=2000   |   163 ms                |   212 ms    |  0.8×  |
    | K=200   n=2000   |   618 ms                |   223 ms    |  2.8×  |
    | K=500   n=2000   |  1.57 s                 |   227 ms    |  6.9×  |
    | K=1000  n=2000   |  3.03 s                 |   233 ms    | 13.0×  |
    | K=2000  n=5000   |  6.63 s                 |   796 ms    |  8.3×  |
    | K=500   n=10000  |  3.36 s                 |   313 ms    | 10.7×  |

  Crossover at K ≈ 100; at K ≥ 200 the batched GPU path wins
  cleanly. Below that, serial CPU wins because Adam's ~300 steps
  can't beat scipy L-BFGS-B's ~10-20 curvature-aware iterations
  when there's no parallelism to amortise.

- **`ARMABatchResult` dataclass** mirrors `ARIMAResult` shape-wise:
  `ar`, `ma` have shape `(K, p)` / `(K, q)`, `sigma2` and `converged`
  are length-K, plus `n_iter` (max across series) and `mean`
  (per-series). Frozen dataclass.

- **6 new tests** in `tests/timeseries/test_arima.py`
  (`TestArimaBatch`, `TestArimaBatchGPU`): CPU equivalence, backend
  errors (invalid backend / non-Whittle method / 1-D input), tensor
  input routing, Rule 1 refusal on tensor + backend='cpu', GPU vs
  serial CPU convergence match.

### GPU-resident PCAResult

- **`pca(..., device_resident=True)`** — opt-in path that returns a
  `PCAResult` whose numeric fields (`sdev`, `rotation`, `center`,
  `scale`, `x`) stay as `torch.Tensor` instances on the fit's device
  instead of being materialised as numpy arrays. The ~150 ms D2H
  copy of a 1M × 100 FP32 scores matrix otherwise dominates every
  op downstream of PCA in an amortized GPU pipeline. Measured 3.4×
  end-to-end speedup at 1M × 100 FP32 via `DataSource.to('cuda')`
  (202.9 ms → 59.3 ms per fit). Default `device_resident=False`
  preserves 1.8.0 behavior — no API break.

- **`PCAResult.to_numpy()`** — returns a new PCAResult with all
  numeric fields materialised to numpy. Idempotent on numpy-backed
  results. This is the explicit "I want CPU numpy" escape hatch,
  matching the rest of the library's no-silent-migration contract.

- **`PCAResult.to(device)`** — move between devices. `to('cpu')` is
  the same as `to_numpy()`; any other device requires torch and
  materialises fields as tensors on that device. Round-trip via CPU
  is value-exact.

- **`PCAResult.device` property** — reports `'cpu'`, `'cuda'`, or
  `'mps'` depending on where the numeric fields live.

- **`PCAResult.explained_variance_ratio` and
  `cumulative_variance_ratio` are array-type polymorphic** — return
  the same dtype as `sdev`. No forced D2H just to compute a
  length-k vector.

- **`PCAResult.summary()`** materialises internally via
  `to_numpy()`, so callers never need to think about the device to
  print a summary. The D2H cost is negligible (length-k vectors).

- **8 new tests** in `tests/multivariate/test_multivariate.py`
  (`TestPCADeviceResident`): back-compat, tensor-fields, to_numpy
  equivalence, idempotence, CPU-no-op, polymorphic variance ratio,
  summary, cross-device `.to()`.


## 1.8.0

### Whittle approximate MLE for ARMA (new method, CPU + GPU)

- **`method='Whittle'` on `arima()`** — frequency-domain approximate
  MLE based on the sample periodogram. Modules:
  `pystatistics/timeseries/_whittle.py` (CPU numpy) and
  `pystatistics/timeseries/backends/whittle_gpu.py` (GPU torch /
  autograd). The Whittle concentrated NLL takes one real FFT of the
  centred, differenced series and then, per L-BFGS-B evaluation,
  one elementwise ``log(|MA(e^{iω})|² / |AR(e^{iω})|²)`` plus two
  reductions — no per-iteration time-domain recursion. Initialised
  with Yule-Walker AR starts (always stationary) so the optimizer
  doesn't drift into the mirror-image non-stationary basin that the
  frequency-domain likelihood is symmetric under.

  Measured per-fit latency on an RTX 5070 Ti (ARMA(2, 0, 1) on
  stationary synthetic data):

    | n         | CSS-ML   | Whittle CPU | Whittle GPU | speedup (ML → GPU) |
    |-----------|---------:|------------:|------------:|-------------------:|
    | 2 000     |  10.8 ms |   2.7 ms    |  17.1 ms    |  0.6× (too small) |
    | 10 000    |  25.3 ms |  11.9 ms    |  12.1 ms    |   2.1×            |
    | 50 000    | 578.5 ms |  33.0 ms    |  41.8 ms    |  13.8×            |
    | 200 000   | 486.0 ms |  90.0 ms    |  45.6 ms    |  10.7×            |
    | 1 000 000 |    2.46 s|   1.70 s    |  67.4 ms    |  **36.4×**        |

  Whittle CPU alone wins over CSS-ML at every size from n ≈ 2000
  upward (1.4-17.5× depending on shape). The GPU path is the real
  prize at n ≳ 50 000 where FFT-dominated cost scales much better
  than exact-ML's Kalman recursion. For very short series (n < 3000)
  stay on CSS-ML.

- **Scope limitations (documented, will raise)** — Whittle is
  non-seasonal ARMA only in 1.8.0: passing `seasonal=(...)` with
  `method='Whittle'` raises `ValidationError`. Non-zero differencing
  (`d > 0`) is supported — differencing is applied before the FFT.
  Variance-covariance (`result.vcov`) is returned as a NaN matrix
  with the right shape; users needing coefficient SEs should use
  `method='ML'` or `'CSS-ML'`.

- **Two-tier validation for Whittle** — CPU Whittle is validated
  against the exact time-domain ML fit on long stationary series
  (agrees to within Whittle's O(1/n) approximation floor). GPU
  Whittle is validated against CPU Whittle: FP64 on CUDA matches
  coefficients and σ² to ~1e-8; FP32 matches σ² and log-likelihood
  at the `GPU_FP32` tier. `TestArimaWhittle` + `TestArimaWhittleGPU`
  add 11 new tests in `tests/timeseries/test_arima.py`.

### CPU multinomial speedup (backport from GPU work)

- **Multinomial vcov: analytical Hessian on CPU** —
  `pystatistics/multinomial/_solver.py:_compute_vcov`. The original
  CPU path used central-difference finite differencing on the
  analytical gradient — ``2 · n_params`` full gradient sweeps per
  vcov call, dominating total fit time past modest scale. Replaced
  with the analytical block Hessian

      H[j, k] = X' · diag(W_{jk}) · X

  that the GPU backend has used since 1.8.0-dev.2 (``W_{jj} =
  diag(p_j(1-p_j))``, ``W_{jk} = diag(-p_j p_k)`` for j ≠ k). One
  weighted GEMM per (j, k) block; no finite-difference step-size
  trade-off; bit-identical to the true Hessian. Zero new
  dependencies — pure numpy.

  Measured on a consumer x86 (single-threaded numpy, MKL):

    | shape (n, p, J)       | old (finite diff) | new (analytical) | speedup |
    |-----------------------|------------------:|-----------------:|--------:|
    | n=1000   p=10  J=4    |   4.5 ms          |   0.2 ms         | 28.7×  |
    | n=5000   p=20  J=5    |  58.0 ms          |   1.9 ms         | 29.9×  |
    | n=20000  p=30  J=6    | 536.5 ms          |  16.2 ms         | 33.2×  |

  Standard-error difference vs. the old finite-difference Hessian is
  ~1e-10, well inside the inversion-noise floor at these condition
  numbers — output is statistically unchanged. The full
  ``tests/multinomial`` suite (48 tests) passes unchanged.

### GAM GPU backend (1.6.x additions, cont.)

- **GAM GPU backend** — `pystatistics/gam/backends/gpu_pirls.py` +
  `backends/_gpu_family.py`. `GAMGPUFitter` holds ``X_aug``, ``y``,
  and the stacked penalty tensor device-resident for the full fit.
  The outer L-BFGS-B over ``log(lambda)`` (GCV or REML; ~50 P-IRLS
  evaluations per fit) now pays transfer only for the
  ``log_lambdas`` vector going in and the scalar criterion coming
  out per evaluation — the large design matrix and penalty stack
  never re-cross the PCIe bus during the lambda search. All four
  supported CPU families (gaussian / binomial / poisson / gamma)
  are on GPU. Measured per-fit latency on an RTX 5070 Ti:

    | shape                          | CPU       | GPU (numpy per-call) | speedup |
    |--------------------------------|----------:|---------------------:|--------:|
    | n=500  1-smooth  k=10          |  16.2 ms  |   3.4 ms             |  4.8×  |
    | n=2000  1-smooth  k=10         |  14.5 ms  |   3.9 ms             |  3.7×  |
    | n=10000  1-smooth  k=10        |  31.8 ms  |   7.4 ms             |  4.3×  |
    | n=50000  1-smooth  k=10        | 156.8 ms  |  26.2 ms             |  6.0×  |
    | n=500   3-smooth  k=10         |  69.7 ms  |   6.7 ms             | 10.4×  |
    | n=5000  3-smooth  k=15         | 209.2 ms  |  16.5 ms             | 12.7×  |
    | n=20000 3-smooth  k=15         |   1.25 s  |  43.7 ms             | 28.6×  |

  Bigger wins with multiple smooths, where the outer optimizer does
  more work per fit.

- **Hat-trace solve routes through numpy LAPACK for CPU parity** —
  the penalised normal-equation matrix ``X'WX + sum lambda_j S_j``
  has cond(A) up to ~1e17 in FP64 when the optimizer probes small
  lambda (the penalty does not fully eliminate the design matrix's
  null space). LAPACK ``getrf`` and cuSOLVER ``getrf`` pivot
  differently on near-singular systems and give trace(F) values
  that can differ by factors of two (including sign flips) for
  lambdas differing by 1e-6. To keep GCV/REML scores aligned with
  CPU within the optimizer's bracketing precision, the p×p hat-
  trace solve is done on host via ``np.linalg.solve``. The p is
  typically 30-80 so the ~100 µs round-trip is negligible next to
  the n×p GEMM that forms X'WX (which stays on device).

- **GAM accepts ``backend`` / ``use_fp64`` kwargs** — matching
  ``pca()``, ``multinom()``, ``polr()``, and ``gee()``. Default is
  ``backend='cpu'`` per the regulated-industry convention.
  ``backend='gpu'`` raises if no GPU is available;
  ``backend='auto'`` falls back to CPU silently. If a family is
  outside the GPU family table (currently all four CPU families are
  supported, so this is future-proofing), ``backend='gpu'`` raises
  and ``backend='auto'`` falls back.

- **Two-tier validation for GAM** — CPU path remains validated
  against R ``mgcv::gam()``. GPU path is validated against CPU:
  FP64 on CUDA matches fitted values to ~1e-6, deviance and GCV to
  ~1e-4 (bounded by the intrinsic hat-trace instability above);
  FP32 matches at the ``GPU_FP32`` tier. ``TestGAMGPU`` in
  ``tests/gam/test_gam.py`` adds the standard 7 GPU-backend tests.

### polr GPU backend (1.6.x additions, cont.)

- **polr GPU backend** — `pystatistics/ordinal/backends/gpu_likelihood.py`.
  The CPU polr builds its vcov via `scipy.approx_fprime` on each row
  of the analytical gradient — an O(n_params²) sweep of gradient
  evaluations that, past toy sizes, dominates total fit time. On GPU,
  `PolrGPULikelihood` holds X and y_codes on device across all
  L-BFGS-B evaluations, computes the NLL and its gradient via torch
  autograd in one forward+backward per optimizer step, and builds the
  Hessian (for vcov) via `torch.autograd.functional.hessian` — one
  backward per parameter, O(n_params) rather than O(n_params²), all
  batched over the full n-row sample. Supports all three CPU link
  functions (logistic, probit, cloglog). Measured per-fit latency on
  an RTX 5070 Ti:

    | shape (n, p, K)       | CPU      | GPU (numpy per-call) | GPU (DataSource) | speedup |
    |-----------------------|---------:|---------------------:|-----------------:|--------:|
    | n=1000 p=3 K=4        | 12.8 ms  | 15.9 ms              | 16.0 ms          | 0.8×   |
    | n=5000 p=5 K=5        | 78.5 ms  | 18.8 ms              | 18.9 ms          | 4.2×   |
    | n=20000 p=10 K=4      | 427.8 ms | 27.1 ms              | 22.5 ms          | 19.0×  |
    | n=100000 p=20 K=5     | 14.05 s  | 35.6 ms              | 31.4 ms          | 448×   |

  Small problems (n ≲ 2000) see no benefit — the Python/driver
  overhead on the GPU path exceeds the numpy fit time. Crossover is
  around n = 3000; above n = 20000 the GPU path is an order of
  magnitude faster even without DataSource amortization because the
  CPU vcov step scales quadratically in n_params × n.

- **polr accepts torch.Tensor input** — `polr()` now takes either
  numpy arrays or `torch.Tensor` for `y` and `X`, with backend
  auto-inferred from the input (numpy → cpu, GPU tensor → gpu),
  matching the convention established by `pca()`, `multinom()`, and
  `gee()`. Explicit `backend='cpu'` with a GPU tensor raises (Rule 1).

- **Two-tier validation for polr** — CPU path remains validated
  against R `MASS::polr()`. GPU path is validated against CPU: FP64
  on CUDA matches to machine precision; FP32 matches on log-
  likelihood and deviance at the `GPU_FP32` tier (rtol=1e-4,
  atol=1e-5). `TestPolrGPU` in `tests/ordinal/test_ordinal.py` adds
  the standard 7 GPU-backend tests.

### GPU backends for new modules (1.6.x additions)

- **`DataSource.to(device)` — device-resident data handles.**
  Adds an explicit `.to('cuda')` / `.to('cpu')` / `.to('mps')` method
  on `DataSource`. Returns a new DataSource (immutable — Rule 5)
  whose arrays are `torch.Tensor` on the target device (or numpy on
  CPU). The `DataSource.device` property reports the current device.

  This unlocks the amortized-transfer pattern for multi-fit
  workflows:
  ```python
  ds = DataSource.from_arrays(X=X, y=y)
  gds = ds.to("cuda")                   # pay H2D transfer once
  pca(gds["X"])                          # no H2D
  multinom(y, gds["X"])                  # no H2D
  ```
  Without this, every GPU fit re-transfers X from host memory; on a
  1M × 100 FP32 matrix that's ~66 ms of PCIe traffic per call that
  dominates the ~5 ms of actual compute.

  Measured end-to-end on RTX 5070 Ti, 1M × 100 FP64 input:

  | Path | Per-fit time | vs CPU |
  | ---- | ------------ | ------ |
  | CPU (numpy) | 5,984 ms | 1.0× |
  | GPU (numpy, per-call transfer) | 563 ms | 10.6× |
  | GPU (DataSource, amortized) | **155 ms** | **38.6×** |
  | `.to('cuda')` transfer (paid once) | 43 ms | — |

- **PCA `backend` default is now inferred from input type.**
  Passing a numpy array defaults to `backend='cpu'` (regulated-
  industry default: unspecified ⇒ R-reference). Passing a
  GPU-resident `torch.Tensor` (the output of `ds.to('cuda')['X']`)
  defaults to `backend='gpu'`. Explicit `backend='cpu'` with a GPU
  tensor raises — no silent device migration.

- **`multinom()` accepts GPU `torch.Tensor` input, same convention
  as `pca()`.** Passing `gds['X']` from a GPU DataSource skips the
  per-call H2D transfer; the `MultinomialGPULikelihood` helper accepts
  the tensor directly and only casts dtype if the fit precision
  doesn't match. Backend default is inferred from input type (numpy →
  `cpu`, GPU tensor → `gpu`); explicit `backend='cpu'` on a GPU tensor
  raises per Rule 1. Post-fit ``fitted_probs`` and log-likelihood are
  computed through the same on-device likelihood object, avoiding a
  redundant H2D round-trip for the final answer.

  The amortization benefit for multinomial is smaller than for PCA
  (on n=500k, p=20, J=10: 98 ms fresh-numpy → 88 ms DataSource-
  amortized, 1.1×) because the L-BFGS-B outer loop naturally amortizes
  the initial transfer across ~9 iterations. Pattern still applies
  uniformly for consistency across the GPU-capable modules.

- **`pystatistics/GPU_BACKEND_CONVENTION.md` (NEW).** Internal
  convention doc for authors of GPU backends. Codifies the
  input-type-driven backend inference, the Rule 1 refusal on
  explicit device mismatches, the `sys.modules`-probe for torch
  (avoids forcing torch import on CPU paths), keeping moments on GPU
  to avoid sync points, output-dtype tracking compute dtype, the
  `GPU_FP32` tier for tests, the FP32 `ftol` floor, and the standard
  `Test<Module>GPU` test class template. PCA and multinomial are the
  exemplars; GEE, polr, GAM, and the future timeseries GPU path will
  follow the same shape.

- **PCA GPU path output dtype matches compute dtype.** When
  `use_fp64=False` (the GPU default), scores/rotation/sdev are
  returned as float32 rather than being force-promoted to float64 at
  the final `.cpu()`. Promotion doubled the D2H payload for scores
  (400 MB → 800 MB) without adding precision — the fit ran in FP32,
  and zero-padded FP64 bits don't change that. Eliminates ~140 ms of
  wasted PCIe traffic on 1M-row fits.


- **`multinomial/backends/gpu_likelihood.py` (NEW): GPU backend for
  `multinom()`.** Keeps `X` and the one-hot `y` on the device for the
  full optimization. Per scipy L-BFGS-B iteration only the parameter
  vector crosses the bus going in, and (scalar NLL, gradient vector)
  come out — gradient is computed by PyTorch autograd in one
  forward+backward pass rather than by the CPU path's two separate
  analytical passes. The variance-covariance matrix is also computed
  on GPU, analytically, via the softmax block Hessian
  ``H[j, k] = X' · W_{jk} · X`` instead of the CPU path's 300-call
  numerical-Hessian finite-difference loop. That was the hidden
  bottleneck: once the optimizer was on GPU, the CPU vcov was taking
  ~99% of wall time.

  Backend dispatch follows the `backend='cpu'` default /
  `'auto'`-prefers-GPU / `'gpu'`-is-explicit convention used
  throughout the new GPU paths.

  FP32 default with `use_fp64=False` for throughput; the
  optimizer's tolerance is automatically floored at 1e-5 in FP32
  because gradient precision is ~1e-7 and tighter tol routinely
  trips scipy's ABNORMAL line-search stall on noise.

  Measured on RTX 5070 Ti, FP32:

  | n × p × J | CPU (ms) | GPU (ms) | Speedup |
  | --------- | -------- | -------- | ------- |
  | 2,000 × 10 × 5   | 20    | 11   | 1.9× |
  | 10,000 × 20 × 10 | 623   | 85   | 7.3× |
  | 50,000 × 30 × 10 | 4,425 | 91   | 48.7× |
  | 200,000 × 50 × 8 | 31,520 | 172 | 183× |
  | 500,000 × 20 × 10 | 62,359 | 413 | 151× |

  Log-likelihood agreement CPU vs GPU on MASS::fgl: CPU −108.003909,
  GPU FP32 −108.003938, GPU FP64 −108.003909. Fitted probabilities
  within 1.3e-3 on FP32 and 2e-6 on FP64 of the CPU path.



This cycle brings GPU support to the modules added in 1.6.0 that
previously had only CPU implementations. The two-tier validation rule
from the README applies: CPU is the reference (validated against R to
near machine precision); GPU is validated against CPU. FP32 is the
default on GPU — consumer NVIDIA GPUs have deliberately crippled FP64
throughput (RTX 5070 Ti is ~1/64 FP64 rate of FP32), so FP32 is where
any speedup lives. FP64 remains available on CUDA for users who need
machine-precision parity.

- **`multivariate/backends/gpu_pca.py` (NEW): GPU backend for `pca()`
  via `torch.linalg.svd`.** Adds `backend` and `use_fp64` keyword
  arguments. Backend dispatch: `'cpu'` (default — the R-reference
  numpy SVD, matches `prcomp()` to rtol = 1e-10; pystatistics is
  aimed at regulated industries where the methodology must be
  defensible in a paper's methods section and the CPU path is the
  validated one); `'auto'` (prefers GPU when PyTorch + CUDA/MPS is
  available, falls back to CPU) — opt in when you have verified
  your pipeline does not need R-level reproducibility; `'gpu'`
  (explicit — raises if no GPU is available per Rule 1, no silent
  fallback). The entire
  pipeline — centering, scaling, SVD, sign-fix, scores — runs on
  GPU with a single host-to-device transfer of X and a single
  device-to-host transfer of the (small) result tensors. FP32 is
  the default on GPU (consumer NVIDIA parts have ~1/64× FP64
  throughput, so FP64-on-GPU is slower than numpy LAPACK on most
  shapes); validated against CPU at the project's `GPU_FP32`
  tolerance tier (rtol = 1e-4, atol = 1e-5) — "statistically
  equivalent", per the README's "Design Philosophy" section.
  FP64 remains available on CUDA via `use_fp64=True` for users who
  need machine-precision CPU parity.

  Measured on RTX 5070 Ti vs. single-threaded numpy LAPACK, FP32:

  | Shape | CPU | GPU | Speedup |
  | ----- | --- | --- | ------- |
  | (5000, 500)   | 104 ms  | 26 ms  | 4.0× |
  | (2000, 2000)  | 689 ms  | 248 ms | 2.8× |
  | (10000, 200)  | 43 ms   | 14 ms  | 3.0× |
  | (20000, 500)  | 373 ms  | 100 ms | 3.7× |
  | (5000, 2000)  | 1261 ms | 296 ms | 4.3× |

  TF32 matmul is deliberately NOT enabled: its 10-bit mantissa gives
  ~1e-3 precision per op, which composes past the `GPU_FP32` rtol =
  1e-4 contract on the `scores = X_centered @ V` matmul. Future GPU
  paths may enable TF32 per-kernel where the tier tolerance allows.

- **`pca()` gains `method` and `force` parameters: Gram-matrix path.**
  Adds a second GPU algorithm — eigendecomposition of X'X — alongside
  the SVD path. The Gram path replaces the iterative cuSOLVER SVD
  with one big GEMM (X'X) plus a symmetric eigendecomposition on a
  p×p matrix, both GPU sweet spots. For tall-skinny
  well-conditioned data and wider data (p ≥ 500) Gram is 1.5–2×
  faster than the SVD path; for pathological shapes where cuSOLVER's
  SVD runs out of resources entirely (e.g. n = 10⁷, p = 20 on a
  16GB card), Gram completes successfully — so it also extends the
  operating envelope of the library, not just throughput.

  Precision cost: cond(X'X) = cond(X)². The Gram path refuses on
  ill-conditioned data unless `force=True`, matching the OLS Cholesky
  pattern already in `gpu.py`. Thresholds are precision-tiered:
  cond(X) ≤ 10⁶ for FP64, cond(X) ≤ 10³ for FP32. Auto mode
  (`method='auto'`) tries Gram when n > 2p and falls back to SVD
  silently on condition failure — the "best safe GPU algorithm for
  this data shape" dispatch.

  Policies:
    - `method='svd'` (default) — always safe, SVD of X.
    - `method='gram'` — Gram eigendecomposition; raises on ill-
      conditioned inputs.
    - `method='auto'` — Gram when viable, SVD fallback.
    - `force=True` — bypass the condition gate (for users who know
      their data is well-conditioned despite the estimator's
      disagreement).

  Gram vs. SVD (on GPU, RTX 5070 Ti, FP32, well-conditioned Gaussian
  data):

  | Shape | GPU-SVD | GPU-Gram | Gram / SVD |
  | ----- | ------- | -------- | ---------- |
  | (5000, 500)   | 28 ms  | 17 ms  | 1.7× |
  | (5000, 1000)  | 87 ms  | 42 ms  | 2.1× |
  | (10000, 1000) | 126 ms | 77 ms  | 1.6× |
  | (10000, 2000) | 377 ms | 172 ms | 2.2× |
  | (100000, 500) | 406 ms | 372 ms | 1.1× |
  | (5M, 20)      | 834 ms | 759 ms | 1.1× |

  Speedups are more modest than the originally-projected 30-100×
  because on these shapes GPU-SVD is already within a small constant
  of the physical memory-bandwidth floor — the Gram path principally
  wins when SVD cost becomes substantial (p ≥ 500). The envelope
  extension is the more consistent benefit.


## 1.7.0

### Performance

- **`core/result.py`: eliminate 500+ ms cold-import cost on first fit.**
  `_default_provenance()` used to run `import torch` on every `Result()`
  construction so it could record `torch.__version__`. On CPU-only code
  paths (where torch has not otherwise been loaded) this triggered a
  full torch module graph load — ~800 ms for 770 modules — on the first
  fit of every session. Fix: (1) only probe torch if it is already in
  `sys.modules` (GPU code paths will have imported it themselves; CPU
  paths legitimately shouldn't pay the cost), (2) cache the probe result
  so subsequent `Result()` constructions are a dict copy. Measured: OLS
  on California Housing (n=20,640) first-call went from 578 ms to ~5 ms.
  Steady-state unchanged (~3 ms, faster than R's `lm()`).

- **`timeseries/_arima_factored.py` (NEW) + `_arima_fit.py` wiring:
  optimize SARIMA in factored (ma1, sma1) space instead of expanded
  (ma_eff_1..ma_eff_{q+sq·m}) space.** Previously seasonal models were
  optimized over the expanded MA polynomial — for Box-Jenkins airline
  SARIMA(0,1,1)(0,1,1)[12] that meant scipy's L-BFGS-B exploring 13
  parameters on a 2-D manifold, with ~1600 likelihood evaluations per
  fit. The factored path optimizes 2 params directly, mirroring R's
  ``stats::arima`` parameterization. Measured: SARIMA airline model on
  log(AirPassengers) went from 149 ms (Kalman with expanded params) to
  **14 ms** (Kalman with factored params) — at parity with R's 11 ms.

- **`timeseries/_arima_fit.py`: fix MA sign-convention bug in
  `_multiply_polynomials` for MA composition.** pystatistics' AR and MA
  use opposite sign conventions:
      AR:  e_t = y_t − Σ ar_i y_{t−i}      (polynomial 1 − Σ ar_i B^i)
      MA:  e_t = y_t − Σ ma_j e_{t−j}      (polynomial 1 + Σ ma_j B^j)
  The existing ``_multiply_polynomials`` was written for the AR
  convention; calling it for MA as the original non-factored
  implementation did was accidentally cancelled out by the expanded-
  form optimizer (which absorbed the sign freely). The factored path
  exposed the bug: airline-model fit converged to an inferior local
  minimum with NLL −240.5 instead of R's −244.7. Added a new
  ``_multiply_ma_polynomials`` with the correct sign; verified that
  fitting log(AirPassengers) now produces ma1=−0.402, sma1=−0.558,
  matching R's reported −0.402 / −0.557 to 3 decimals.

- **`timeseries/_arima_kalman.py` (NEW): state-space Kalman-filter
  exact ML for ARMA.** Replaces the O(n³) innovations algorithm with
  the Gardner–Harvey–Phillips (1980) state-space representation used
  by R's `stats::arima`. The Kalman forward pass is JIT-compiled with
  numba and exploits the companion-matrix structure of the ARMA
  transition matrix T (T[i, 0] = φ_{i+1}, T[i, i+1] = 1) so each
  step is O(r²) instead of O(r³). The stationary initial covariance
  P₀ solving `P = T P T' + RR'` is computed in a JIT'd fixed-point
  iteration, replacing scipy's `solve_discrete_lyapunov` which was
  150 µs per call on r=13 after the main loop was optimized. Falls
  back to a diffuse init (kappa=1e6, matching R's `makeARIMA`) if
  the fixed-point iteration fails to converge. Measured: SARIMA
  airline model on log(AirPassengers) went from 2100 ms (original
  innovations) → 220 ms (vectorized innovations) → 149 ms (Kalman +
  numba). Further improvement to R's 11 ms would require switching
  from expanded-MA parameterization to factored (ma1, sma1) so that
  scipy's L-BFGS-B optimizes a 2D surface instead of 13D — a
  separate refactor.

- **`pyproject.toml`: add `numba>=0.59` as a required dependency.**
  The Kalman filter inner loop is tight enough that pure-numpy
  per-call overhead on r <= 25 matrices dominates vs. R's Fortran
  implementation. Numba JIT closes the gap within a ~10x factor
  instead of ~200x. Torch remains optional (GPU backend only).

- **`timeseries/_arima_likelihood.py`: vectorize hot paths; use
  `scipy.signal.lfilter` for CSS residuals.** Three changes:
  (1) `arima_css_residuals` now calls `scipy.signal.lfilter(b, a, y)`
      instead of a double-nested Python loop. The difference equation
      `e[t] = y[t] - Σ ar_i y[t-i] - Σ ma_j e[t-j]` maps directly to
      lfilter's IIR form; lfilter runs in compiled C. Eliminates
      ~500k Python `np.dot` calls per SARIMA fit.
  (2) `_innovations_algorithm` inner j-sum is now a numpy dot product;
      numerical-guard clips (previously per-scalar `np.clip` / builtin
      `min`) are now Python comparisons or array-level `np.minimum`.
  (3) `exact_loglik` prediction-error inner loop is a dot product, and
      the log-likelihood aggregation is a single vectorized sum.
  Measured: SARIMA(0,1,1)(0,1,1)[12] on log(AirPassengers) went from
  2.1s to 0.22s per fit (~10× faster). Remaining gap to R's 11 ms is
  algorithmic — R uses a Kalman filter (O(n·s²)); the innovations
  algorithm is O(n³).

- **`ordinal/_likelihood.py`: vectorize `cumulative_negloglik`.** The
  negative log-likelihood was computed by a per-observation Python loop
  that made two `link.linkinv(np.atleast_1d(scalar))` calls per row to
  fill a `prob[i]` array one element at a time. On MASS::housing
  (n=1681) that was ~100k scalar `linkinv` calls per fit, each paying
  full numpy per-call overhead. The `_cumulative_probs_vectorized`
  helper right next to it already computes the full (n, K) category-
  probability matrix in one vectorized `linkinv` call — we now call it
  and index into it with `cat_probs[np.arange(n), y_codes]`. Measured:
  polr on MASS::housing went from 277 ms to 23 ms per fit (~12× faster,
  now at parity with R's MASS::polr at ~20 ms).

- **`timeseries/_arima_forecast.py`: fix latent off-by-one and
  uninitialized-memory bug in `_forecast_differenced`.** AR lag index
  was `n + k - i` (treats series as 1-indexed but `y_diff` is
  0-indexed), and `forecasts` was allocated with `np.empty`. The k=1,
  i=1 case read `forecasts[0]` before writing it. Worked only because
  fresh OS pages are zeroed; perturbations to allocator state (e.g.,
  from the SARIMA changes above) exposed it, producing forecasts of
  4e50 from latent garbage. Indexing corrected to `idx = n + k - i - 1`
  and the array is now `np.zeros`.


## 1.6.2

### Re-release of 1.6.1 fixes

**Why 1.6.2 exists:** the 1.6.1 release commit (`Release v1.6.1`) was
created after the source fixes were staged but before they were actually
committed to the branch. The CI `publish.yml` workflow then built the
PyPI package from the `v1.6.1` tag, which pointed at the version-bump
commit only — **the compiled wheel lacked the ARIMA / Gamma / var /
scipy fixes it was supposed to ship**. PyPI does not allow re-uploading
the same version number, so a patch version was the only clean path.
Users who installed `pystatistics==1.6.1` should upgrade to `1.6.2`.

The release script flow was adjusted in this cycle to ensure the release
commit carries all staged fixes; see Historical Notes in
`.release/CHECKLIST.md`.

### Fixed — content is the same as the 1.6.1 changelog entry

All fixes listed under 1.6.1 in `CHANGELOG.md` are now actually present
in the shipped wheel:

- **`timeseries.arima(method='CSS-ML')` silent fallback removed.** Raises
  `ConvergenceError` instead of silently returning CSS estimates labeled
  as CSS-ML.
- **`timeseries.arima` zero-parameter case.** Closed-form MLE for
  ARIMA(0,d,0); bypasses scipy's `nit=0 "ABNORMAL"` degenerate path.
- **`regression.GammaFamily.log_likelihood`** on non-positive dispersion
  returns explicit NaN instead of emitting `RuntimeWarning` and silently
  returning NaN from `np.log(negative)`.
- **`descriptive.var(n=1)`** short-circuits to NaN without triggering
  numpy's `Degrees of freedom <= 0` warning.
- **scipy 1.18 forward-compat**: removed deprecated `disp` option from
  `scipy.optimize.minimize` in mvnmle CPU and GPU backends.
- **mvnmle test suite** updated: `TestMissvalsDataset` uses EM
  explicitly; `TestDirectNonConvergence` codifies the fail-loud contract
  on the missvals pathological dataset.


## 1.6.1

### Fixed — Coding Bible Rule 1 violations (silent failures / degraded paths)

- **`timeseries.arima(method='CSS-ML')` silent fallback removed.** When ML
  refinement failed, the previous code emitted a `UserWarning` and silently
  returned CSS estimates while labeling the result "CSS" — despite the
  user having requested "CSS-ML". Now raises `ConvergenceError` with
  actionable guidance (use `method='CSS'`, adjust `tol`/`max_iter`).
  `pystatistics/timeseries/_arima_fit.py`.

- **`timeseries.arima` zero-parameter case.** For ARIMA(0,d,0) (and any
  configuration with p_eff = q_eff = 0), the code was calling scipy's
  `minimize` with a near-MLE start, which causes L-BFGS-B to exit with
  `nit=0, "ABNORMAL"` and trip the silent fallback path. The MLE is
  closed-form here (sample mean of the differenced series, or a constant
  if no mean) — no optimization is needed. Added an explicit closed-form
  branch that bypasses scipy. `pystatistics/timeseries/_arima_fit.py`.

- **`regression.GammaFamily.log_likelihood` on non-positive dispersion.**
  When the Gamma GLM fit perfectly (e.g. constant y), dispersion = dev/df
  came out as ≈ 0 or slightly negative, causing `np.log(rate)` to emit a
  `RuntimeWarning: invalid value encountered in log` and silently return
  NaN. Now validates dispersion > 0 explicitly and returns `nan` without
  triggering numpy's warning. `pystatistics/regression/families.py`.

- **`descriptive.var` of single-observation input.** For n=1 the sample
  variance is undefined. numpy correctly returns NaN but emits
  `RuntimeWarning: Degrees of freedom <= 0 for slice`. Added a short-circuit
  that returns NaN explicitly (matching R `var()`) without triggering the
  internal numpy warning. `pystatistics/descriptive/backends/cpu.py`.

### Fixed — scipy 1.18 forward-compatibility

- **Removed deprecated `disp` option from `scipy.optimize.minimize`** in
  mvnmle CPU and GPU backends. scipy 1.18 emits `DeprecationWarning` for
  `disp`/`iprint` on L-BFGS-B; the option is removed entirely (we do not
  print optimizer progress, so the default is fine).
  `pystatistics/mvnmle/backends/{cpu,gpu}.py`.

### Fixed — mvnmle test suite reflects code contract

- **`TestMissvalsDataset` now uses EM explicitly.** The `missvals` dataset
  (n=13, p=5, high missingness) is pathological for L-BFGS-B direct
  optimization: the likelihood surface is near-flat at this sample size,
  and direct does not converge. R's `mvnmle` uses an EM-equivalent
  algorithm — so the R-comparison tests must also use EM in pystatistics.
  EM converges to machine precision on this dataset and matches R exactly.

- **`TestEMMatchesDirect::test_missvals_*` removed.** These tests asserted
  that EM and direct estimates agree on missvals, but direct genuinely
  cannot converge on that dataset. Replaced with a single test that
  verifies EM matches R on missvals — the contract that actually holds.

- **Added `TestDirectNonConvergence`.** Codifies the explicit fail-loud
  contract: on pathological datasets like missvals, the direct optimizer
  must return `converged=False` rather than silently returning a
  meaningless answer. A future change which "fixes" direct non-convergence
  (e.g. by switching optimizers) must update this test deliberately.

### Test impact

- `pytest tests/` passes clean under
  `-W error::UserWarning -W error::RuntimeWarning -W error::DeprecationWarning`
  (was 20 warning-induced failures, now 0). Normal run: 2,301 passing,
  0 failing, 19 skipped.


## 1.6.0

### Summary

Major expansion of classical statistics coverage. Five new top-level modules
(`ordinal`, `multinomial`, `multivariate`, `timeseries`, `gam`), two new GLM
families (`Gamma`, `NegativeBinomial`), and reinforced "fail loud" numerical
policy. Adds ~650 new tests across all modules. Estimated coverage of standard
applied frequentist statistics goes from ~85% to ~95%.

### Added

#### GLM Families: Gamma and Negative Binomial

- **`GammaFamily`** — Gamma regression for positive continuous data (cost data,
  survival times, insurance claims). V(μ) = μ². Supports inverse (default), log,
  and identity links. Dispersion (1/shape) estimated from Pearson chi-squared /
  df_residual. Validates against R `stats::Gamma()`.

- **`NegativeBinomial`** — Negative binomial regression for overdispersed count
  data. V(μ) = μ + μ²/θ. Default link: log. Two usage modes:
  (1) `NegativeBinomial(theta=5)` for fixed θ via standard IRLS;
  (2) `fit(X, y, family='negative.binomial')` for automatic θ estimation via
  alternating profile likelihood, matching R `MASS::glm.nb()`.

#### Ordinal Regression Module

- **`polr(y, X, method='logistic')`** — Proportional odds (cumulative link)
  model matching R `MASS::polr()`. Supports logistic, probit, and complementary
  log-log links. Threshold ordering enforced via unconstrained parameterization
  (incremental exp-transform). L-BFGS-B with analytical gradient.

#### Multinomial Regression Module

- **`multinom(y, X)`** — Multinomial logit (softmax) regression matching R
  `nnet::multinom()`. Estimates (J-1) × p coefficient matrix with last class as
  reference. Log-sum-exp trick for numerical stability, L-BFGS-B with analytical
  gradient.

#### Multivariate Analysis Module

- **`pca(X, center=True, scale=False)`** — PCA via SVD matching R
  `stats::prcomp()`. Enforces R sign convention.

- **`factor_analysis(X, n_factors, rotation='varimax')`** — Maximum likelihood
  factor analysis matching R `stats::factanal()`. Varimax and promax rotations.

#### Time Series Module (Complete)

Full time series analysis framework. Validates against R packages `stats`,
`tseries`, and `forecast`.

- **ACF / PACF** — `acf(x)` and `pacf(x)` matching R `stats::acf()` / `stats::pacf()`.
- **Stationarity tests** — `adf_test(x)` and `kpss_test(x)` matching R
  `tseries::adf.test()` / `tseries::kpss.test()`.
- **Differencing** — `diff(x)` and `ndiffs(x)` matching R `base::diff()` /
  `forecast::ndiffs()`.
- **ETS** — `ets(y, model='ANN')` fitting 12 ETS model types matching R
  `forecast::ets()`. `forecast_ets()` with prediction intervals.
- **ARIMA / SARIMA** — `arima(y, order=(p,d,q), seasonal=(P,D,Q,m))` with CSS,
  ML, and CSS-ML methods matching R `stats::arima()`. `forecast_arima()` with
  MA(∞) psi weights. `auto_arima(y)` with stepwise or grid search matching R
  `forecast::auto.arima()`.
- **Decomposition** — `decompose(x, period)` and `stl(x, period)` matching R
  `stats::decompose()` / `stats::stl()`.

#### Generalized Additive Models Module

- **`gam(y, smooths=[s('x1')], smooth_data={...})`** — Penalized regression
  spline GAMs via P-IRLS matching R `mgcv::gam()`. Cubic regression splines and
  thin plate splines. GCV and REML smoothing parameter selection.
- **`s(var_name, k=10, bs='cr')`** — Smooth term specification matching `mgcv::s()`.

### Changed

- **GPU behavior enforces "fail loud" policy** — Explicit `backend='gpu'` calls
  on unsupported operations now raise `NotImplementedError` instead of silently
  falling back to CPU. Users who want automatic fallback should use
  `backend='auto'`.
- **GPU GLM tests require CUDA** — Skip condition narrowed from "any GPU" to
  "CUDA available" (MPS does not support `torch.linalg.lstsq`).

### Fixed

- **5 stale GPU hypothesis tests** — Tests expecting silent CPU fallback updated
  to expect `NotImplementedError`, matching v1.2.1 "fail loud" behavior.

### Tests

~650 new tests. Total: 2,275 fast + 13 slow = 2,288.

## 1.3.0

### Summary

Linux/NVIDIA validation on RTX 5070 Ti revealed that GPU backends for Monte
Carlo methods were stubs raising `NotImplementedError`. This release implements
working GPU acceleration for permutation tests and bootstrap resampling.

### Added

- Vectorized GPU backend for `permutation_test(backend='gpu')` using
  mean-difference statistic. Generates random permutations directly on GPU via
  `torch.rand` + `argsort` (random-key sorting), avoiding the CPU bottleneck of
  sequential `rng.permutation()` calls.
- Vectorized GPU backend for `boot(backend='gpu')` on simple mean statistic
  with 1-D data. Generates bootstrap index sets via `torch.randint` on GPU and
  computes all R means in a single vectorized pass.
- Chunked processing for permutation test GPU backend keeps VRAM usage under
  ~1 GB regardless of problem size.
- Auto-detection for both backends: permutation test detects mean-difference
  statistic, bootstrap detects simple mean. Non-vectorizable statistics fall
  back transparently to CPU.

### Changed

- `backend='auto'` now selects GPU when CUDA is available for both
  `permutation_test()` and `boot()` (was CPU-only before). Bootstrap
  auto-selection additionally requires the statistic to be vectorizable.
- GPU RNG (PyTorch) differs from CPU RNG (NumPy). P-values and bootstrap
  replicates are statistically equivalent but not bitwise identical across
  backends. Observed statistics (`t0`, `observed_stat`) remain identical since
  they are computed on the original data.

### Performance

- Permutation test GPU benchmarks (RTX 5070 Ti, R=50,000 permutations):
  - n=1,000: 5x speedup (CPU 1.4s, GPU 0.28s)
  - n=10,000: 23x speedup (CPU 6.7s, GPU 0.29s)
  - n=50,000: 23x speedup (CPU 33s, GPU 1.4s)

### Tests

- Updated backend name assertions to check for `'bootstrap'` / `'permutation'`
  rather than hardcoding `'cpu'`, reflecting GPU auto-selection behavior.

## 1.2.1

### Summary

Full codebase audit and refactor to enforce strict adherence to the project's
seven coding rules. Silent model switches are now hard errors, module files are
split to stay under 500 lines, and numerical guard comments document every
stability operation.

### Added

- `seed` parameter for `chisq_test()` and `fisher_test()` enabling reproducible
  Monte Carlo hypothesis tests when `simulate_p_value=True`.
- `# NUMERICAL GUARD:` comments on ~30 numerical stability operations (clipping,
  clamping, floors) documenting why each exists.

### Changed

- Split `regression/solution.py` into `_linear.py`, `_glm.py`, and
  `_formatting.py` (backward-compatible re-export shim maintained).
- Split `hypothesis/design.py` factory methods into `_design_factories.py`
  (classmethods still work via thin wrappers).
- All files now under 500 code lines.
- Removed dead code (`signaltonoise` try/except in Wilcoxon test).

### Breaking

- Silent model switches are now errors. Functions that previously fell back
  silently to ridge regularization, LSTSQ, or CPU backends now raise
  `NumericalError` or `RuntimeError` with descriptive messages suggesting
  alternatives. Affected call sites:
  - `mvnmle.mcar_test.regularized_inverse()` -- raises on ill-conditioned matrices
  - `mvnmle.mlest()` -- raises if EM encounters non-PD covariance
  - `mvnmle` parameter extraction -- raises instead of returning identity covariance
  - `regression.fit(backend='gpu')` -- raises on Cholesky failure (use `force=True`
    to proceed with LSTSQ, or `backend='cpu'` for QR)
  - `mixed.lmm()` / `mixed.glmm()` -- raises on singular random effects covariance
- `backend='gpu'` now errors when GPU is unavailable. Previously fell back to
  CPU silently. `backend='auto'` still falls back silently (it means "best
  available").
- GPU bootstrap/permutation raises `NotImplementedError`. These were silently
  running on CPU while reporting a GPU backend name. Now they honestly report
  that GPU acceleration is not yet implemented for these operations.

### Tests

- `tests/test_code_quality.py` -- LOC limit enforcement.
- `tests/regression/test_module_split.py` -- split integrity + named coefficients.
- `tests/hypothesis/test_design_split.py` -- factory split + seed reproducibility.
- `tests/mvnmle/test_no_silent_fallback.py` -- hard stop verification.

## 1.1.0

### Summary

Named coefficients bring R-style labeled output to regression and survival
models. Summary output for OLS and Cox PH now matches R's formatting, and a
22-analysis PBC clinical trial test suite validates end-to-end correctness.

### Added

- `names=` parameter for `fit()`, `coxph()`, and `discrete_time()` enabling
  labeled output matching R's style.
- `result.coef` dict property for accessing coefficients by variable name
  (e.g., `result.coef["albumin"]`) on `LinearSolution`, `GLMSolution`,
  `CoxSolution`, and `DiscreteTimeSolution`.
- `result.hr` dict property for accessing hazard ratios by name on
  `CoxSolution` and `DiscreteTimeSolution`.
- Intercept auto-detection: when `names` has one fewer element than columns in
  X, `"(Intercept)"` is prepended automatically.
- OLS summary now includes residual quantiles (Min, 1Q, Median, 3Q, Max) and
  overall F-statistic with p-value, matching R's `summary(lm())`.
- Cox PH summary now includes hazard ratio confidence intervals table
  (`exp(coef)`, `exp(-coef)`, `lower .95`, `upper .95`), matching R's
  `summary(coxph())`.

### Fixed

- Kaplan-Meier `summary()` printing literal `{ci_pct}` instead of the actual
  confidence level percentage (e.g., `95`).

### Tests

- PBC clinical trial integration test suite (`tests/test_pbc_analysis.py`) with
  22 end-to-end analyses on the Mayo Clinic PBC dataset.
- R cross-validation test (`tests/test_pbc_vs_r.py`) confirming all 15
  comparable analyses match R to `rtol=1e-10`.

## 1.0.2

### Summary

Initial stable release with all eight statistical modules complete and validated
against R.

### Added

- All 8 modules: regression, descriptive, hypothesis, montecarlo, survival,
  anova, mixed, mvnmle.
- CPU backends validated against R to `rtol=1e-10`.
- GPU backends validated against CPU per documented tolerance tiers.
