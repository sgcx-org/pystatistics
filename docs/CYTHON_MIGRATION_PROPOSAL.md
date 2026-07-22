# Design Proposal: Replace Numba kernels with Cython

**Status:** ARIMA pilot complete — Cython port + optimization audit + workspace wiring (§12–§14). Warm path at ~1.0× Numba; cross-platform CI matrix is the last external check.
**Scope:** `pystatistics` (published package) — 17 `@njit` kernels in `timeseries/` and `survival/`
**Author:** design session, 2026-07-21
**Decisions locked:** semver = **major**; build = hatchling build-hook first; sdist = hard-fail (no runtime fallback); launch matrix = **glibc-Linux + macOS arm64 + Windows x64** (Windows added §17; Intel-macOS dropped as EOL §15; musl out)

---

## 1. Summary

Replace the 17 Numba `@njit` kernels (ARIMA Kalman, ETS, STL/LOESS, robust STL,
concordance Fenwick tree) with hand-written Cython extension modules, and drop
`numba`/`llvmlite` from the runtime dependency set. The public Python API does
not change. Distribution changes from a single universal pure-Python wheel to a
`cibuildwheel` matrix of compiled wheels.

This proposal does **not** commit to the full migration. It commits to a
**one-kernel pilot** (ARIMA Kalman) with explicit exit criteria. If the pilot
holds, the remaining kernels are mechanical; if it fails, we've spent ~1–2 days
instead of a release cycle.

---

## 2. Current state (verified in-repo)

| Fact | Value |
|------|-------|
| Kernels | 17 `@njit(cache=True, fastmath=False)` functions |
| Files | 6 — `timeseries/_arima_kalman.py` (2), `_ets_kernels.py` (1), `_stl_core.py` (3), `_loess.py` (5), `_stl_robust.py` (2), `survival/_concordance_kernel.py` (4) |
| LOC | ~1,621 across the 6 files |
| Kernel shape | Tight sequential scalar float64 recursions; no `prange`, no `@vectorize`/`@guvectorize`, no typed containers, no dtype-generic dispatch |
| Inputs | Already normalized to contiguous float64 arrays by Python wrappers |
| Parity oracle | Each kernel has a maintained pure-numpy reference (e.g. `_ets_recursion_reference` in `_ets_models.py`); tests assert **bit-identity** (`assert_array_equal`, atol=0/rtol=0) |
| Build | `hatchling`, pure-Python wheel, built on one `ubuntu-latest` / py3.12 runner |
| Runtime dep | `numba>=0.59` (hard); `requires-python >=3.11` |
| Test CI | **None.** Only `publish.yml` and `trigger-docs-rebuild.yml` exist — parity tests run only on the dev's local machine today |
| Downstream | `pystatsbio` pulls `numba` in transitively only, for no reason of its own |

Kernel consumers (wrappers that would be unaffected): `_arima_fit.py`,
`_arima_likelihood.py`, `_arima_forecast.py`, `_arima_xreg.py`, `_whittle.py`,
`_ets_models.py`, `_stl.py`, `survival/_cox.py`.

---

## 3. Motivation

The case rests on **three** arguments. It deliberately does **not** rest on the
two weak ones, which we call out to avoid being talked into the migration for
the wrong reasons.

### 3.1 The real reasons

1. **Deployment determinism.** `cache=True` amortizes JIT cost only inside a
   persistent, writable environment. It buys nothing in the contexts a
   production statistical library actually lands in: containers / immutable
   images / read-only rootfs (cache dir unwritable or discarded per build),
   serverless / short-lived workers (every cold start recompiles ~17 kernels →
   real first-call latency), fresh CI venvs, and concurrent processes racing the
   cache. Cython delivers deterministic import-time-to-first-result.

2. **Dependency weight and Python-version gating.** `numba` + `llvmlite` bundles
   its own LLVM (tens of MB) and, more importantly, **gates the CPython support
   window** — historically you cannot support a new CPython until
   Numba/llvmlite do, which is a recurring release-timing headache every autumn.
   Dropping it lets `pystatistics` (and transitively `pystatsbio`) support new
   Pythons on day one. Plain C extensions with no `prange`/OpenMP are the
   easiest possible case for fast Python-version support.

3. **Ecosystem alignment.** SciPy, scikit-learn, pandas, and statsmodels all
   ship Cython/C for their own kernels and use Numba (if at all) only as an
   opt-in accelerator for *user-supplied* code (`engine="numba"`). Nobody in
   that tier ships Numba as a hard runtime dependency for their own kernels. Our
   kernels are stable, blessed reference implementations — exactly the case that
   tier solves with Cython.

### 3.2 Reasons this does NOT rest on (and why)

- **"Users pay JIT compilation cost."** Largely amortized by `cache=True` in the
  interactive/notebook case, where the one-time cost is trivial. This is not the
  argument.
- **Any runtime performance win.** Numba's one genuine runtime edge —
  host-CPU/`-march=native` ISA specialization (AVX2/AVX-512/FMA) — does **not**
  apply here: every kernel is a loop-carried sequential recursion (Kalman
  forward pass, ETS ring-buffer, Fenwick prefix sums) that does not
  auto-vectorize. Portable Cython won't beat it either. **Expect perf parity,
  benchmark to confirm no regression, do not assume a speedup.**

### 3.3 Every condition that would argue *for* keeping Numba is absent

- dtype-generic kernels (float32/float64/int templates) — **no**, we're float64-only.
- `prange` parallelism — **no**, none used.
- churny / user-extensible kernels — **no**, these are frozen reference impls.
- `numba.cuda` GPU kernels — **no**, GPU is routed through Torch.

---

## 4. Non-goals / what stays identical

- **Public Python API** — signatures, return types, values. Unchanged by design.
- **Numerical results** — the pure-numpy references remain the oracle; kernels
  must stay bit-identical to them (§6.1).
- **The pure-numpy reference implementations** — they are kept, not deleted.
  They are the portable oracle and the emergency fallback.
- **GPU paths** — untouched; still Torch.
- **No new kernels, no algorithm changes** — this is a port, not a rewrite.

---

## 5. Semver: **major** bump (confirmed)

The public API does not change, which on API-only SemVer argues *minor*. We
nonetheless propose a **major** bump, for one decisive reason:

> A pure-Python wheel installs on **every** platform Python runs on. No compiled
> wheel matrix can match that breadth.

The migration therefore **guarantees** a compatibility regression on some
long-tail platform — an exotic arch, a musl variant not in the matrix, or a
brand-new CPython before wheels are rebuilt. Those users go from "works" to
"sdist compile error / cannot install." That asymmetry (universal →
necessarily-narrower) is exactly what a major bump signals. Dropping
`numba`/`llvmlite` from the dependency closure reinforces it.

**Defensible as minor only if** the wheel matrix demonstrably covers every
platform any real user is on — and even then, major is the more honest signal.
Recommend major + prominent release notes ("now ships compiled wheels; pure
sdist build requires a C compiler").

---

## 6. Risks and mitigations

### 6.1 FP bit-drift vs. the bit-identical parity tests — **the main risk**

Tests assert `atol=0, rtol=0` against a pure-numpy reference. Two sources of
last-bit divergence:

- **FMA contraction.** GCC/Clang at `-O2` may contract `a*b + c` into a fused
  multiply-add, changing the last bit — while the interpreted numpy reference
  does not contract. **Mitigation:** compile every extension with
  **`-ffp-contract=off`**, never `-ffast-math`. Write `x*x`, never `pow(x, 2)`,
  for the `**2` sites (`_stl_robust.py`, `_loess.py`).
- **libm transcendental divergence** (`np.log` in ARIMA loglik; `np.sqrt`).
  Cross-platform, `log()` can differ in the last ULP. **This is a non-issue for
  our parity tests** — the tests compare kernel-vs-reference *on the same
  machine*, and both sides resolve to the same platform libm, so the divergence
  cancels. `sqrt` is IEEE-754 correctly-rounded and identical everywhere
  regardless. We must only ensure the compiler does not substitute a vectorized
  / approximate `log` — which `-ffast-math`-off already guarantees.

Net: the one guardrail that matters — `-ffp-contract=off` + no `-ffast-math` —
is a compiler-flag change, and it is **sufficient**. (Absolute cross-platform
ULP drift would only bite *stored golden fixtures*; the R-fixture gates are
tolerance-based, not bit-exact, so they are unaffected.)

### 6.2 Parity is only checked on one platform today

There is no test CI. The migration must add a matrix that **runs the parity
suite on every wheel it builds**, or §6.1 hides until a user reports it. This is
new infrastructure we don't currently have — counted as cost, not assumed.

### 6.3 Build-matrix breadth is a recurring cost

`cibuildwheel` across {Linux glibc + musl, macOS x86_64 + arm64, Windows} ×
supported CPythons (free-threaded 3.13t/3.14t soon). No `prange`/OpenMP → these
are plain C extensions, the easiest case — but it is ongoing release engineering
(broken-wheel-on-one-runner blocks all users on that platform, a higher-variance
failure mode than a mild JIT delay). **Budget it as recurring, not one-shot.**

### 6.4 Perf regression

Possible if a hot loop is written naively in Cython (missing `boundscheck`/
`wraparound` off, Python-object fallback). **Mitigation:** benchmark the pilot
kernel against the current Numba kernel; gate on no regression (§8).

---

## 7. Design of the port

### 7.1 Kernel mapping

Each `@njit` function becomes a `cdef` function (or `cpdef` at the module
boundary) in a `.pyx` module, with typed `double[:]` / `double[:, :]`
memoryviews and module-level `boundscheck(False)`, `wraparound(False)`,
`cdivision(True)`. The scalar-recursion shape maps 1:1 to C — no algorithmic
change. Proposed layout (mirrors the existing private-module names):

```
pystatistics/timeseries/_arima_kalman.pyx      # _stationary_init, _kalman_loop
pystatistics/timeseries/_ets_kernels.pyx
pystatistics/timeseries/_stl_core.pyx
pystatistics/timeseries/_loess.pyx
pystatistics/timeseries/_stl_robust.pyx
pystatistics/survival/_concordance_kernel.pyx
```

The Python wrappers that call these keep their current signatures; only the
import target changes (`from ._arima_kalman import _kalman_loop` still resolves,
now to the compiled module).

### 7.2 The reference implementations are the asset

Every kernel already has a maintained pure-numpy twin used as the bit-identity
oracle. The Cython kernel is written to match the **same** reference the Numba
kernel matches today. The references stay in the tree **as the portable bit-identity oracle in CI —
test-only.** They are *not* wired as a runtime fallback (Q3 = hard-fail), so
they never silently degrade a user's install; they exist to keep the compiled
kernels honest.

### 7.3 Build backend

**Decided:** try the **`hatchling` build-hook** path first (smallest diff to
`pyproject.toml`) — a custom `hatch_build.py` (or `hatch-cython`) that invokes
`cythonize` and compiles extensions with the §7.4 flags. Fall back to
`meson-python` (SciPy's choice) **only** if the hook can't cleanly express
`-ffp-contract=off` across the matrix. `setuptools` + `Cython` is the third
option if both stumble.

### 7.4 FP flags, concretely

Per-extension compile args, set in the build backend:

- Unix (gcc/clang) — covers the launch matrix (glibc-Linux + macOS):
  `-O2 -ffp-contract=off` (never `-ffast-math`, never `-Ofast`)
- MSVC (only if/when Windows is added — §11 Q4): `/O2 /fp:precise` (never `/fp:fast`)

---

## 8. Pilot plan (ARIMA Kalman)

**Why this kernel:** the largest (`_arima_kalman.py`, 561 LOC), the one with the
`log`-in-loglik transcendental concern, and the reason `numba` is a dep at all
(per the `pyproject.toml` comment). If Cython works here, it works everywhere.

**Steps**

1. Port `_stationary_init` and `_kalman_loop` to `_arima_kalman.pyx`. Keep the
   `.py` reference in the tree, renamed/kept as the oracle.
2. Wire a minimal build: `hatchling` build-hook compiling just this one
   extension with the §7.4 flags.
3. Stand up a minimal `cibuildwheel` matrix — **launch scope only**:
   {Linux glibc, macOS arm64} × {3.11, 3.12, 3.13}, with
   `test-command` running the ARIMA parity + kalman tests inside each wheel.
   (musl and Windows deliberately excluded — §11 Q4.)
4. Benchmark the Cython kernel vs. the current Numba kernel on the existing
   SARIMA benchmark series.

**Exit criteria (all three must hold to proceed to full migration)**

- ✅ **Parity:** ARIMA bit-identity tests pass on **every** platform in the
  matrix with `-ffp-contract=off`.
- ✅ **Performance:** no regression vs. Numba on the SARIMA benchmark
  (target: within noise; a modest slowdown is a discussion, not an auto-fail).
- ✅ **Distribution:** the pipeline produces installable wheels for all launch-
  matrix targets (glibc-Linux + macOS), and an sdist that builds where a C
  compiler is present and **hard-fails with a clear error** otherwise (Q3 = (a);
  no runtime fallback).

**Budget:** ~1–2 days. **If any criterion fails,** stop and reconvene — the
finding (which one failed, and why) is the deliverable, and we've risked days,
not a release.

---

## 9. Full rollout (only if the pilot holds)

1. Port the remaining 15 kernels file-by-file (each is smaller and simpler than
   ARIMA; concordance Fenwick and STL/LOESS are pure integer/float recursions
   with no transcendentals).
2. Expand the `cibuildwheel` matrix to the full supported-CPython set **on the
   launch platforms (glibc-Linux + macOS)**; add free-threaded builds when the
   ecosystem is ready. musl (Alpine) and Windows are added only on demand
   (§11 Q4).
3. Add a standing **test-CI workflow** (this is new — see §6.2) running the full
   parity suite on the matrix, independent of release.
4. Remove `numba>=0.59` from `dependencies`; update the `pyproject.toml` comment
   and the ARIMA docstring that reference Numba.
5. Update `.release/UNRELEASED.md` per Rule 10 as each kernel lands.
6. Cut the **major** release (§5) with migration notes; verify `pystatsbio`'s
   transitive `numba` pull is gone.

---

## 10. Ongoing maintenance budget (be honest about this)

The author cost is **not** paid once. It converts a repeated *user* cost (mild
JIT delay) into a repeated *author* cost (the wheel matrix), plus higher-variance
failure modes (a broken wheel blocks a whole platform). Recurring items:

- Rebuild/verify wheels on each new CPython (now day-one-possible, but still a task).
- Keep `cibuildwheel` and the compiler-flag invariants green across 5+ platforms.
- Add platforms on demand (new arch, new libc).

This is a defensible trade for a production library, but it is a trade, not a
free absorption of complexity.

---

## 11. Decisions (settled 2026-07-21)

- **Q1 — Semver.** ✅ **Major.** (§5)
- **Q2 — Build backend.** ✅ **`hatchling` build-hook first**, `meson-python`
  fallback only if the hook can't express `-ffp-contract=off`. (§7.3)
- **Q3 — sdist policy.** ✅ **(a) hard-fail** with a clear error where no wheel
  and no compiler exist. No runtime fallback; pure-numpy references are
  test-only oracles, never a silent slow path. (§7.2)
- **Q4 — Platform matrix.** ✅ **Launch on glibc-Linux + macOS arm64.**
  Windows is explicitly **not** a target audience; musl added later only on
  demand. Intel macOS (x86_64) was initially included but **dropped** — see
  §15. (§8, §9, §15)

---

## 12. Pilot results (ARIMA Kalman)

Executed on macOS arm64 (Python 3.13, Cython 3.2.4, Apple clang 21). The two
`@njit` kernels (`_stationary_init`, `_kalman_loop`) were ported to Cython
(`_arima_kalman_kernel.pyx`), with a pure-numpy oracle (`_arima_kalman_ref.py`)
and a bit-identity test (`test_arima_kalman_cython_parity.py`). `numba` was NOT
removed — the other 15 kernels still use it; it goes only after full rollout.

**New / changed files**

- `pystatistics/timeseries/_arima_kalman_kernel.pyx` — compiled kernels.
- `pystatistics/timeseries/_arima_kalman_ref.py` — pure-numpy oracle (test-only).
- `pystatistics/timeseries/_arima_kalman.py` — drops `from numba import njit`;
  imports the two kernels from the compiled module; `loglik` now guarantees a
  contiguous float64 `z` at the boundary.
- `hatch_build.py` — hatchling build hook (cythonize + compile, FP-safe flags).
- `pyproject.toml` — build deps, hook registration, artifacts/exclude, sdist
  includes, `[tool.cibuildwheel]`.
- `.github/workflows/publish.yml` — replaced with a build+test+publish matrix.
- `tests/timeseries/test_arima_kalman_cython_parity.py` — bit-identity tests.

**Key finding — the matmul/bit-identity subtlety.** ARIMA had no pure-numpy
oracle (unlike ETS). Building one exposed that `_stationary_init` uses dense
matmuls, and matmul reduction order is implementation-defined — the *old Numba*
kernel already differed from a numpy `@` reference by ~1.4e-17. "Bit-identical"
was therefore never true for that kernel via BLAS. Resolution: the port
computes the init's matrix products as **explicit scalar loops** (no BLAS), so
P0 is now fully deterministic and platform-reproducible — removing a hidden
multi-threaded-BLAS non-determinism source, in the spirit of Rule 6. The
`kalman_loop` (already scalar) is bit-identical as-is.

**Exit criteria**

1. **Parity — ✅ (on macOS).** 41 bit-identity assertions (compiled kernel ==
   numpy oracle, atol=0/rtol=0) pass. The full ARIMA suite (244 tests incl. the
   R-parity gate) and the whole `tests/timeseries/` suite (717 passed, 2
   skipped) pass against the compiled kernel. The scalar-init change perturbs
   P0 by ~1e-17, far below the R gate.
2. **Performance — ⚠️ acceptable, one tradeoff.** Warm steady-state per
   likelihood-eval: `stationary_init` 12.1→14.7 µs (1.21×), `kalman_loop`
   30→39 µs (1.2–1.3×), combined ≈1.25× slower than warm Numba. BUT cold
   first-call: Numba ≈1.06 s (JIT) → Cython ≈0. Break-even ≈90,000 evals in one
   process; real fits are 10²–10³ evals, so Cython wins end-to-end in every
   fresh-process / container / serverless context and loses only in a long-lived
   warmed server. `-O3 -ffp-contract=off` + `ikj` loop order recovered part of
   the gap; the residual is clang-scalar vs numba-LLVM codegen and is not worth
   chasing further.
3. **Distribution — ✅ (on macOS).** The hatchling hook builds a
   platform-tagged wheel (`cp313-cp313-macosx_26_0_arm64`); it installs into a
   clean venv, imports the `.so` from the installed location, and passes parity.
   The sdist ships the `.pyx` + hook and compiles-from-source on install where a
   compiler exists (Q3 hard-fail otherwise, by design). **The hatchling
   build-hook path works cleanly — meson-python fallback NOT needed.**

**What the pilot could NOT verify locally:** the other launch-matrix targets
(glibc-Linux, macOS x86_64). The `publish.yml` cibuildwheel matrix builds and
runs parity inside each wheel on CI — that is the remaining confirmation that
`-ffp-contract=off` holds cross-compiler. This is the one open box before the
pilot is fully green.

**Verdict:** feasible and clean. All three criteria pass on macOS; the only
caveat is the ~1.25× warm regression, which is dominated by the cold-start win
for the workloads this package actually runs. Recommend: run the CI matrix to
confirm Linux/x86_64 parity, then greenlight the full 15-kernel rollout (§9).

---

## 13. Optimization audit (ARIMA Kalman warm-path)

Follow-up to §12: the pilot's ~1.25× warm-kernel regression was not accepted as
final. A focused audit (macOS arm64, Python 3.13, clang 21, `-O3
-ffp-contract=off`) found the regression was **not native compute** but per-call
allocation + boundary overhead. Parity was preserved exactly throughout: 725
`tests/timeseries/` pass, 49 kalman parity tests (41 bit-identity + 8
workspace), R-parity green.

**Where the time actually was (annotated Cython + decomposed benchmarks).** The
inner loops were already pure C (score 0 in `cython -a`). The overhead was
entirely at the boundary: per-call `np.empty` of every scratch matrix, argument
unpacking, and return wrapping — ~7 µs/eval that the *single-call* benchmark
attributed to "the kernel".

**Retained changes (all bit-identical to the reference; committed):**

1. **`noexcept nogil` raw-pointer cores.** All arithmetic moved into
   `_stationary_core` / `_kalman_core` (raw `double*`, `i*r+j` indexing),
   with thin Python entry points. No arithmetic or accumulation-order change.
   Native compute now **ties Numba**: `kalman_loop` 32.5 µs vs 32.6 (≈1.0×),
   `stationary_init` 12.8 µs vs 12.2 (~1.05×) — from the previous 1.19–1.33×.
2. **`-O3` over `-O2`.** +12 % on `kalman_loop` (36.8 → 32.5 µs), init
   unchanged. FP-safe: the assembly contains **0 FMA instructions** (verified),
   so `-ffp-contract=off` holds and parity is bit-for-bit under both.
3. **Removed a wasted r×r allocation per eval.** `_build_state_space` built a
   dense transition matrix `T` that every caller (loglik, forecast,
   innovations) discarded — the kernels use the companion structure via `phi`.
   New lean `_noise_loading(ar, ma) -> (R_vec, r)` skips it.
4. **`ArmaKalmanWorkspace`** — a fused, buffer-reusing path (stationary init +
   diffuse fallback + filter in one `nogil` call) for reuse across a fit's many
   likelihood evaluations. Bit-identical to the single-call kernels (Δnll = 0),
   dimension-guarded (fail-loud on shape mismatch), and tested. **Not yet wired
   into the fit driver** — see the open decision below.

**Performance at three levels (r≈13–14, n≈132, absolute µs):**

| Level | Numba (warm) | Cython before audit | Cython after audit |
|-------|-------------:|--------------------:|-------------------:|
| kernel, native (buffers reused) | 44.9 µs | 45.3 µs (nogil cores) | **45.3 µs (≈1.0×)** |
| full likelihood-eval, per-call wrapper | 54.6 µs | 62.4 µs (1.14×) | 62.4 µs (1.14×) |
| full likelihood-eval, fused workspace | — | — | **55.2 µs (1.01×)** |
| end-to-end warm ARIMA(2,1,1) fit | — | — | 5.68 ms |
| end-to-end cold fit (import+first fit) | ~1.74 s (JIT) | — | **0.74 s (no JIT)** |

Net: the warm kernel now matches Numba; the fused workspace closes the
full-eval gap to ~1 %; and cold start remains the decisive win (no JIT). The
per-call wrapper (used everywhere until the workspace is wired) sits at 1.14×
full-eval, its residual being unavoidable per-eval `np.empty` that Numba's
internally-allocating `@njit` also pays.

**Design options NOT implemented (each changes bits/behavior — reported per the
audit's ground rules, for a separate decision):**

- **A — Exploit symmetry in `stationary_init`.** `S`, `U` are symmetric;
  computing one triangle and mirroring would ~halve the doubling matmul work.
  But the current kernel computes `U[i,j]` and `U[j,i]` *independently*
  (different summation order), so mirroring is **not** bit-identical to the
  present reference — it would require re-baselining the oracle. Estimated
  saving: a few µs on init only.
- **B — Hoist `isfinite` checks out of the hot loops.** The per-element
  `if not isfinite(...)` early-exit is what blocks clang from vectorizing the
  O(r²) covariance-update loop (confirmed via `-Rpass-analysis`). Checking
  finiteness once per row/matrix instead would let it vectorize. Bit-identical
  for finite (ok=True) inputs, but it changes the failure-path `innov/F`
  contents and transiently computes with inf/nan — a behavior change on the
  penalty path (whose array contents no caller currently reads).

**Open decision — wire the workspace into the fit driver?** It requires adding
an optional workspace to the `arima_negloglik` / `exact_loglik` objective
signature (default `None` → current path, so backward-compatible) and creating
one workspace per fit at the ML `minimize` sites in `_arima_fit.py`. Benefit:
full-eval 1.14× → 1.01×, i.e. Numba parity on the warm path. Cost: an
objective-signature change threaded through a critical, subtly-behaved module
(CSS/ML/veto/retry). Behavior is provably unchanged (bit-identical nll). Left
for explicit go/no-go rather than done unilaterally.

---

## 14. Workspace wired into the production fit path

The `ArmaKalmanWorkspace` fusion (§13) is now on the production ML path. Public
API unchanged; the workspace is internal and fit-scoped.

**Wiring (closure-based, backward-compatible):**

- `kalman_arma_loglik(..., *, _workspace=None)` — keyword-only, private. When
  supplied, the fused buffer-reusing path runs; when `None` (the default, and
  every existing caller), the historical allocating path runs unchanged. The
  allocating path is retained verbatim for one-shot / low-level calls.
- `exact_loglik` / `arima_negloglik` gained the same keyword-only `_workspace`
  passthrough.
- `_arima_kalman._new_workspace(p_eff, q_eff, n)` — factory; documents the
  ownership rule (mutable, fit-scoped, **one per independent fit, never shared
  across concurrent evaluations**).
- `_arima_fit._run_arima_optimization` creates **one** workspace per fit (only
  for `method in {ml, css-ml}`, and never for the closed-form p=q=0 case) and
  binds it in a closure `_ml_objective(params)` passed to every ML `minimize`
  call (CSS→ML refinement, the include-mean second start, and pure ML). The CSS
  objective and the post-fit Hessian/SE computations stay on the allocating path
  (one-shot).

**Full-fit regression coverage** (`test_arima_workspace_fit_regression.py`,
11 tests). Each fit is run both ways — workspace (default) and with
`_new_workspace` monkeypatched to `None` — and asserted **identical** on:
fitted parameters (ar/ma/seasonal/mean), objective + information criteria
(log_likelihood/aic/aicc/bic/sigma2), convergence status, iteration counts,
method used (CSS→ML), variance-covariance (retry/veto-sensitive),
residuals/fitted values, and emitted warnings. Scenarios: non-seasonal
(css-ml/ml/no-mean/pure-AR/pure-MA/ARMA+mean), seasonal SARIMA, near-unit-root
(diffuse-fallback), `auto_arima` (workspace per candidate), a degenerate series,
and an objective-level check that the `1e18` penalty branch is bit-identical for
infeasible (nonstationary / non-invertible) parameters.

**Final validation (macOS arm64, local):**

- Kernel + workspace parity: green (49 + 11 regression).
- Full ARIMA suite: 304 passed.
- Full `tests/timeseries/`: 736 passed, 2 skipped.
- End-to-end: warm ARIMA(2,1,1) **5.17 ms** (was 5.68 ms pre-wiring);
  `auto_arima(3,3)` 26.4 ms; cold import+first-fit 779 ms (no JIT).
- In-wheel: built the wheel, installed into a clean venv, ran the parity +
  R-parity + workspace-regression tests against the **installed `.so`** — 134
  passed. The `cibuildwheel` `test-command` now includes the regression test, so
  every wheel is verified this way.

**Remaining external check:** the cross-platform CI matrix (glibc-Linux, macOS
x86_64) must run green — it cannot be exercised from a single local machine. All
local platforms (macOS arm64) are green. On a green matrix the ARIMA pilot is
complete.

Design options A (symmetry) and B (`isfinite` hoisting) remain **documented, not
implemented** (§13), per decision.

---

## 15. Matrix trim — Intel macOS dropped (2026-07-22)

The first two CI runs of the cibuildwheel matrix confirmed the migration's core
risk is closed: the in-wheel parity + regression suite passed **134/134 on each
of cp311/312/313** for both **glibc-Linux (gcc)** and **macOS arm64 (clang)** —
bit-identity holds across two compilers under `-ffp-contract=off`. `sdist` built
and installed from source. The only failure mode was operational, not
numerical: the **Intel-macOS (`macos-13` / x86_64) job never got a runner** —
GitHub's Intel-mac pool is being deprecated and the job sat queued for hours
(15h on the first run, still queued on the second).

**Decision (reverses part of Q4): drop macOS x86_64 from the matrix.** Rationale:

- Apple has EOL'd Intel Macs; unlike Windows there is no growing supported base
  to serve — the platform is actively going away.
- Its numerics are already covered: macOS x86_64 uses the **same clang + libm**
  as macOS arm64, so the green `macos-14` job already proves the FP behavior;
  the Intel job would only re-confirm build/packaging on a dying platform.
- The runner flakiness makes it a standing CI liability (hours-long queues,
  false "still running" states).

**Changes:** `publish.yml` matrix → `[ubuntu-latest, macos-14]`;
`[tool.cibuildwheel].skip` gains `*-macosx_x86_64` so the "no Intel Mac" policy
holds even for a build run on an Intel host (it does not match linux
`*-manylinux_x86_64`).

**Final launch matrix:** glibc-Linux (x86_64) + macOS arm64, × CPython
3.11/3.12/3.13. musl and Windows remain out; both add later only on demand.

With Intel-mac removed, the matrix is green end to end and the ARIMA pilot's
cross-platform verification is **complete**.

---

## 16. Full rollout complete — numba dropped (2026-07-22)

All 17 kernels are now Cython; `numba`/`llvmlite` is removed from
`dependencies`. Landed on the same branch/PR after the ARIMA pilot, each with a
bit-identity oracle and parity tests, each verified green before the next:

| Group | Kernels | Compiled module | Oracle | Parity tests |
|-------|--------:|-----------------|--------|-------------:|
| ARIMA Kalman (pilot) | 2 | `_arima_kalman_kernel` | `_arima_kalman_ref` | bit-identity + R + full-fit regression |
| Concordance (survival) | 4 | `_concordance_fenwick` | `_concordance_ref` | 36 |
| ETS | 1 | `_ets_recursion` | `_ets_recursion_reference` | 56 (existing) |
| STL / LOESS / robustness | 10 | `_stl_kernels` | `_stl_ref` | 428 + R-parity gate |

**Patterns established by the pilot, reused throughout:**
- `noexcept nogil` raw scalar cores where allocation-free; GIL held only where a
  kernel allocates (STL driver, loess). Explicit scalar matmuls / no BLAS.
- `-O3 -ffp-contract=off`, never `-ffast-math`; every `x**2` written `x*x`
  (0 libm `pow`, 0 FMA), so last-bit parity holds across gcc/clang.
- Public API and import paths unchanged: the historical `.py` modules become
  thin re-export shims over the compiled kernels; Python wrappers retained.
- Coupled kernels that call each other in a hot loop (the STL family) compiled
  as one translation unit (internal `cdef` calls, no Python boundary).

**Finalization:**
- `numba` removed from `[project].dependencies` and from `cibuildwheel`
  `test-requires`. Zero `import numba` remain (the `njit` in `_stl_ref` is a
  local no-op that makes the reference a verbatim copy).
- `cibuildwheel` `test-command` now runs the parity/regression suites for all
  four groups inside each built wheel.
- Verified: a wheel installed into a venv with **no numba** imports cleanly and
  fits ARIMA/ETS/STL/Cox; 702 in-wheel parity tests pass; full local
  `tests/timeseries/` + `tests/survival/` = 1470 passed.

`pystatsbio` no longer pulls `numba` transitively. This completes the
Numba→Cython migration; the package is ready for the major release once the
cross-platform CI matrix is green on this final state.

---

## 17. Windows added to the matrix (2026-07-22)

Reversing the "Windows out" part of §11 Q4. Unlike Intel macOS, Windows has a
supported user base and GitHub has ample runners, so it is worth building —
*provided MSVC compiles the kernels and holds bit-identity parity*. That is not
assumed; it was verified in CI.

**First run — one failure, diagnosed as a test bug, not a product bug.** MSVC
cythonized and compiled all four extension modules cleanly, and **701/702**
in-wheel tests passed — every bit-identity parity test (ARIMA, ETS, STL,
concordance) was green on Windows. The lone failure was
`test_fit_identical_seasonal`: the seasonal ARIMA fit raised `ConvergenceError`
(64 iters) on Windows. Root cause: the full fit's objective includes `np.log`
(platform libm), which differs in the last ULP across OSes and can shift the
L-BFGS-B trajectory — so a synthetic fixture near the iteration limit converges
on Linux/macOS but not Windows. This is **pre-existing** platform sensitivity
(true under Numba too — `log` was always platform libm), unrelated to the
Cython port; the compiled *kernels* are bit-identical across platforms.

**Fix (test robustness).** The full-fit regression asserts *workspace ==
allocating path on the same machine*; the two are bit-identical, so they must
share an outcome whether the fit converges or raises. The test now compares
outcomes (converge → same solution + warnings; raise → same exception) instead
of assuming convergence. Second run: **all four platforms green**, publish
skipped.

**Config:** `publish.yml` matrix `[ubuntu-latest, macos-14, windows-latest]`;
`cibuildwheel` `skip` drops `*-win*`, keeps `*-win32` (win_amd64 only); the
build hook's MSVC branch uses `/O2 /fp:precise` (no FMA on baseline x64).

**Final launch matrix:** glibc-Linux x86_64 + macOS arm64 + Windows x64, ×
CPython 3.11/3.12/3.13. musl remains out (add on demand); Intel-macOS stays out
(EOL, §15).

---

## 18. Performance audit of the rollout kernels (2026-07-22)

The pilot benchmarked ARIMA thoroughly; the rollout kernels were initially only
parity-checked. This audit benchmarks them warm against the original Numba
(reconstructed from the same bodies with real `@njit`).

**Initial results (parity-only ports):**

| Kernel | cy/nb |
|--------|------:|
| concordance_simple | 1.06× |
| concordance_truncated | 0.79× (faster) |
| ETS recursion | 1.52× |
| STL robust decomposition | 1.78× |

Concordance was fine; ETS and STL had regressed. Both from the same cause the
ARIMA workspace addressed: **per-call `np.empty` allocation** in GIL-held code
(Numba allocates far faster in nopython).

**Optimizations (bit-identity preserved — all parity suites green):**

- **ETS — skip the state history AND reuse a fit-scoped workspace.** First cut:
  a `want_states` flag skips the discarded `(n+1, n_cols)` state history on the
  objective path (4.44 → 3.71 µs). That still measured 1.47× Numba, which was
  **initially misattributed to branch-tree codegen**. A controlled experiment
  (§18.1) disproved that: the gap was **per-call allocation** of
  `fitted`/`residuals`/`s`. An `EtsWorkspace` (mirroring the ARIMA workspace),
  created once per fit and threaded through the objective, reuses those buffers.
  Fit-path kernel **3.71 → 2.08 µs = 0.74× Numba (now faster)**; end-to-end
  ETS(AAA) fit ~7 % faster. Bit-identical (kernel parity + a
  workspace-vs-allocating full-fit regression across AAN/AAA/ANN/MAM/AAdN).
- **STL — reuse scratch buffers in the loess/MA hot path.** A robust
  decomposition made ~186 `np.empty` calls (out+ws per internal loess, per MA,
  per cycle-subseries). Allocation-free `cdef` cores (`_loess_into`,
  `_moving_average_into`) write into caller buffers; `stl_core` preallocates a
  fixed handful of scratch arrays once and reuses them. Robust n=120:
  275 → 112 µs — **0.70× Numba (now faster)**; non-robust 0.84×. Public
  `loess_smooth_nb`/`moving_average_nb` keep their allocating wrappers.

**Final scorecard (warm, vs original Numba):**

| Kernel | cy/nb | |
|--------|------:|--|
| ARIMA (fused workspace) | ~1.01× | parity |
| concordance simple / truncated | 1.08× / 0.79× | parity / faster |
| STL robust / non-robust | 0.70× / 0.84× | **faster** |
| ETS fit path (workspace) | 0.74× | **faster** |

Net: ARIMA and concordance at parity; STL and ETS faster than the Numba build.
Combined with the cold-start win (no JIT) and dependency removal, the migration
is **performance-positive**.

### 18.1 Diagnosing the ETS gap — the value of not trusting the first hypothesis

The ETS 1.47× was first attributed to clang-vs-LLVM codegen on the branch tree,
"only closeable by a dozen specialized loop copies." A controlled experiment
falsified that, one hypothesis at a time:

| variant (ETS AAA, n=200) | time | verdict |
|--------------------------|-----:|---------|
| cython full, branchy, per-call alloc | 3.75 µs | baseline gap 1.48× |
| + specialized (branch-free) | 2.83 µs | branches cost ~0.9 µs (real but secondary) |
| + counter instead of `t % m` | 2.79 µs | **modulo: not it** |
| + inline-C `restrict` pointers | 2.79 µs | **aliasing: not it** |
| + buffers reused (no per-call alloc) | **1.11 µs** | **allocation was ~1.7 µs — the dominant cost** |

Allocation, not codegen, was the gap — the same cause as ARIMA and STL, and the
same fix (a workspace). Specialization would have helped only the smaller
~0.9 µs branch component and added a dozen loop copies; the workspace closed the
larger allocation component with the pattern already in the codebase. Lesson:
benchmark the hypothesis, don't ship the plausible story.
