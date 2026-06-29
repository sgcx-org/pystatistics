# GLM IRLS inner-solve design space on Apple Silicon (MPS) float32 — a mapping study

**Status:** research note / design input. **Does not change library behaviour.**
A follow-up production-implementation task (scoped in §8) would change
`pystatistics/regression/backends/gpu_glm.py`; this note only maps the space and
recommends a policy.

**Governing rule (Prime Directive / A6).** The exact request is honoured, or it
fails loud. Never silently wrong; never slower than R on CPU for no benefit. The
GPU path is itself a compensating benefit R cannot offer, so a *loud* escape
hatch (route to CPU / `gpu_fp64`) is acceptable there — a silent wrong answer is
not, regardless of speed. The single disqualifying outcome in this study is a
solver that returns a non-converged iterate **without raising**.

---

## 1. Problem

`survival.discrete_time(backend='gpu')` forwards to
`regression.fit(family='binomial', backend='gpu')` over a person-period design.
On MPS float32 it intermittently FAILS on some flchain person-period designs. The
GPU GLM IRLS inner solve forms the weighted normal-equations matrix `XᵀWX` and
runs Cholesky on it. Forming `XᵀWX` squares the condition number; in float32 the
computed matrix can come out not-positive-definite and Cholesky raises.

(The separate 4.2.3 fix — a premature stop at the float32 round-off floor, now
stopped on the relative Newton decrement with monotone step-halving — is **not**
the subject here. This note is about the not-PD breakdown.)

## 2. Method

**Data.** flchain person-period (binomial logit), four bin widths giving four
problem sizes ("regimes"):

| regime | bin (days) | rows × cols | intervals |
|---|---|---|---|
| yearly | 365.25 | 82,919 × 21 | 14 |
| quarterly | 91.3 | 319,508 × 62 | 55 |
| monthly | 30.4 | 951,967 × 172 | 165 |
| biweekly | 15.2 | 1,899,947 × 336 | 329 |

Design layout `[interval one-hot | 7 covariates]`, no intercept. Coefficient
error is measured on the **7 always-identifiable covariate columns** (max
relative error vs the reference), so baseline-hazard near-separation in the tail
intervals does not pollute the metric.

**References.** CPU float64 = `pystatistics.regression.fit(..., backend='cpu')`
on the identical design (the ground truth used for coef error). CUDA
`gpu_fp64` confirms an exact device reference (§6).

**IRLS shell (identical across solvers).** `b=0` init; `μ=σ(η)`; `w=clamp(μ(1−μ),
1e-20)`; `z=η+(y−μ)/w`; inner solve of `(XᵀWX)b = Xᵀ(Wz)` warm-started from the
previous `b`; monotone-descent step-halving on the deviance; ≤60 outer iters with
a relative-step early stop. The **acceptance gate is identical for every solver**:
after the loop, the *host float64* relative Newton decrement λ²/(|dev|+0.1) is
computed from the returned coefficients; accept iff < 1e-6 (the same constant the
library's gpu_glm uses), else the result is REFUSED (in production this raises).
The gate is the A6 contract made measurable, and it is independent of the
solver's own internal residual.

**Solvers.** (1) Cholesky on `XᵀWX` (baseline). (2) LM-damped Cholesky:
`(XᵀWX+λ·diag)δ=score`, λ adapted by trust-region accept/reject and driven →0 so
the fixed point is the *unmodified* MLE (solver damping, **not** ridge). (3)
Matrix-free CG on `Hv = Xᵀ(W(Xv))`, `rhs = Xᵀ(Wz)`, warm-started — gated only by
the host fp64 decrement (so a non-converged CG iterate fails loud). (4) Hybrid:
try Cholesky, fall back to CG on not-PD. (5) CPU-QR: keep η/w/z on MPS, ship the
two length-n vectors to the host, solve the n×p WLS in float64 via QR
(`np.linalg.lstsq`) with the host-resident `X` (X never leaves the host).

**Environment.** MPS: Apple-silicon Mac, torch **2.12.1**, fp32,
`PYTORCH_ENABLE_MPS_FALLBACK=0` (non-native ops error rather than silently fall to
CPU — verifies each path is genuinely MPS-native). CUDA: Forge RTX 5070 Ti
(sm_120), torch **2.11.0.dev+cu128**.

**Reps and intermittency.** Within one process, fp32 here is deterministic (6
reps identical). The not-PD failure turned out to be intermittent at **process**
granularity, so it is additionally sampled across many independent process
launches (§4).

---

## 3. Within-process matrix (6 reps each)

### MPS float32

| solver | yearly | quarterly | monthly | biweekly |
|---|---|---|---|---|
| **cholesky** | ✓ 2.4e-5 | ✓ 2.6e-5 \* | ✓ 4.3e-5 | ✓ 1.9e-4 |
| **cg** | ✓ 7.2e-5 | ✓ 1.7e-5 | ✓ 1.3e-5 | ✓ 3.5e-5 |
| **lm** | ✗ refuse | ✗ refuse | ✓ 3.2e-5 | ✓ 2.9e-4 |
| **hybrid** | ✗ refuse | ✗ refuse | ✓ 4.3e-5 | ✓ 1.9e-4 |
| **cpu_qr** | ✓ 9.1e-8 | ✓ 1.3e-5 | ✓ 4.2e-8 | (slow, see §5) |

✓ = all 6 reps converged & gate-accepted (number = max covariate coef rel err vs
CPU fp64). ✗ refuse = all 6 fail loud. **No `silent_wrong` in any cell.**
\* this process ran `yearly` first, warming the MPS context — see §4 for why that
matters for `cholesky`/`quarterly`.

CG iterations/IRLS step (MPS): ~11 (yearly), ~17 (quarterly), ~18–68 (monthly),
~20–88 (biweekly). Median wall (s): cholesky 0.0/0.1/2.5/17; cg 0.2/0.8/8.4/34.

### CUDA float32

Every solver × regime **converged, 6/6, no crash, no refuse, no silent_wrong.**
cholesky 2.5e-5 / 5.3e-5 / 4.6e-4 / 9.0e-4; cg 2.4e-5 / 2.0e-5 / 7.3e-5 / 3.4e-4;
lm 7.4e-6 / 7.3e-5 / 4.7e-4 / 9.7e-4; cpu_qr ~1e-7. CUDA's native Cholesky path is
robust across the whole sweep — it does **not** need the robust solvers.

---

## 4. The not-PD failure is state-dependent, not random (key finding)

Sampling `cholesky`/`quarterly` across **independent process launches**:

| context | launches | converged | crash (fail-loud) | silent_wrong |
|---|---|---|---|---|
| **cold** (quarterly is the first MPS op) | 24 | **0** | **24** | 0 |
| **warm** (run a `yearly` Cholesky first) | 4 | **4** | 0 | 0 |

This is clean and reproducible: a **cold** MPS context produces a not-PD `XᵀWX`
at quarterly; a **warmed** context does not. Mechanism: MPS lazily selects/compiles
the fp32 matmul kernel for a given shape, and the cold-context kernel's reduction
order yields a slightly different (less accurate) `XᵀWX` that tips the knife-edge
quarterly Gram matrix below PD. The earlier baseline runs are all explained by it
— a cold script crashed; the full matrix (which ran `yearly` first) converged.

For production this is **worse than random**: the *same* fit crashes or succeeds
depending on what ran before it in the process — a reproducibility violation. It
is, however, always a **crash (fail-loud)**, never a silent wrong answer.

Cross-process intermittency, all probes (each row = independent launches):

| solver/regime | launches | converged | fail-loud | silent_wrong | MPS verdict |
|---|---|---|---|---|---|
| cholesky / **quarterly** | 24 | 0 | 24 | 0 | unreliable (cold-crash) |
| hybrid / **quarterly** | 24 | 2 | 22 | 0 | unreliable |
| cg / **quarterly** | 6 | **6** | 0 | 0 | **robust** |
| lm / quarterly | 6 | 0 | 6 | 0 | fail-loud |
| cholesky / yearly | 6 | 6 | 0 | 0 | robust |
| cholesky / monthly | 6 | 6 | 0 | 0 | robust |
| cholesky / biweekly | 6 | 6 | 0 | 0 | robust |

The cold-context Cholesky crash is **specific to quarterly** — the one regime
whose `XᵀWX` sits on the fp32 PD knife-edge. yearly/monthly/biweekly form a
comfortably-PD Gram matrix and Cholesky is robust there even cold. **CG converges
cold at quarterly (6/6)** — the squaring-free solver is robust exactly where the
squaring solver is fragile. On CUDA every one of these rows is 24/24 or 6/6
converged: no cold/warm pathology at all.

---

## 5. Per-solver findings

**Cholesky (baseline).** Correct and fast where `XᵀWX` is comfortably PD
(yearly/monthly/biweekly: ✓, 0.0–17 s). At **quarterly** it is on the fp32 PD
knife-edge and crashes from a cold MPS context (24/24). Fail-loud, never silently
wrong — but unreliable and non-reproducible (§4).

**Matrix-free CG (gated).** Converges to fp32 tier across the **entire sweep**,
cold or warm, yearly→biweekly (1.3e-5–7.2e-5), including quarterly where Cholesky
crashes. It never forms `XᵀWX`, so it does not square the condition number and is
immune to the knife-edge. Cost: a matvec pair per CG iteration (~11–88 iters/step
depending on regime), so ~2–4× the wall of a successful Cholesky at large n. The
*ungated* silent-wrong hazard the briefing warned about (small CG residual, large
solution error) **did not reproduce on torch 2.12.1** — and is in any case closed
by the host fp64 gate: a deliberately-crippled CG (1 inner iter) was REFUSED, as
were every other non-converged iterate. **This is the only solver that is both
robust and squaring-free on MPS.**

**LM-damped Cholesky.** The damping does guarantee the factor exists (kills the
crash), but it is **not** a fix: (a) it still *forms* `XᵀWX`, so at quarterly it
inherits the squared-conditioning corruption and lands off the optimum → REFUSE;
(b) its conservative damping under-steps when the optimum needs large coefficient
moves (near-separation at yearly/quarterly here), so it fails to converge within
budget and REFUSES even where plain Cholesky succeeds. On CUDA (cleaner fp32) the
same implementation converges everywhere, confirming the MPS failures are the
fp32/kernel interaction, not the estimator. Net: fail-loud, never silently wrong,
but dominated by CG. (Implementation caveat: a more elaborate LM could narrow the
yearly/quarterly gap; it cannot beat CG's squaring-free robustness, so it was not
pursued further.)

**Hybrid Cholesky→CG.** Dominated by plain CG. Cold, the Cholesky attempts use a
corrupted `XᵀWX` (and the not-PD point arrives mid-iteration), so the blended path
lands badly (2/24 at quarterly). It adds complexity for strictly worse robustness
than CG alone. Reject.

**CPU-QR.** Robust and the most accurate (it solves the un-squared n×p system in
float64): 9e-8–1.3e-5 everywhere. But the per-iteration host round-trip + host
factorization is **expensive at scale**: quarterly ~4.9 s, **monthly ~243 s**,
biweekly ~600 s (on CUDA's host: monthly ~116 s, biweekly ~594 s). That is far
slower than R would take and violates the spirit of the Directive as a *default*.
It is a correct **last-resort fallback**, not a default.

**Acceptance gate.** Across **every** solver × regime × rep × launch in this study
(>300 runs), `silent_wrong = 0`. Every non-converged iterate — cold Cholesky
crash, LM at quarterly, crippled CG, the bad hybrid runs — was REFUSED by the host
fp64 Newton-decrement gate; every accepted fit had true covariate coef error
≤ ~1e-4 vs CPU fp64. **The gate is sound and is the load-bearing safety property.**

---

## 6. CUDA `gpu_fp64` is the exact reference

| regime | rows × cols | `gpu_fp64` max rel err vs CPU fp64 |
|---|---|---|
| yearly | 82,919 × 21 | 2.6e-14 |
| quarterly | 319,508 × 62 | 3.4e-13 |
| monthly | 951,967 × 172 | 1.4e-12 |
| biweekly | 1,899,947 × 336 | 9.6e-13 |

CUDA `gpu_fp64` matches CPU fp64 to ~1e-12: an exact device path exists for users
who need it.

---

## 7. Per-regime classification (MPS float32)

Using the four-way scheme requested. "fp32 tier" = covariate coef rel err
~1e-4–1e-5 vs CPU fp64.

| regime | class | rationale |
|---|---|---|
| **yearly** | **1 — MPS fp32 SAFE** | `XᵀWX` comfortably PD; Cholesky robust cold & warm; CG also fine. |
| **quarterly** | **2 — safe ONLY with a squaring-free solver (CG)** | Cholesky on the fp32 PD knife-edge → cold-context crash (24/24). CG converges cold 6/6 at 1.7e-5. Not impossible, not route-to-CPU — a robust solver fixes it on-device. |
| **monthly** | **1 — MPS fp32 SAFE** | All XᵀWX-forming solvers converge; CG converges; coef err ~1e-5–4e-5. |
| **biweekly** | **1 — MPS fp32 SAFE** | Same; coef err ~3e-5–1.9e-4 (drifting up with n·p but still fp32 tier). |

No regime fell into class 3 (genuine precision floor — even a robust solver can't
hit fp32 tier) or required class 4 (route to CPU/`gpu_fp64`) **on this hardware /
torch 2.12.1**. The precision floor was not reached within the tested range:
CG holds fp32 tier out to biweekly (1.9M × 336). Class 4 (route to
CPU/`gpu_fp64`) remains the correct **user-chosen** remedy for inputs beyond this
envelope or whenever the gate refuses — reached by a loud failure that names the
options, not by the library silently switching device (§8).

**Caveat — torch-version dependence.** The briefing's prior session observed (a)
intermittent Cholesky failure *and* (b) silent CG non-convergence at monthly. On
torch 2.12.1 here, (b) did not reproduce — CG is robust monthly and biweekly — and
(a) is the cold/warm effect of §4. The MPS kernel behaviour is clearly
version-sensitive (Forge's torch 2.11.0.dev is different again). **Any MPS solver
policy must be re-validated on the torch version actually shipped**, and the
fail-loud gate (which is version-independent) is what makes a wrong call safe
rather than silent.

---

## 8. Recommended production policy

**Default MPS GLM inner solve → matrix-free gated CG** on `Hv = Xᵀ(W(Xv))`,
warm-started from the previous IRLS `b`. It is the only solver that is both robust
across the whole tested sweep (cold or warm) and squaring-free, so it removes the
quarterly knife-edge crash at its root rather than masking it. Keep the existing
host fp64 Newton-decrement acceptance gate exactly as the A6 safety net.

**Resolution for `backend='gpu'` on MPS (no automatic device substitution):**

1. **gated CG** (the MPS default inner solve). Accept iff host fp64 relative
   Newton decrement < 1e-6.
2. on REFUSE → **fail loud**: raise `NumericalError` whose message recommends
   `backend='cpu'` or `backend='gpu_fp64'` (and `force=` where applicable). The
   call does **not** silently move the computation off the requested MPS device.

**CPU-QR is NOT an automatic rung.** Although CPU-QR is correct and was the most
accurate solver measured (§5), running it automatically after the MPS gate
refuses would be a *silent device substitution* — `backend='gpu'` quietly
returning a CPU result — which is exactly what A6 forbids (and the cost is high at
large n anyway). CPU-QR therefore belongs only behind an **explicit user opt-in**
(e.g. a `mps_cpu_recovery=True` / `force=`-style flag, or an explicitly invoked
recovery path), never inside the default `backend='gpu'` resolution. The honest
default is: MPS fits on MPS, or it raises and names `backend='cpu'` /
`backend='gpu_fp64'` for the user to choose deliberately.

**Cholesky is *not* the MPS default** (it is fine on CUDA and may stay the CUDA
default): on MPS its correctness depends on context-warming state it cannot
control, which violates reproducibility. **LM-damping and Hybrid are rejected** —
both still form `XᵀWX`, neither matches CG's robustness, both add fragility.

**Fail-loud gates (all version-independent):**
- inner CG: cap iterations, and if `pᵀAp ≤ 0` (SPD lost in fp32) stop — do **not**
  return that iterate as if converged.
- post-loop: host fp64 relative Newton decrement < 1e-6 to ACCEPT; otherwise
  **raise**. This is what converts CG's residual-vs-error gap into a loud failure
  and is the property that makes the whole path A6-safe.

**Route-to-CPU/`gpu_fp64` boundary.** Not a size threshold on this hardware — CG
holds fp32 tier to biweekly. The boundary is **decided by the user, surfaced by a
loud failure**: when the MPS gate refuses, the library raises and the user
re-issues with `backend='cpu'` or `backend='gpu_fp64'`. The library never crosses
the device boundary on its own. This honours the Directive: the GPU default stays
a genuine fp32-tier benefit; the requested device is never silently abandoned, and
the user is never handed a quietly-wrong answer.

**Tie to the Prime Directive.** The CPU path is untouched (no slower-than-R
regression). The GPU path keeps its bonus by *widening* the set of designs it
fits without crashing (quarterly now works on-device), while every
non-fp32-reachable case fails loud with named remedies. No outcome in this study
is silently wrong.

---

## 9. What a follow-up implementation chip would change in `gpu_glm.py`

Scoped as a separate, well-bounded task (its own release + survival re-validation):

1. **MPS branch of the inner solve:** replace the `XᵀWX` + `torch.linalg.cholesky`
   block (currently ~lines 272–292) with a matrix-free CG solve of
   `Hv = Xᵀ(W(Xv))`, warm-started from the current coefficient vector, with an
   iteration cap and a `pᵀAp ≤ 0` guard. **Leave the CUDA branch on Cholesky**
   (robust there) — gate the change on `device.type == 'mps'`.
2. **On REFUSE, raise — do not substitute the device.** On post-loop gate REFUSE,
   raise `NumericalError` whose message recommends `backend='cpu'` /
   `backend='gpu_fp64'`. Do **not** automatically run CPU-QR or any CPU path under
   `backend='gpu'` (that would be a silent device substitution, an A6 violation).
   If a CPU-QR recovery is wanted, expose it only behind an explicit opt-in flag,
   designed as its own decision. Reuse the existing host fp64 Newton-decrement
   gate (`_newton_decrement` / `_FP32_REL_DECREMENT_TOL`) verbatim — do not
   re-tune it.
3. **No change** to: the 4.2.3 stop criterion (relative Newton decrement +
   monotone step-halving), the acceptance threshold, the deviance/convergence
   logic, or the CPU path.
4. **Tests:** add a person-period quarterly-scale binomial case that crashes the
   current MPS Cholesky from a cold context and asserts the CG path converges to
   fp32 tier; assert a crippled/ill-conditioned case still fails loud (no
   silent_wrong). Re-run the survival discrete-time GPU validation and re-issue
   the frozen v4.2.3 report.
5. **Re-validate on the shipped torch version** (§7 caveat) before release.

---

## Appendix — reproduction

Standalone prototypes (not library code): `lab/` (pp_data, irls_common,
solver_scoring, solver_lm, run_matrix, single_run), `intermittency.sh`,
`forge_fp64.py`. Env: venv with `pystatistics==4.2.3 torch pandas h5py numpy` +
`pip install -e validation`; `KMP_DUPLICATE_LIB_OK=TRUE`,
`PYTORCH_ENABLE_MPS_FALLBACK=0`, `MVNMLE_DATA_DIR=<datasets>`,
`PYTHONPATH=<pystatistics-validation>`. Designs from
`drivers.survival.datasets.load_flchain` + `_person_period.build_person_period`.
Raw results: `matrix_mps.json`, `matrix_cuda.json`, `intermittency_mps.jsonl`,
`intermittency_cuda.jsonl`, `fp64_ref.json`. CUDA reference produced on Forge
(RTX 5070 Ti) under the standing CUDA-testing allowance; throwaway env removed
after the run.
