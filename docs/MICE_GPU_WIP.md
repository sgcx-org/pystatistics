# ⚠️ TEMPORARY — MICE GPU working list — DELETE THIS FILE WHEN EMPTY

Not a shipped/public doc. Module-local follow-ups for the **MICE GPU path only**.
Delete when all items are done. (Cross-module survey + the MPS dispatch-floor
finding live in `MPS_LINALG_TODO.md`; this is MICE-specific and disposable.)

Done and shipped in 3.14.0 (kept here only as breadcrumbs; remove with the file):
sync-free per-step Cholesky + end-of-sweep fail-loud guard; matmul-series inverse
(`batched_tri_inv_series`) on MPS above n_obs 3000, `solve_triangular` below and
on CUDA/CPU; ~1.7–2x over 3.13.0. The small-n floor (MPS per-op dispatch overhead,
no graph capture) is intrinsic and documented in `MPS_LINALG_TODO.md`.

Still pending:

- [ ] **Categorical / nominal imputation on GPU** (feasibility established — not
  impossible, not infeasible; bounded Newton/IRLS is batchable). Library = broad
  capability, so pursue. Build order, each gated on R-fidelity at FP32:
  1. Layer 1 — dummy-encode categorical *predictors* with numeric targets (cheap,
     reuses the Gaussian path; no IRLS).
  2. Binary `logreg` — batched IRLS prototype (load-bearing). Build the per-step
     solve via the matmul-series inverse (NOT `solve_triangular` on MPS).
  3. Multinomial `polyreg` (ragged K — efficiency risk lives here).
  4. Ordinal `polr` (ordered thresholds under FP32).

- [x] **(done)** Migrated the MVNMLE GPU objective to `batched_tri_inv_series`
  and retired the block inverse `batched_tri_inv` — one inverse primitive in core.
  The series is now autograd-safe (detached-Newton wrapper → exact inverse VJP,
  matches solve oracle to ~5e-16), a full drop-in. MVNMLE 158 + MICE 171 green;
  MVNMLE MPS end-to-end matches CPU (max|Δmu|~6e-5, max|ΔSigma|~5e-4). Internal,
  no user-facing change; sitting in UNRELEASED for the next release.

- [ ] **(deferred, low priority)** Only remaining lever against the small-n
  dispatch floor: cut op-count per sweep step (fuse the per-step predictor
  gathers/scatter, ~800us/step at small n). Attacks dispatch directly; diminishing
  returns vs complexity — revisit only if small-n MICE becomes a real bottleneck.
