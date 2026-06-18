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

- [x] **Categorical / nominal imputation on GPU — DONE (all four layers).** Every
  categorical target type now imputes on GPU; the full mixed bin/nom/ord dataset
  matches R `mice` 3.19.0 imputed proportions directly on MPS (worst max|Δ| ~0.013,
  tol 0.06) — see `TestMpsCategoricalMatchesR`. Build order, each gated on
  R-fidelity at FP32:

  **CUDA validation (Forge RTX 5070 Ti, torch 2.11/cu128, 2026-06-17) — DONE, two
  levels:**
  (a) *Kernels* — `scratch/mice_cat/cuda_validation.py` (self-contained torch,
  piped into the GPU worker container): CUDA-fp64 == CPU-fp64 to machine precision
  (logreg 1.1e-16, polyreg 4.4e-16, polr 2.2e-16), gradient-norm ~0 at each
  solution (it IS the MLE), thresholds ordered in fp64 & fp32, deterministic,
  finite. The CUDA-specific paths (`solve_triangular` branch, polr autograd
  double-backward) all work.
  (b) *Full backend end-to-end* — `scratch/mice_cat/cuda_integration.py`: synced
  the current working tree into the GPU container (tar-pipe to `/tmp/pyst`,
  `PYTHONPATH=/tmp/pyst`; no commit/push, no product-stack change, cleaned up
  after) and ran the REAL `mice(..., backend='gpu')` → device=cuda for all three
  methods. Proportions vs CPU: logreg 0.0037, polyreg 0.0104, polr 0.0140; direct
  CUDA-vs-R on the mixed fixture: bin 0.0099, nom 0.0081, ord 0.0038 (all < 0.06);
  valid codes, no-NaN, deterministic. NOTE: pystatistics is a public lib on the
  owner's own dev box — the OPERATIONS.md "no-source" rule is about the *product*
  (sgcbio/sgccore) build image, not this. The in-repo CUDA tests
  (`tests/mice/test_gpu.py::TestGpuCategoricalTargets` / `...MatchesR`) skip
  locally (no CUDA) and run on any CUDA box with current pystatistics.
  1. [x] Layer 1 — dummy-encode categorical *predictors* with numeric targets
     (cheap, reuses the Gaussian path; no IRLS). Shipped.
  2. [x] Binary `logreg` — batched IRLS (load-bearing experiment): **succeeded**.
     `backends/_gpu_logreg.py` (`batched_logistic_irls` + `gpu_logreg_impute`);
     per-step solve via `cholesky_ex` + `batched_tri_inv_series` on MPS, per-chain
     convergence freezing, posterior draw `N(beta_hat, (X'WX)^-1)`. Validation
     (scratch `scratch/mice_cat/logreg_mps_proto.py`): fp64 algorithm matches CPU
     `_fit_logistic` to Δbeta 2e-16 / Δcov 3e-9 when converged; under separation
     beta diverges (as in CPU/R) but predicted probs match to ~5e-5; FP32-on-MPS
     imputed proportion within ~0.005 of CPU; deterministic. Code<->index sweep
     wrapping in `gpu.py` via `_gpu_encode.codes_to_indices`/`indices_to_codes`;
     refusal lifted per-method. MPS tests in `tests/mice/test_gpu_mps.py`
     (`TestMpsLogregBinaryTarget`). Gate = CPU-as-oracle at FP32 (CPU is itself
     R-validated on the `bin` column in `test_r_validation_categorical.py`); a
     direct GPU-vs-R run on the mixed R fixture isn't possible until polyreg/polr
     land (that fixture's nom/ord columns are still refused).
  3. [x] Multinomial `polyreg` — batched multinomial-logit Newton: **succeeded**.
     `backends/_gpu_polyreg.py` (`batched_multinomial_newton` + `gpu_polyreg_impute`);
     class-major block Hessian, `cholesky_ex` + `batched_tri_inv_series` via the new
     shared `backends/_gpu_spd.py` (`solve_spd` / `apply_inv_factor_T`, extracted
     from the logreg path so polyreg/polr don't couple to logreg), per-chain Newton
     freezing, posterior draw `N(coef, H^-1)`, inverse-CDF sampling. Validation
     (scratch `scratch/mice_cat/polyreg_mps_proto.py`): fp64 Newton vs the true CPU
     MLE (multinom at tol 1e-12) — Δcoef 1.9e-6 / Δvcov 9e-7 / Δprob 9e-7; FP32-on-MPS
     category proportions within ~0.002 of CPU; deterministic. NOTE on the "ragged-K"
     risk: a non-issue — the sweep calls one method per column and all m chains share
     that column, so K is fixed within a call (no pad/mask needed; ragged-K would only
     arise batching *different* columns together, which the sweep never does). MPS
     tests in `TestMpsPolyregNominalTarget`. Same CPU-as-oracle gate as logreg.
  4. [x] Ordinal `polr` — batched proportional-odds Newton: **succeeded**.
     `backends/_gpu_polr.py` (`batched_polr_newton` + `gpu_polr_impute`). Fit in the
     RAW (unconstrained) threshold parameterization `[alpha_0, log(alpha_j-alpha_{j-1})]`,
     so thresholds stay ordered by construction — the "ordered thresholds under FP32"
     risk is structurally eliminated, no clipping. Gradient + observed Hessian by
     autograd on the batched NLL: chains are independent, so P+1 backward passes give
     every chain's Hessian (no `torch.func` — broken on this MPS build; no per-chain
     loop). Faithful to the CPU draw's natural-mean / raw-covariance convention
     (`OrdinalSolution.vcov` is raw-parameterization). Validation (scratch
     `scratch/mice_cat/polr_mps_proto.py`): fp64 vs CPU polr at tol 1e-12 — Δalpha
     1.7e-9 / Δbeta 3e-9 / Δvcov 2.6e-9 / Δprob 3e-9; FP32-on-MPS proportions within
     ~0.004 of CPU; deterministic. Heaviest GPU method (autograd double-backward per
     Newton step; ~7.6s for m=30/maxit=10) — analytical Hessian is a future lever.
     MPS tests in `TestMpsPolrOrderedTarget` + direct R gate in `TestMpsCategoricalMatchesR`.

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
