# GPU/MPS dense-linalg optimization — survey & TODO

**Status:** survey only. This is a triage list for *future, dedicated, one-module-
at-a-time* sessions. Nothing here is scheduled or in-flight except MICE (the
module that prompted the survey). Do **not** treat this as a batch work order.

## Finding: the MPS small-n floor (why MPS can't fully match CUDA there)

Established while optimizing MICE (per-op MPS-vs-CUDA timing + controlled in-sweep
runs). Two distinct causes, both defensible:

1. **A few torch ops have pathologically slow MPS kernels** — `solve_triangular`
   ~250x slower than CUDA, `searchsorted` ~1136x (n=20k), `cholesky_solve`/`eigh`
   unimplemented. These ARE recoverable by rebuilding from MPS-fast primitives
   (matmul/cholesky/sort): MICE replaced `searchsorted`→merge-rank and
   `solve_triangular`→matmul-series inverse, closing most of the *mid/large-n* gap
   to ~the raw FP32 silicon ratio (~3-4x).
2. **Per-op dispatch overhead is the small-n floor and is NOT recoverable.** In a
   sequential per-step sweep (maxit·p steps that cannot be re-batched), MPS pays
   ~0.5-1ms of command-encode overhead *per op*, and — unlike CUDA — has no
   graph-capture to amortize it. So at small n the sweep is dispatch-bound, not
   compute-bound: a fast many-op method and a slow-single-kernel method net out,
   and no linalg reformulation helps (the cost is launches, not math). This is an
   intrinsic platform limitation; the honest answer to "why is MPS's small-n
   speedup-over-CPU far below CUDA's" is this dispatch floor, not silicon.

The only lever left against (2) is reducing *op count per step* (fuse the per-step
gathers/scatter) — attacks the dispatch floor directly; deferred (diminishing
returns vs complexity).

## The recurring fact

Apple MPS executes batched **matmul** and **cholesky** fast, but its small dense
**factor-and-solve** kernels are slow or absent:

| op | MPS status (torch 2.10/2.11, this repo's measurements) |
|---|---|
| `matmul`, `cholesky_ex`, `sort`, `cumsum`, `gather` | fast |
| `solve_triangular` | slow (~4 ms for a 20×20 batch, **n-independent**) |
| `linalg.solve`, `linalg.inv`, `pinv` | slow (~100-300× matmul) |
| `cholesky_solve` | **unimplemented** (errors on MPS) |
| `linalg.eigh` | **unimplemented** (errors on MPS) |
| `searchsorted` | fine at small size, slow at scale |

## The in-house remedy (to be promoted to a core primitive)

`pystatistics/mvnmle/_objectives/_batched_cholesky.py` already solves this:
- `_tri_inv_blocked(torch, L)` — matmul-only block-recursive inverse of a batched
  lower-triangular factor. No `solve_triangular`/`inv` kernel touched.
- `_use_blocked(L, method)` — device dispatch: matmul-inverse on MPS,
  `solve_triangular` on CUDA/CPU (fast there). One shared path, one device bridge.

**DRY action (MICE + MVNMLE both need it → promote):** lift `_tri_inv_blocked` +
the `_use_blocked` dispatch into `pystatistics/core/compute/linalg/` as a shared
primitive; have MVNMLE and MICE both import it. (It already lived siloed in
mvnmle, which is why MICE 3.13.0 re-hit the wall with `solve_triangular`.)

## Per-module candidates (verify reachability on the MPS hot path before acting)

Priority key: **P1** unimplemented-on-MPS (path breaks/refuses if reached) ·
**P2** slow on MPS hot path · **P3** likely cold/CPU/needs-check.

| module | file:line | op | MPS issue | candidate fix | pri |
|---|---|---|---|---|---|
| mice | `_gpu_linreg.py:106,110` | `solve_triangular` | slow | matmul-inverse (THIS work) | — in progress |
| multivariate | `backends/gpu_pca.py:252,277,278` | `eigh` | **unimplemented** | SVD-based PCA, or matmul/CPU bridge | P1 |
| regression | `backends/gpu.py:151` | `cholesky_solve` | **unimplemented** | 2 tri-solves / matmul-inverse | P1 |
| mvnmle | `backends/_em_batched_torch.py:29,89,91,270` | `cholesky_solve` | **unimplemented** | matmul-inverse (shared primitive) | P1 |
| gam | `backends/gpu_pirls.py` (8× `linalg.solve`) | `linalg.solve` | slow; iterative PIRLS magnifies | chol + matmul-inverse per IRLS step | P2 |
| regression | `backends/gpu.py:155,158` | `solve_triangular` | slow | matmul-inverse | P2 |
| multinomial | `backends/gpu_likelihood.py:219,221` | `inv`/`pinv` | slow | chol + matmul-inverse | P2 |
| ordinal | `backends/gpu_likelihood.py:231,233` | `inv`/`pinv` | slow | chol + matmul-inverse | P2 |
| mvnmle | `backends/_em_batched_torch.py:206` | `solve_triangular` | slow | matmul-inverse (shared) | P2 |
| timeseries | `_arima_batch.py:99` | `linalg.solve` | slow | chol + matmul-inverse | P2 |
| regression | `backends/gpu.py:175,180`, `gpu_glm.py` | `lstsq` | MPS support/speed unverified | verify; QR or normal-eq+chol | P3 |
| core | `compute/linalg/batched.py` | `solve`/`solve_triangular`/`lstsq` | shared util — promotion home; review its own ops | add matmul-inverse here | P3 |
| regression | `_linear.py:114` | `linalg.inv` | likely CPU path | verify device | P3 |

Notes:
- `mvnmle/_objectives/_batched_cholesky.py` `solve_triangular` calls (236/237/257/
  265) are the **CUDA/CPU branch** of `_use_blocked` — correct as-is, not a defect.
- Each P1 should first be checked for whether the MPS path is *guarded/refused*
  (then it's a "lift the refusal + reformulate" task, like MICE was) or *reached
  and crashing* (a latent bug).
- Each fix is its own task with its own R-fidelity + FP32 re-validation on MPS
  **and** CUDA (the shared-primitive change touches both devices).
