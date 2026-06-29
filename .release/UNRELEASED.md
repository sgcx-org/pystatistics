# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **GPU GLM on Apple Silicon (MPS): the float32 IRLS inner solve is now a
  matrix-free conjugate-gradient step, replacing Cholesky of the weighted
  normal-equations matrix.** The old path formed `XᵀWX` on-device and factored it;
  forming `XᵀWX` squares the design's condition number, and on a cold MPS context
  the float32 Gram matrix at person-period quarterly scale could come out
  not-positive-definite and abort the Cholesky. The new solve uses only the
  operator `H v = Xᵀ(W(X v))` (and right-hand side `Xᵀ(W z)`), warm-started from
  the previous IRLS iterate, so `XᵀWX` is never formed and the squaring-induced
  breakdown cannot occur. Net effect: `regression.fit(family=..., backend='gpu')`
  and `survival.discrete_time(backend='gpu')` now **converge** on MPS across the
  flchain person-period sweep (yearly/quarterly/monthly), including the quarterly
  scale that previously failed. This is a **reliability** improvement, not a
  correctness fix — the old path always failed loud, never returned a wrong
  answer. Affects `regression/backends/gpu_glm.py`; the MPS plain-float32 path
  only. CUDA is unchanged (its Cholesky path is robust); `gpu_fp64`, ridge, and
  the CPU path are unchanged.
- **MPS refuse message no longer suggests `backend='gpu_fp64'`.** When a GPU GLM
  fit cannot reach a reliable float32 solution and fails loud, the message now
  recommends `backend='cpu'` on Apple Silicon and omits `gpu_fp64` there
  (`gpu_fp64` requires CUDA, so suggesting it to an MPS user only produces a
  follow-on "CUDA required" error). The CUDA message is unchanged. No automatic
  device substitution is added: an unreliable `backend='gpu'` fit raises and the
  user chooses the explicit fallback — it never silently falls back to CPU.
- The host float64 Newton-decrement acceptance gate is unchanged and remains the
  sole arbiter of whether a float32 GPU fit is accepted: a conjugate-gradient
  iterate that does not reach a stationary (float32-tier) solution is refused,
  exactly as before. Note: the MPS solver path is validated against the shipped
  torch version (MPS kernel behaviour is torch-version-sensitive); the host
  float64 gate is the version-independent guarantee that an unreliable fit fails
  loud rather than returning silently wrong.
