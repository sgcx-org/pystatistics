# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **Faster negative-binomial and Gamma GLM fits (vectorized deviance residuals).**
  `regression/backends/cpu_glm.py` (and the GPU IRLS backend) computed deviance
  residuals for any family outside gaussian/binomial/Poisson via a per-observation
  Python loop calling `family.deviance` once per row — which, with the NB θ-profile
  refitting the GLM several times, dominated the fit (≈56% of negative-binomial
  runtime on `MASS::quine`). Added vectorized unit-deviance branches for Gamma and
  negative binomial (identical formulas to `family.deviance`, so residuals are
  unchanged: Σ deviance_residual² still equals the deviance exactly). Negative
  binomial on quine: ~5.0 ms → ~2.2 ms (2.2×), now faster than R's `MASS::glm.nb`.
- **GPU GLM now works on Apple Silicon (MPS).** `regression/backends/gpu_glm.py`
  previously solved each IRLS weighted-least-squares step with
  `torch.linalg.lstsq`, which Metal does not implement — so every GLM fit with
  `backend='gpu'` on a Mac raised "GPU LSTSQ failed on this device". Replaced the
  `lstsq` inner solve with the weighted normal equations (XᵀWX)β = XᵀWz solved by
  Cholesky + two triangular solves (the same primitives the OLS GPU backend
  already uses on MPS). GLM (binomial/Poisson/Gamma/negative-binomial) now runs on
  MPS and matches the CPU/R reference to fp32 round-off (~5e-7 relative on
  coefficients, identical IRLS iteration counts). Also faster on CUDA: the p×p
  Gram solve replaces an `lstsq` over the full n×p system.
- **GPU OLS condition check no longer does an n×p SVD every solve.**
  `regression/backends/gpu.py` computed `torch.linalg.svdvals(X)` on the full n×p
  design purely for the ill-conditioning guard — and on MPS (no `svdvals`) it
  round-tripped the entire design to the CPU on every fit. Now the check uses the
  p×p Gram matrix XᵀX (formed for the solve anyway): `cond(X) = sqrt(cond(XᵀX))`,
  computed from a p×p SVD. Identical guard semantics, orders of magnitude less
  work and data movement (especially on MPS); the fp32 Cholesky remains the hard
  backstop for genuinely ill-conditioned designs.
