# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

### GAM module: numerical core rewritten (fixes silently-wrong smoothing selection)

- **Fixed: automatic smoothing-parameter selection returned essentially arbitrary
  fits.** 4.5.x smooth bases carried no identifiability constraint, so every
  smooth's span contained the constant function and the design matrix was
  *exactly singular* against the intercept (cond(X'WX) ~ 2e17). The
  normal-equations EDF computation then returned garbage (measured: erratic
  0.45–13.33 against a true monotone 13→4.5 on `MASS::mcycle`; `total_edf =
  −3.12` at k=10), which corrupted the GCV/REML objective and sent the
  λ-optimizer to arbitrary optima — over-fitting mcycle (EDF 11.9 vs mgcv 8.4)
  and over-smoothing a 4-smooth `gamSim` fit to near-flat (deviance 2889 vs
  mgcv 1511.6), all silently with `converged=True`. Every smooth now absorbs
  the mgcv sum-to-zero constraint (a smooth declared with `k` basis functions
  contributes `k−1` coefficients, exactly as mgcv), and all solves go through
  augmented/reduced QR (Wood 2011) instead of the penalized normal equations.
  After the fix, free-selection fits match `mgcv::gam` 1.9-3: mcycle GCV sp
  1.534910 vs 1.53491, EDF 9.38953 vs 9.38953; gamSim-4-smooth EDF
  [2.49, 2.40, 7.64, 1.00] / GCV 4.0694 / deviance 1511.6 — all equal to
  mgcv's displayed digits.
- **Fixed: `bs='tp'` could not represent a straight line.** The 4.5.x thin-plate
  basis projected out its polynomial null space {1, x} and kept only penalized
  eigenvectors, so a tp smooth of a purely linear signal (y = 2+3x+ε) fit a
  *flat* line (EDF 0.0, RMSE 0.81 vs truth, silently). The tp basis is now the
  Wood (2003) construction with the null space retained (function-space
  identical to mgcv's: hat-matrix agreement ~4e-17 at fixed λ; free-selection
  fitted values within 7e-7 of mgcv on mcycle) and knots capped at 2000 unique
  values (mgcv's `max.knots`), replacing the previous O(n³) full-data
  eigendecomposition.
- **Fixed: coefficient standard errors were a placeholder.** `_param_se`
  returned `sqrt(scale/n)` for *every* coefficient (its own docstring said
  "This is a placeholder"), making all summary z/p values meaningless. The
  Bayesian posterior covariance `Vp = scale·(X'WX+S_λ)⁻¹` (mgcv's `Vp`) is now
  computed from the QR factors, stored on `GAMParams.covariance`, and exposed
  as `GAMSolution.se` / `GAMSolution.covariance`; diagonal agrees with mgcv to
  ~6e-15 relative. Parametric-table p-values now use t (estimated scale) or z
  (known scale), matching `summary.gam`.
- **Fixed: the REML criterion was wrong.** The 4.5.x "simplified REML" omitted
  its own documented Σedf·logλ term, derived its scale from the corrupted EDF,
  and took a log-determinant of the singular unpenalized crossproduct ridged by
  1e-10. `method='REML'` is now the Laplace REML of Wood (2011) with mgcv's
  exact conventions (verified to 0 (gaussian) / 6e-11 (poisson) absolute
  against mgcv's reported score at fixed sp; free REML selection matches
  mgcv's sp to ~4 significant digits). REML for free-dispersion families other
  than Gaussian-identity (e.g. Gamma) raises a clear `ValidationError` (mgcv
  estimates their scale inside REML; we do not yet) — use `method='GCV'`.
- **Fixed: `method='GCV'` now follows mgcv `GCV.Cp` semantics** — GCV for
  estimated-scale families, UBRE for known-scale families (poisson/binomial).
  4.5.x always used GCV, which is not what mgcv's default does. UBRE verified
  to 4e-13 against mgcv at fixed sp; free selection matches sp to 5 digits.
- **Fixed: `bs='cr'` is now mgcv's exact cubic-regression-spline construction**
  (Wood 2017 §5.3.1: knots at type-7 quantiles of the unique covariate values,
  banded D/B second-derivative relations, penalty D'B⁻¹D, and mgcv's
  `scale.penalty` normalisation). Verified against
  `mgcv::smoothCon(s(x, bs="cr"), absorb.cons=FALSE)`: basis matrix to 8e-16,
  knots to 1e-16, penalty (after `S.scale`) to 2e-16, `S.scale` itself to
  4e-16. Consequence: reported smoothing parameters are *directly comparable*
  to mgcv's `sp` for `cr` smooths, and `k` now means exactly what mgcv's `k`
  means (the 4.5.x B-spline construction produced k+2 columns for `s(k)`).
- **Removed: the GAM GPU backend (`backend=` parameter gone).** The fp32 GPU
  path produced silently wrong EDF in the small-λ regime the λ-optimizer
  probes by design (proven on CUDA: 33.6-DOF EDF error and a *negative* EDF at
  n=100k; reproduced on MPS: a wiggly n=1000/k=40 fit silently over-smoothed
  from EDF 24.1 to 3.0 with fitted values off by 0.50 on a unit-amplitude
  signal), and the fp64 GPU variant measured *slower* than CPU at typical GAM
  sizes (0.67× at p=50 on an RTX 5070 Ti; datacenter fp64 hardware unmeasured).
  Per the library convention ("a module with no GPU path exposes no
  `backend=` at all"), `gam()` no longer accepts `backend`; passing it now
  fails with `TypeError`. The GAM module is CPU-only; `backend_name` reports
  `cpu_qr_pirls`.
- **Added: `sp=` parameter on `gam()`** (mgcv-parity): fix the smoothing
  parameters and skip selection — needed to reproduce a specific fit and for
  fixed-λ cross-engine validation.
- **Added: `GAMSolution.lambdas`, `.se`, `.covariance`, `.ubre`,
  `.reml_score`, `.outer_converged`;** `GAMParams` gains `lambdas`,
  `s_scales`, `covariance`, `reml_score`, `outer_converged`, `backend_name`
  (large arrays are `repr=False`). `SmoothInfo` gains `lambda_` and `s_scale`
  and its `ref_df` is now mgcv's Ref.df (`tr(2H−HH)` per block).
- **Robustness:** complete separation in binomial GAMs now produces R's
  "fitted probabilities numerically 0 or 1" warning and a finite fit (4.5.x
  architecture fed inf/NaN into the solver); genuinely divergent P-IRLS raises
  `ConvergenceError`; rank deficiency (concurvity, over-specified `k`) is
  detected by column-pivoted QR and reported with a warning naming the smooths
  while dependent columns are dropped (mgcv behaviour) — never a silent ridge
  (4.5.x silently added 1e-8 ridges in three places). `k` is validated against
  the number of *unique* covariate values (mgcv's rule), not just n.
- **Behaviour notes:** AIC uses the classical GAM df convention
  (`total_edf + 1` when the scale is estimated); mgcv ≥ 1.8.x corrects df for
  smoothing-parameter uncertainty ('edf2'), so mgcv's `AIC()` is systematically
  slightly larger — documented on `GAMParams.aic`. Smooth-term test statistics
  follow the shape of mgcv's Wood (2013) test but are a simplified form
  (documented on `SmoothInfo.chi_sq`). `SmoothTerm` is now a pure immutable
  spec (no more per-fit mutable caches), safe to reuse across `gam()` calls.
- **Performance:** Gaussian-identity λ-search evaluations run in reduced
  p-space from one cached QR (O(p³) per evaluation instead of O(np²));
  non-Gaussian fits warm-start from the previous λ-evaluation's fit. mcycle
  free-GCV fit: 6 ms; 4-smooth gamSim (n=400): 24 ms.
- **Fixed (adversarial review of the rewrite): GCV/UBRE selection is now
  scale-invariant.** The outer optimizer's termination tests were keyed to
  the absolute criterion magnitude (GCV scales as c² under y→c·y), so the
  same data in different units silently selected different smoothness
  (measured: EDF 4.00/2.19/3.77 for y×{1, 0.01, 1e6}; mgcv is invariant at
  3.77). The criterion is now normalised by its value at the starting point
  and the finite-difference step sized above the inner-P-IRLS noise floor;
  the selected EDF is bit-identical across response scales and equals
  mgcv's. Also fixes a stall-short-of-optimum on very flat GCV surfaces
  (Gamma log-link: selected GCV now matches mgcv's to 7 digits).
- **Fixed (review): `names` with the wrong length silently mislabelled
  smooth coefficients as parametric terms** in `summary()` (or silently
  dropped genuine parametric rows). `gam()` now raises `ValidationError`
  when `len(names)` differs from the parametric column count.
- **Fixed (review): smooth-term significance statistics for `tp` smooths
  were understated up to ~14×** (a Ref.df-rank truncation of the covariance
  pseudo-inverse discarded well-determined directions). The full
  pseudo-inverse quadratic form now matches mgcv's reported F/Chi.sq to
  ~0.2% on validated cases.
- **Fixed (review): dispersion estimate for free-dispersion non-Gaussian
  families (Gamma) now uses the Fletcher (2012) estimator** — mgcv's
  default — instead of plain Pearson/(n−edf); matches mgcv `sig2` exactly
  on the validated case (SEs previously ~1–2% off).
- **Fixed (review): REML on rank-deficient designs mixed the full column
  count with the numerical rank** (inconsistent φ-corrections and inflated
  null-space dimension); all REML bookkeeping now uses the numerical rank.
- **Fixed (review, summary output):** the summary now labels the smooth
  statistic `F` (estimated scale) vs `Chi.sq` (known scale) and the
  parametric statistic `t` vs `z`, driven by the family's dispersion flag
  (negative-binomial was previously mis-branched by a name check); the
  Formula line includes parametric terms instead of printing `y ~ 1`; the
  footer reports `UBRE` for known-scale selection.
