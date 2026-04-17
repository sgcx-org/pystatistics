# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py <version>`
> That script uses this file to build the CHANGELOG entry, bumps versions
> everywhere, and resets this file for the next cycle.

## Changes

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
