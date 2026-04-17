# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py <version>`
> That script uses this file to build the CHANGELOG entry, bumps versions
> everywhere, and resets this file for the next cycle.

## Changes

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
