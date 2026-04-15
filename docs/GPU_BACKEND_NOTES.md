# GPU Backend Notes

Hard-won knowledge about GPU behavior across CUDA and MPS backends.
Written because people won't know this unless they hit the specific
issue in the specific way.

---

## CUDA vs MPS: When Metal Falls Off a Cliff

### The Short Version

NVIDIA CUDA and Apple MPS are not interchangeable GPU backends. Certain
memory access patterns that are fast on CUDA are **catastrophically slow**
on MPS — not 2x slower, but 1000x slower. If your algorithm uses
`scatter_add_` with sparse, irregular bucket IDs, MPS is the wrong backend.

The pystatistics and pystatsbio libraries detect this and fail fast rather
than silently delivering a 15-minute wait.

---

### The Pattern That Breaks MPS: `scatter_add_` with Sparse Targets

**What the code needs to do** (example: midrank computation for batch AUC
with 5,000 biomarkers × 500 patients):

1. Sort each column — MPS is fine at this.
2. Find groups of tied values — MPS fine.
3. **For each tie group, sum up the ranks of its members and count them**
   — this is `scatter_add_`.
4. Divide to get the average rank per group — MPS fine.

Step 3 is the problem. `scatter_add_` says: "I have 2.5 million items.
Each one has a bucket ID. Add each item's value into its bucket." The
bucket IDs are sparse and irregular — each column has different tie
patterns.

### Why CUDA Handles This Fine

NVIDIA GPUs have **atomic operations in shared memory**. When 1,000
threads simultaneously try to add values to the same bucket, CUDA
serializes them at the hardware level using atomic compare-and-swap — a
few clock cycles per conflict. The GPU's memory controller is designed
for this pattern.

**Measured**: ~0.05 ms per `scatter_add_` call on RTX 5070 Ti.

### Why MPS Is Catastrophically Slow

Apple's Metal GPU architecture was designed for **graphics** — rendering
pixels, texture mapping, vertex shading. These workloads are *regular*:
every pixel does the same work, reads from the same textures, writes to
a predictable framebuffer location.

`scatter_add_` is the opposite: thousands of threads writing to *random,
unpredictable* memory locations with *read-modify-write* semantics.
Metal has to:

1. **Serialize conflicting writes** — Metal's atomic support is weaker
   than CUDA's. When two threads hit the same bucket, one stalls.
2. **The memory access pattern defeats the cache** — buckets are spread
   across a 2.5M-element array. Each access is essentially a cache miss.
3. **Metal's command encoding adds overhead** — Metal batches GPU
   commands through a command buffer; every scatter operation requires a
   full encode-dispatch-wait cycle, unlike CUDA's inline kernel execution.

**Measured**: ~150 ms per `scatter_add_` call on M2 Ultra — **3,000x
slower** than the same operation on CUDA.

### The Cascading Effect

The midrank algorithm calls `scatter_add_` 3 times (sum ranks, count
members, map back) inside `_midranks_vectorized`, which is itself called
3 times (pooled ranks, case within-ranks, control within-ranks). The
flattening trick that makes CUDA fast (process all 5,000 columns in one
scatter) makes MPS *worse* because it creates a single huge sparse array.

**Result**: Batch AUC for 5,000 markers takes ~0.015s on CUDA, ~20s on
MPS, vs ~0.9s on CPU. MPS is 22x slower than CPU.

---

## What We Did About It

### pystatsbio `batch_auc`

- `backend='gpu'` on CUDA: uses the vectorized `scatter_add_` kernel.
  49-63x faster than CPU.
- `backend='gpu'` on MPS: **raises `RuntimeError`** with an actionable
  message explaining why. Fail fast, fail loud (Coding Bible Rule 1).
- `backend='auto'` on MPS: routes to CPU silently (auto means "best
  available", and CPU *is* best on MPS for this workload).

### Could It Be Fixed for MPS?

Yes, but it requires a **completely different algorithm**:

- **Don't scatter.** Use `torch.unique_consecutive` on the sorted data
  to group ties *in-place* without random memory access. Then use
  `cumsum` to compute group sizes and rank sums. All operations are
  sequential/streaming, which Metal handles well.
- **Or process columns one at a time** — 5,000 iterations of a
  500-element sort + midrank is fast even in Python because each
  operation is small and cache-friendly.
- **Or use the CPU.** `scipy.stats.rankdata` is Cython running in L1
  cache on a single core. For 500 elements it takes ~0.13ms. Even
  looping 5,000 times: 650ms. The CPU wins not because it's "faster"
  than the GPU, but because the problem isn't parallel enough to
  justify GPU dispatch overhead for this workload shape.

The fundamental issue isn't Apple Silicon's compute power — it's that
Metal's programming model doesn't expose the low-level atomic memory
operations that make scatter patterns efficient on CUDA. This is a
deliberate Apple design choice (simpler programming model, optimized for
graphics/ML inference, not scientific computing). It may improve in
future Metal versions, but today, if your algorithm requires
`scatter_add_` into sparse targets, MPS is the wrong backend.

---

## General Rules for GPU Backend Selection

Based on validation across Mac Studio M2 Ultra and Linux RTX 5070 Ti:

### Operations That Are Fast on Both CUDA and MPS

- Matrix multiply (`X.T @ X`, `X @ beta`)
- Cholesky decomposition and triangular solves
- Element-wise operations (add, multiply, exp, log)
- Reductions (sum, mean, max along a dimension)
- `argsort` (used for ranking)
- `torch.rand`, `torch.randint` (random number generation)

### Operations That Are Fast on CUDA but Slow on MPS

- **`scatter_add_` with sparse/irregular indices** — the specific killer
- `scatter_` in general with non-contiguous write patterns
- Any operation that requires atomic read-modify-write to random locations
- Operations that create very large intermediate tensors with irregular
  access patterns

### When GPU Wins Over CPU

- **Large matrix operations**: n × p with n > 10,000 and p > 50.
  The GPU amortizes transfer overhead. Below this, CPU is often faster.
- **Embarrassingly parallel tasks**: R=50,000 permutation tests where
  each permutation is independent. GPU computes all R at once.
- **Batch operations**: fitting 5,000 dose-response curves simultaneously,
  computing AUC for 20,000 genes at once.

### When CPU Wins Over GPU

- **Small problems**: n < 1,000. GPU launch overhead dominates.
- **Sequential algorithms**: iterative methods where each step depends
  on the previous (e.g., Cox PH Newton-Raphson with few iterations).
- **Sparse scatter patterns on MPS**: see above.
- **User-supplied Python callbacks**: bootstrap/permutation with arbitrary
  statistic functions — the function runs on CPU regardless.

---

## Benchmark Reference

All benchmarks measured on Forge (RTX 5070 Ti, CUDA 12.0) and Mainframe
(Mac Studio M2 Ultra) during the April 2026 Linux/NVIDIA validation.

### Regression (pystatistics)

| Problem | CPU | GPU (CUDA) | Speedup |
|---------|-----|------------|---------|
| OLS 500K × 200 | 5.4s | 0.13s | **42x** |
| OLS 1M × 100 | 5.4s | 0.12s | **44x** |
| GLM binomial 50K × 100 | 0.4s | 0.08s | **5x** |

### Batch AUC (pystatsbio)

| Markers × Samples | CPU | GPU (CUDA) | GPU (MPS) | CUDA Speedup |
|--------------------|-----|------------|-----------|--------------|
| 100 × 1,155 | 0.018s | 0.9s | N/A | 0.02x (CPU wins) |
| 1,000 × 1,155 | 0.18s | 0.003s | N/A | **63x** |
| 20,000 × 1,155 | 3.6s | 0.074s | N/A | **49x** |

### Permutation Test (pystatistics)

| n samples | R perms | CPU | GPU (CUDA) | Speedup |
|-----------|---------|-----|------------|---------|
| 1,000 | 50,000 | 1.4s | 0.28s | **5x** |
| 10,000 | 50,000 | 6.7s | 0.29s | **23x** |
| 50,000 | 50,000 | 33s | 1.4s | **23x** |
