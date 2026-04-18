# GPU Backend Convention

This document defines the input/output and dispatch conventions that
every GPU-capable module in pystatistics follows. Authors of new GPU
backends: follow this contract so users get consistent semantics
across the library.

Two exemplars to read alongside this doc:

- `pystatistics/multivariate/_pca.py` + `backends/gpu_pca.py`
- `pystatistics/multinomial/_solver.py` + `backends/gpu_likelihood.py`

## 0. When to add a GPU backend — and when not to

A GPU backend exists because the underlying computation is a good fit
for GPU hardware, not because the module is public and therefore
"deserves" one. Adding a GPU backend that doesn't make algorithmic
sense is worse than having none at all: it invites users to pick a
slower path, adds a maintenance surface, and forces convention
compliance (Rules 1–9 below) for no benefit.

**Reasonable targets:** large dense linear algebra (QR / SVD /
Cholesky), per-iteration gradient evaluations on large design
matrices, big-N likelihood evaluations, batched independent fits,
frequency-domain transforms.

**Not reasonable targets — do not add a GPU backend for these:**

- Many-small-fits workflows where per-call launch overhead dominates
  compute (e.g. ANOVA SS computation, which fits a chain of tiny
  regressions — the CPU LAPACK call is microseconds, the GPU kernel
  launch is milliseconds).
- Inherently sequential algorithms without exploitable parallelism
  (most EM variants below a certain dataset size, Cox PH partial
  likelihood).
- Algorithms already bounded by PCIe / D2H transfer, where the
  compute is cheap enough that you spend more time copying data than
  running the kernel.
- Algorithms where the existing scipy / LAPACK implementation is
  already within an order of magnitude of hardware peak — a 2× speedup
  is not worth the convention-compliance + test + review cost.

**Signal that GPU makes sense:** measured CPU time > 100 ms on a
representative input, with > 80 % of that time in a compute kernel
that maps to cuBLAS / cuSOLVER / cuFFT. Below that threshold, the
launch + transfer overhead eats the win.

If a public module has no `backends/gpu*.py`, that is a deliberate
statement — like `anova`, `timeseries.ets`, `survival.coxph`,
`multivariate.factor_analysis`, and the acf / stationarity families.
The public API correspondingly does *not* expose a `backend=`
parameter, which is the correct state (Rule 1: don't offer choices
you can't honor; exposing `backend='gpu'` on a CPU-only module would
violate fail-fast).

## 1. Input type determines default backend

The top-level fit function accepts either a numpy array *or* a
`torch.Tensor`. The `backend` keyword argument is declared with a
`None` sentinel default:

```python
def fit(X: ArrayLike | "torch.Tensor", *, backend: str | None = None, ...):
```

Resolve `backend` from the input type when unspecified:

| Input | `backend` unspecified → resolves to |
|---|---|
| numpy array / CPU `DataSource` array | `"cpu"` |
| torch.Tensor on GPU (`cuda` / `mps`) | `"gpu"` |
| torch.Tensor on CPU (rare) | `"cpu"` |

**Why:** creating a GPU `DataSource` via `ds.to("cuda")` is already the
opt-in to GPU. Requiring the user to also type `backend="gpu"` is
redundant friction. numpy input defaults to `cpu` because pystatistics
targets regulated industries where "unspecified backend" must mean the
R-reference path.

## 2. Explicit `backend="cpu"` with a GPU tensor raises

No silent device migration. Rule 1 applies — an explicit user request
that can't be honored should fail loudly:

```python
raise ValidationError(
    "backend='cpu' was specified but X is a torch.Tensor "
    f"on device {X.device}. Either pass a numpy array / "
    "CPU DataSource to the CPU backend, or call `.to('cpu')` "
    "on the DataSource explicitly to move it back."
)
```

## 3. Tensor input skips numpy validation

`check_array` / `check_finite` are numpy-only helpers. For tensor
input, validate directly via torch primitives:

```python
import sys as _sys
_is_torch_tensor = (
    "torch" in _sys.modules
    and isinstance(X, _sys.modules["torch"].Tensor)
)
if _is_torch_tensor:
    import torch
    if X.ndim != 2:
        raise ValidationError(...)
    if not torch.isfinite(X).all():
        raise ValidationError(...)
```

The `_is_torch_tensor` check uses `sys.modules` to avoid force-
importing torch on the numpy path (users on CPU-only installs must
not pay torch's ~800 ms cold import cost).

## 4. Keep moments on GPU until final transfer

When the tensor path runs, computing per-column means / SDs produces
tiny per-column vectors, but naively pulling them to CPU each fit
triggers a synchronous D2H sync point — each one blocks on prior
GPU work draining. A tight sequence of 5 such syncs costs ~200 ms on
an RTX-class GPU for a 1M × 100 matrix even though the data copied
is 100 floats.

Solution: keep moments as GPU tensors and transfer once, at the end,
in the result-assembly step:

```python
# In the fit body, moments stay on GPU:
col_means_gpu = X.mean(dim=0)

# In the result finalizer:
if isinstance(col_means_cpu, torch.Tensor):
    col_means_cpu = col_means_cpu.detach().cpu().numpy()
```

## 5. Output dtype tracks compute dtype

When `use_fp64=False`, return FP32 outputs rather than force-promoting
to FP64 at the transfer boundary. Promoting a 400 MB FP32 scores
tensor to FP64 on GPU doubles the D2H payload to 800 MB and adds
~140 ms of PCIe traffic for "precision" the fit never actually
computed. The `GPU_FP32` tolerance tier in the project's tolerances
module already documents that FP32-on-GPU and FP64-on-CPU are
statistically equivalent but not bitwise-equal.

## 6. Tolerance against CPU uses the tier from `core.compute.tolerances`

GPU-vs-CPU tests assert `GPU_FP32.rtol` / `GPU_FP32.atol`, not
arbitrary tight numbers. The tier is the project's published contract:

```python
from pystatistics.core.compute.tolerances import GPU_FP32
np.testing.assert_allclose(cpu_val, gpu_val,
                           rtol=GPU_FP32.rtol, atol=GPU_FP32.atol)
```

## 7. Minimum tolerance floor in FP32

FP32 gradient precision is ~1e-7, so `L-BFGS-B` with the numerical
default `ftol=1e-8` / `gtol=1e-8` will report `ABNORMAL` (line-search
stall on the noise floor) routinely on real data. When running FP32,
floor `tol` at 1e-5:

```python
gpu_fp32_min_tol = 1e-5
effective_tol = tol
if backend != "cpu" and not use_fp64 and tol < gpu_fp32_min_tol:
    effective_tol = gpu_fp32_min_tol
```

A user who explicitly asks for FP32 can't reasonably demand FP64
gradient convergence.

## 8. Tests

Every GPU backend ships a `Test<Module>GPU` class mirroring the
exemplars:

- `test_invalid_backend_raises` (bad backend string)
- `test_gpu_unavailable_raises_explicitly` (monkeypatched `detect_gpu`,
  `backend="gpu"` must raise RuntimeError)
- `test_auto_backend_falls_back_to_cpu_when_no_gpu` (monkeypatched,
  `backend="auto"` must succeed)
- `test_gpu_fp64_matches_cpu_<property>` (CUDA only, skip on MPS)
- `test_gpu_fp32_matches_cpu_at_tier` (uses `GPU_FP32` tolerances)
- `test_gpu_datasource_input_matches_gpu_numpy` (amortized path
  equivalence)
- `test_gpu_tensor_with_cpu_backend_raises` (Rule 1 enforcement)

## 9. CHANGELOG entry template

For each new GPU backend, log:

- File path of the new backend module
- Measured per-fit latency table: CPU vs GPU (numpy per-call) vs GPU
  (DataSource amortized), for representative shapes
- Precision tier used (FP32 default, FP64 on CUDA for parity)
- Any algorithmic change (like analytical-Hessian in multinomial)
  that also benefits the CPU path, and whether it's been backported
