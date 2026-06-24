"""Per-evaluation host<->device sync overhead for the GPU MVN MLE objective.

WHY THIS EXISTS
---------------
The direct (gradient-based) MVN MLE runs a host-side optimiser (scipy L-BFGS-B)
that drives a device objective. Every accepted evaluation needs the scalar
objective and its gradient back on the host, which forces a host<->device
synchronisation. Historically that was paid TWICE per evaluation — once for the
objective (``compute_objective`` -> ``.item()``) and once for the gradient
(``compute_gradient`` -> ``.cpu()``) — and the gradient pass redundantly
recomputed the forward Cholesky. On Apple Metal (MPS), where each sync is high
latency, that doubled cost dominated the end-to-end fit at large p: the real
optimiser per-eval was ~2.7x the isolated device compute.

The objective now exposes ``compute_value_and_gradient``, which produces both in
ONE device pass and returns them through a single coalesced device->host copy.
``run_scaled_minimize`` drives scipy with that single ``jac=True`` callable, so
each evaluation pays ONE sync, not two.

This benchmark measures the two per-evaluation paths head to head on whatever GPU
is available (MPS or CUDA) and reports the speedup, so a regression in the
transfer layer (an accidental extra ``.item()``/``.cpu()``, or reverting to the
two-callable driver) shows up as the ratio collapsing back toward 1.0.

Run (on a machine with MPS or CUDA):
    python benchmarks/mvnmle_gpu_per_eval.py            # p=100, n=50000
    python benchmarks/mvnmle_gpu_per_eval.py --p 50 --n 30000 --reps 8

# NON-DETERMINISTIC: wall-clock timing depends on the machine, thermal state,
# and the GPU driver. This is a benchmark, not a pass/fail test; the correctness
# guarantee (fused == separate, to rounding) lives in
# tests/mvnmle/test_gpu_batched_equiv.py and test_optimize.py.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from pystatistics.mvnmle._objectives.gpu_fp32 import GPUObjectiveFP32


def _select_device():
    """Return the active GPU device string, or raise (Rule 1 — no silent CPU).

    A CPU run would not exercise the host<->device sync this benchmark exists to
    measure, so refuse rather than produce a meaningless number.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.cuda.synchronize
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.mps.synchronize
    raise RuntimeError(
        "No GPU (CUDA or MPS) available — this benchmark measures host<->device "
        "sync overhead and is meaningless on CPU."
    )


def _make_data(p: int, n: int, missing: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, p))
    sigma = A @ A.T + p * np.eye(p)
    mu = rng.standard_normal(p)
    X = mu + rng.standard_normal((n, p)) @ np.linalg.cholesky(sigma).T
    X[rng.random((n, p)) < missing] = np.nan
    return X


def run(p: int, n: int, missing: float, reps: int, seed: int) -> None:
    device, sync = _select_device()
    X = _make_data(p, n, missing, seed)
    obj = GPUObjectiveFP32(X, device=device)
    theta = obj.get_initial_parameters()
    n_patterns = int(obj._consts["obs_idx"].shape[0])

    # Warm up kernels/allocator so the timed loops measure steady state.
    obj.compute_objective(theta)
    obj.compute_gradient(theta)
    obj.compute_value_and_gradient(theta)
    sync()

    # OLD path: separate objective + gradient — two syncs, two forward passes.
    sync()
    t = time.perf_counter()
    for _ in range(reps):
        obj.compute_objective(theta)
        obj.compute_gradient(theta)
    sync()
    old = (time.perf_counter() - t) / reps

    # FUSED path: one call, one device pass, one coalesced device->host copy.
    sync()
    t = time.perf_counter()
    for _ in range(reps):
        obj.compute_value_and_gradient(theta)
    sync()
    fused = (time.perf_counter() - t) / reps

    print(f"device={device}  p={p}  n={n}  patterns={n_patterns}  reps={reps}")
    print(f"  per-eval OLD   (obj+grad, 2 syncs) = {old:.4f} s")
    print(f"  per-eval FUSED (value_and_grad)    = {fused:.4f} s")
    print(f"  speedup = {old / fused:.2f}x   saved/eval = {old - fused:.4f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p", type=int, default=100, help="number of variables")
    ap.add_argument("--n", type=int, default=50000, help="number of observations")
    ap.add_argument("--missing", type=float, default=0.15, help="missingness rate")
    ap.add_argument("--reps", type=int, default=6, help="evaluations per path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.p, args.n, args.missing, args.reps, args.seed)


if __name__ == "__main__":
    main()
