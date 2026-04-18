"""Benchmark harness for pystatistics.mvnmle.

Measures wall-clock + iteration count across a representative
spectrum of shapes, for every (algorithm, backend) combination.

Shapes:
    apple        2-var, 18 obs — R-mvnmle reference case
    missvals     5-var, 13 obs — R-mvnmle reference case
    iris         4-var, 150 obs, synthetic MCAR — sklearn demo
    wine         13-var, 178 obs, synthetic MCAR — sklearn demo
                 (the Project Lacuna canary: 100+ patterns)
    breast       30-var, 569 obs, synthetic MCAR — sklearn demo
                 (Lacuna's real per-entry workload)

Run:
    python benchmarks/mvnmle_bench.py            # full sweep
    python benchmarks/mvnmle_bench.py --quick    # skip slow BFGS cases
    python benchmarks/mvnmle_bench.py --tag baseline > baseline.txt
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np


SEED = 0
MISSING_RATE = 0.15


@dataclass
class Case:
    name: str
    data_fn: object
    shape_hint: str
    slow_bfgs: bool


def _load_apple():
    from pystatistics.mvnmle.datasets import apple
    return apple.copy()


def _load_missvals():
    from pystatistics.mvnmle.datasets import missvals
    return missvals.copy()


def _load_sklearn(loader_name):
    from sklearn import datasets as sk
    X = getattr(sk, f"load_{loader_name}")().data.astype(float).copy()
    rng = np.random.default_rng(SEED)
    X[rng.random(X.shape) < MISSING_RATE] = np.nan
    X = X[~np.all(np.isnan(X), axis=1)]
    return X


CASES = [
    Case("apple",   _load_apple,                          "18 x 2",   False),
    Case("missvals", _load_missvals,                      "13 x 5",   False),
    Case("iris",    lambda: _load_sklearn("iris"),        "150 x 4",  False),
    Case("wine",    lambda: _load_sklearn("wine"),        "178 x 13", True),
    Case("breast",  lambda: _load_sklearn("breast_cancer"),"569 x 30", True),
]


def _gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def bench_one(data, algorithm, backend, max_iter, repeat=1):
    """Return dict with time_ms, n_iter, converged, loglik, err.

    Never raises — any exception becomes err=type-name.
    """
    from pystatistics.mvnmle import mlest

    # Warmup
    try:
        _ = mlest(data, algorithm=algorithm, backend=backend,
                  max_iter=max_iter, verbose=False)
    except Exception as e:
        return {"time_ms": None, "n_iter": None, "converged": False,
                "loglik": None, "err": type(e).__name__}

    times = []
    n_iter = None
    converged = None
    loglik = None
    for _ in range(repeat):
        t = time.perf_counter()
        try:
            r = mlest(data, algorithm=algorithm, backend=backend,
                      max_iter=max_iter, verbose=False)
            times.append(time.perf_counter() - t)
            n_iter = r.n_iter
            converged = r.converged
            loglik = r.loglik
        except Exception as e:
            return {"time_ms": None, "n_iter": None, "converged": False,
                    "loglik": None, "err": type(e).__name__}
    median_ms = 1000 * float(np.median(times))
    return {"time_ms": median_ms, "n_iter": n_iter, "converged": converged,
            "loglik": loglik, "err": None}


def bench_mcar_one(data, backend, repeat=1):
    """Benchmark little_mcar_test end-to-end (what Lacuna actually calls)."""
    from pystatistics.mvnmle import little_mcar_test
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = little_mcar_test(data, backend=backend)
    except Exception as e:
        return {"time_ms": None, "err": type(e).__name__, "stat": None}

    times = []
    stat = None
    for _ in range(repeat):
        t = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = little_mcar_test(data, backend=backend)
            times.append(time.perf_counter() - t)
            stat = r.statistic
        except Exception as e:
            return {"time_ms": None, "err": type(e).__name__, "stat": None}
    return {"time_ms": 1000 * float(np.median(times)),
            "err": None, "stat": stat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip BFGS on known-slow cases")
    ap.add_argument("--tag", default="",
                    help="label printed with each row (e.g. 'baseline')")
    ap.add_argument("--repeat", type=int, default=1,
                    help="repetitions per case; reports median")
    ap.add_argument("--max-iter", type=int, default=500)
    args = ap.parse_args()

    gpu = _gpu_available()
    backends = ["cpu"] + (["gpu"] if gpu else [])
    print(f"# GPU available: {gpu}")
    print(f"# Missing rate: {MISSING_RATE}, seed: {SEED}")
    print(f"# Tag: {args.tag!r}")
    print()

    header = f"{'case':<10} {'shape':<10} {'algo':<7} {'backend':<4} {'time_ms':>10} {'n_iter':>7} {'conv':>5} {'err':<15}"
    print(header)
    print("-" * len(header))

    for case in CASES:
        data = case.data_fn()
        for algorithm in ("em", "direct"):
            for backend in backends:
                if args.quick and algorithm == "direct" and case.slow_bfgs:
                    continue
                r = bench_one(data, algorithm, backend,
                              max_iter=args.max_iter, repeat=args.repeat)
                t = f"{r['time_ms']:.1f}" if r["time_ms"] is not None else "--"
                ni = r["n_iter"] if r["n_iter"] is not None else "--"
                cv = "y" if r["converged"] else ("--" if r["converged"] is None else "n")
                err = r["err"] or ""
                print(f"{case.name:<10} {case.shape_hint:<10} {algorithm:<7} "
                      f"{backend:<4} {t:>10} {ni:>7} {cv:>5} {err:<15}")

    print()
    print("# little_mcar_test end-to-end timings:")
    print(f"{'case':<10} {'shape':<10} {'backend':<4} {'time_ms':>10}")
    print("-" * 40)
    for case in CASES:
        data = case.data_fn()
        for backend in backends:
            r = bench_mcar_one(data, backend, repeat=args.repeat)
            t = f"{r['time_ms']:.1f}" if r["time_ms"] is not None else "--"
            err = r["err"] or ""
            print(f"{case.name:<10} {case.shape_hint:<10} {backend:<4} "
                  f"{t:>10}   {err}")


if __name__ == "__main__":
    main()
