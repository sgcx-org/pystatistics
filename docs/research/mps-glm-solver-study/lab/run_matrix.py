"""Run the full solver x regime x reps matrix and write results to JSON.

One job: orchestrate every (solver, regime) over N reps, compute the per-run
metrics (converged/accepted, coef error vs CPU fp64 ref, host Newton decrement,
wall, iters), classify the fail-loud verdict, and dump raw + summarized JSON for
the report. No plotting, no policy logic -- just evidence collection.
"""

from __future__ import annotations

import argparse
import json
import os
import numpy as np

from .pp_data import get_regime, covariate_slice, REGIMES
from .solver_scoring import (run_scoring, inner_cholesky, inner_cg,
                             make_inner_hybrid, make_inner_cpu_qr)
from .solver_lm import run_lm
from .irls_common import (host_rel_newton_decrement, coef_max_rel_err,
                          FP32_REL_DECREMENT_TOL)

FP32_TIER = 1e-3   # generous fp32 coef-tier ceiling for "correct" classification


def _run_one(solver, d):
    """Dispatch a single solver run; return (coef, info)."""
    X, y = d["X_pp"], d["y_pp"]
    if solver == "cholesky":
        return run_scoring(X, y, inner_cholesky)
    if solver == "cg":
        return run_scoring(X, y, inner_cg)
    if solver == "hybrid":
        return run_scoring(X, y, make_inner_hybrid())
    if solver == "cpu_qr":
        return run_scoring(X, y, make_inner_cpu_qr(X.astype(np.float64)))
    if solver == "lm":
        return run_lm(X, y)
    raise ValueError(solver)


def _verdict(raised, accepted, err):
    """One of: fail_loud, converged, silent_wrong (disqualifying)."""
    if raised:
        return "fail_loud"
    if not accepted:
        return "fail_loud"          # gate refused -> caller must raise in prod
    return "converged" if err <= FP32_TIER else "silent_wrong"


def run(solvers, regimes, reps, out_path):
    results = []
    for regime in regimes:
        d = get_regime(regime)
        cov = covariate_slice(d["n_intervals"], d["n_cov"])
        for solver in solvers:
            for rep in range(reps):
                coef, info = _run_one(solver, d)
                if info["raised"]:
                    err = float("nan"); nd = float("nan"); accepted = False
                else:
                    err = coef_max_rel_err(coef, d["ref_coef"], cov)
                    nd = host_rel_newton_decrement(d["X_pp"], d["y_pp"], coef)
                    accepted = nd < FP32_REL_DECREMENT_TOL
                v = _verdict(info["raised"], accepted, err)
                cgi = info.get("cg_iters")
                rec = {
                    "regime": regime, "solver": solver, "rep": rep,
                    "n_rows": int(d["X_pp"].shape[0]), "n_cols": int(d["X_pp"].shape[1]),
                    "raised": bool(info["raised"]), "reason": info.get("reason", ""),
                    "accepted": bool(accepted), "coef_max_rel_err": err,
                    "rel_newton_dec": nd, "verdict": v,
                    "outer_iters": info["outer_iters"], "wall_s": info["wall_s"],
                    "cg_iters_mean": (sum(cgi) / len(cgi)) if cgi else None,
                    "cg_fallbacks": info.get("cg_fallbacks"),
                    "lm_inner_total": info.get("lm_inner_total"),
                    "lm_final_lambda": info.get("lm_final_lambda"),
                    "xfer_s": info.get("xfer_s"), "cpu_solve_s": info.get("cpu_solve_s"),
                }
                results.append(rec)
                print(f"[{regime:9s} {solver:9s} rep{rep}] {v:12s} "
                      f"err={err:.3e} nd={nd:.3e} outer={info['outer_iters']} "
                      f"{info['wall_s']:.1f}s", flush=True)

    summary = _summarize(results)
    with open(out_path, "w") as f:
        json.dump({"results": results, "summary": summary,
                   "fp32_rel_decrement_tol": FP32_REL_DECREMENT_TOL,
                   "fp32_tier": FP32_TIER}, f, indent=2)
    print(f"\nWrote {out_path} ({len(results)} runs)")
    return summary


def _summarize(results):
    keys = sorted({(r["regime"], r["solver"]) for r in results})
    out = {}
    for regime, solver in keys:
        runs = [r for r in results if r["regime"] == regime and r["solver"] == solver]
        verds = [r["verdict"] for r in runs]
        errs = [r["coef_max_rel_err"] for r in runs if np.isfinite(r["coef_max_rel_err"])]
        walls = [r["wall_s"] for r in runs]
        out[f"{regime}/{solver}"] = {
            "n": len(runs),
            "converged": verds.count("converged"),
            "fail_loud": verds.count("fail_loud"),
            "silent_wrong": verds.count("silent_wrong"),
            "err_min": min(errs) if errs else None,
            "err_max": max(errs) if errs else None,
            "wall_med": float(np.median(walls)),
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solvers", default="cholesky,cg,lm,hybrid,cpu_qr")
    ap.add_argument("--regimes", default="yearly,quarterly,monthly,biweekly")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--out", default="matrix_mps.json")
    a = ap.parse_args()
    for r in a.regimes.split(","):
        assert r in REGIMES, r
    run(a.solvers.split(","), a.regimes.split(","), a.reps, a.out)
