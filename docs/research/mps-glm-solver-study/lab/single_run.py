"""Run ONE (solver, regime) once, in this process, and print a compact JSON line.

Purpose: the not-PD Cholesky failure is intermittent at PROCESS granularity (fp32
reduction-order nondeterminism near the PD edge) -- within one process all reps
are identical, so intermittency must be sampled across independent process
launches. A shell loop invokes this N times and tallies the verdicts.

Usage: python -m lab.single_run <solver> <regime>
"""
import sys
import json
import numpy as np
from .pp_data import get_regime, covariate_slice
from .run_matrix import _run_one, _verdict
from .irls_common import (host_rel_newton_decrement, coef_max_rel_err,
                          FP32_REL_DECREMENT_TOL)

solver, regime = sys.argv[1], sys.argv[2]
d = get_regime(regime)
cov = covariate_slice(d["n_intervals"], d["n_cov"])
coef, info = _run_one(solver, d)
if info["raised"]:
    err = float("nan"); nd = float("nan"); accepted = False
else:
    err = coef_max_rel_err(coef, d["ref_coef"], cov)
    nd = host_rel_newton_decrement(d["X_pp"], d["y_pp"], coef)
    accepted = nd < FP32_REL_DECREMENT_TOL
print(json.dumps({
    "solver": solver, "regime": regime,
    "verdict": _verdict(info["raised"], accepted, err),
    "raised": bool(info["raised"]), "coef_max_rel_err": err,
    "rel_newton_dec": nd, "outer_iters": info["outer_iters"],
    "wall_s": info["wall_s"],
}))
