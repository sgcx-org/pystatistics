"""Standalone: pystatistics backend='gpu_fp64' per regime vs CPU fp64 reference."""
import json
import numpy as np
from lab.pp_data import get_regime, covariate_slice
import pystatistics.regression as reg
from pystatistics.regression.design import Design

fp64 = {}
for regime in ["yearly", "quarterly", "monthly", "biweekly"]:
    d = get_regime(regime)
    cov = covariate_slice(d["n_intervals"], d["n_cov"])
    res = reg.fit(Design.from_arrays(d["X_pp"], d["y_pp"]),
                  family="binomial", backend="gpu_fp64")
    coef = np.asarray(res.coefficients, dtype=np.float64)
    r = d["ref_coef"]
    err = float(np.max(np.abs(coef[cov] - r[cov]) / np.maximum(np.abs(r[cov]), 1e-8)))
    fp64[regime] = {"gpu_fp64_vs_cpu_fp64_max_rel_err": err,
                    "n_rows": int(d["X_pp"].shape[0]), "n_cols": int(d["X_pp"].shape[1])}
    print(f"[gpu_fp64 {regime}] err vs CPU fp64 = {err:.3e}", flush=True)

json.dump(fp64, open("fp64_ref.json", "w"), indent=2)
print("wrote fp64_ref.json")
