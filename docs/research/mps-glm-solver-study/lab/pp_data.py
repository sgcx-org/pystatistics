"""Build + cache flchain person-period designs and their CPU fp64 reference fit.

One job: given a bin width (regime), return the person-period design (X_pp, y_pp),
the column index of the identifiable covariates, and the CPU float64 reference
coefficients from pystatistics' own binomial GLM. Cached to .npz so the matrix
runner does not rebuild/refit per rep.
"""

from __future__ import annotations

import os
import numpy as np

from drivers.survival import datasets
from drivers.survival._person_period import build_person_period

from pystatistics.regression.design import Design
import pystatistics.regression as reg

# bin width in days per named regime
REGIMES = {
    "yearly": 365.25,
    "quarterly": 91.3,
    "monthly": 30.4,
    "biweekly": 15.2,
}

_CACHE = os.path.join(os.path.dirname(__file__), "_cache")
os.makedirs(_CACHE, exist_ok=True)


def get_regime(name: str):
    """Return dict with X_pp (f64), y_pp (f64), n_intervals, n_cov, ref_coef (f64).

    ref_coef is the pystatistics CPU float64 binomial GLM fit on the identical
    design. n_cov is the number of trailing covariate columns (always
    identifiable); the leading n_intervals columns are the baseline-hazard
    one-hot.
    """
    bin_days = REGIMES[name]
    path = os.path.join(_CACHE, f"{name}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return {
            "X_pp": d["X_pp"], "y_pp": d["y_pp"],
            "n_intervals": int(d["n_intervals"]), "n_cov": int(d["n_cov"]),
            "ref_coef": d["ref_coef"],
        }

    time, event, X, names = datasets.load_flchain()
    bounds = datasets.flchain_interval_bounds(time, event, bin_days)
    X_pp, y_pp, n_intervals = build_person_period(time, event, X, bounds)
    n_cov = X.shape[1]

    # CPU float64 reference — pystatistics' own binomial GLM on the identical design.
    res = reg.fit(Design.from_arrays(X_pp, y_pp), family="binomial", backend="cpu")
    ref_coef = np.asarray(res.coefficients, dtype=np.float64)

    np.savez(path, X_pp=X_pp, y_pp=y_pp,
             n_intervals=np.int64(n_intervals), n_cov=np.int64(n_cov),
             ref_coef=ref_coef)
    return {"X_pp": X_pp, "y_pp": y_pp, "n_intervals": n_intervals,
            "n_cov": n_cov, "ref_coef": ref_coef}


def covariate_slice(n_intervals: int, n_cov: int) -> slice:
    """Index of the trailing covariate columns (always identifiable)."""
    return slice(n_intervals, n_intervals + n_cov)
