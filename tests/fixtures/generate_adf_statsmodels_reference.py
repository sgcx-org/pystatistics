"""Generates adf_statsmodels_reference.json — statsmodels ``adfuller``
p-value references for the ADF MacKinnon p-value fix (RIGOR R18).

Reads the series embedded in ``stationarity_r_reference.json`` (run
``generate_stationarity_r_reference.R`` first) and records the
statistic and MacKinnon (1994) p-value for every regression type and a
spread of fixed lag orders, covering both tails and the middle of the
p range.

Run from the repo root:
    python tests/fixtures/generate_adf_statsmodels_reference.py
statsmodels 0.14.6 was used for the committed fixture.
"""

import json
from pathlib import Path

import numpy as np
from statsmodels.tsa.stattools import adfuller

FIXTURES = Path(__file__).parent
REG_MAP = {"nc": "n", "c": "c", "ct": "ct"}  # pystatistics -> statsmodels

with open(FIXTURES / "stationarity_r_reference.json") as f:
    series = json.load(f)["series"]

cases = []
for name, values in series.items():
    x = np.asarray(values, dtype=np.float64)
    default_lag = int(np.trunc((len(x) - 1) ** (1.0 / 3.0)))
    for regression, sm_regression in REG_MAP.items():
        for lag in sorted({0, 2, default_lag}):
            stat, pvalue, *_ = adfuller(
                x, maxlag=lag, regression=sm_regression, autolag=None
            )
            cases.append(
                {
                    "series": name,
                    "regression": regression,
                    "n_lags": lag,
                    "statistic": stat,
                    "pvalue": pvalue,
                }
            )

import statsmodels

out = {"statsmodels_version": statsmodels.__version__, "cases": cases}
with open(FIXTURES / "adf_statsmodels_reference.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"wrote adf_statsmodels_reference.json ({len(cases)} cases)")
