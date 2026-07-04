"""Dump PyStatistics' ZZZ-selected ETS fits for R cross-scoring.

Stage 1 of regenerating ``ets_r_reference.json``.  Runs the package's own
``ets()`` auto-selection on every reference series already stored in the
fixture and writes the selected model plus its fitted parameters and
initial states to ``ets_py_params.json``.  Stage 2
(``generate_ets_r_reference.R``) feeds these parameters through R
``forecast:::pegelsresid.C`` to score *our* fits under *R's own*
likelihood — the honest cross-engine comparison (refitting the same model
spec in R only re-runs R's optimiser, which is the thing being compared).

Run from the repo root (after the fixture exists, since it supplies the
input series)::

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=$PWD \
        python tests/fixtures/generate_ets_py_params.py

then::

    Rscript tests/fixtures/generate_ets_r_reference.R
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pystatistics.timeseries import ets

FIXTURE_DIR = Path(__file__).parent
REFERENCE = FIXTURE_DIR / "ets_r_reference.json"
OUT = FIXTURE_DIR / "ets_py_params.json"


def main() -> None:
    reference = json.loads(REFERENCE.read_text())
    out: dict[str, dict] = {}
    for name, case in reference["selection"].items():
        sol = ets(
            np.asarray(case["x"], dtype=np.float64),
            model=case["model_arg"],
            period=case["period"],
        )
        spec = sol.spec
        out[name] = {
            "picked": spec.name,
            "error": spec.error,
            "trend": spec.trend,
            "season": spec.season,
            "damped": spec.damped,
            "period": spec.period,
            "alpha": sol.alpha,
            "beta": sol.beta,
            "gamma": sol.gamma,
            "phi": sol.phi,
            "init_level": sol.init_level,
            "init_trend": sol.init_trend,
            # Engine order: oldest first (s_{1-m} .. s_0).  R's state
            # vector wants most-recent first — the R script reverses.
            "init_season": (
                list(sol.init_season) if sol.init_season is not None else None
            ),
            "aicc_py_convention": sol.aicc,
            "converged": bool(sol.converged),
        }
        print(f"  {name:24s} -> {spec.name}")
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(FIXTURE_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
