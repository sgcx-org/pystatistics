"""
Distributional validation of categorical MICE against R's `mice` (3.19.0).

Imputes the same mixed-type dataset with logreg/polyreg/polr and checks that the
marginal distribution of imputed category codes matches R's, per column. As with
the numeric validation, agreement is distributional (same models, same defaults)
rather than RNG-stream parity.

Skips if the R fixtures are absent (run generate_categorical_fixtures.R).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pystatistics.mice import mice, MICEDesign

REF_DIR = Path(__file__).parent / "references"
REF_JSON = REF_DIR / "mice_categorical_reference.json"
DATA_CSV = REF_DIR / "mice_categorical_data.csv"

pytestmark = pytest.mark.skipif(
    not REF_JSON.exists() or not DATA_CSV.exists(),
    reason="R categorical fixtures absent (run generate_categorical_fixtures.R)",
)

_KINDS = ["numeric", "binary", "categorical", "ordered"]
_COL_BY_NAME = {"bin": 1, "nom": 2, "ord": 3}


@pytest.fixture(scope="module")
def reference():
    with open(REF_JSON) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def solution(reference):
    matrix = np.genfromtxt(DATA_CSV, delimiter=",", skip_header=1)
    design = MICEDesign.from_array(matrix, column_kinds=_KINDS)
    meta = reference["meta"]
    return design, mice(design, m=meta["m"], maxit=meta["maxit"], seed=20260614)


def _imputed_proportions(sol, design, col, levels):
    imp = sol.imputations(col).ravel()  # imputed category codes across m
    counts = np.array([np.sum(imp == float(lv)) for lv in levels], dtype=float)
    return counts / counts.sum()


@pytest.mark.parametrize("name", ["bin", "nom", "ord"])
def test_imputed_category_proportions_match_r(reference, solution, name):
    design, sol = solution
    col = _COL_BY_NAME[name]
    levels = [int(lv) for lv in reference[name]["levels"]]
    ours = _imputed_proportions(sol, design, col, levels)
    r_prop = np.asarray(reference[name]["proportions"], dtype=float)
    # Marginal category proportions should match within Monte-Carlo + method
    # tolerance (independent RNG streams; m=50).
    np.testing.assert_allclose(ours, r_prop, atol=0.06, err_msg=(
        f"{name}: ours={np.round(ours,3)} R={np.round(r_prop,3)}"
    ))


def test_all_imputed_values_are_valid_codes(solution):
    design, sol = solution
    for name, col in _COL_BY_NAME.items():
        allowed = set(design.levels_for(col).tolist())
        assert set(np.unique(sol.imputations(col))).issubset(allowed)
