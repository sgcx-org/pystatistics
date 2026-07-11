"""Tensor-product basis (``te``) coordinate-exact check vs mgcv smoothCon.

The unconstrained ``te`` tensor of ``cr`` margins is coordinate-stable
(the marginal ``cr`` bases match mgcv exactly, the row-wise Kronecker order
is fixed, and the tensor-level ``scale.penalty`` is deterministic), so its
basis matrix and penalty list match ``smoothCon(te, absorb.cons=FALSE)`` to
machine precision. ``ti`` and isotropic ``s(x, z)`` involve eigen/centring
rotations and are validated in function space by the full-fit tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pystatistics.gam._basis_te import assemble_tensor

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _load():
    return json.loads((_FIXTURES / "gam_tensor_mgcv.json").read_text())


def test_te_unconstrained_basis_matches_mgcv():
    d = _load()
    x = np.array(d["x"])
    z = np.array(d["z"])
    ref = d["te_basis"]
    X, S_blocks, _ = assemble_tensor(
        [x, z], ks=[5, 4], bss=["cr", "cr"], interaction=False,
    )
    assert X.shape == (len(x), 20)
    assert np.allclose(X, np.array(ref["X"]), atol=1e-9)
    assert np.allclose(S_blocks[0], np.array(ref["S1"]), atol=1e-9)
    assert np.allclose(S_blocks[1], np.array(ref["S2"]), atol=1e-9)


def test_te_penalty_ranks_and_overlap():
    d = _load()
    x = np.array(d["x"])
    z = np.array(d["z"])
    _, S_blocks, _ = assemble_tensor(
        [x, z], ks=[5, 4], bss=["cr", "cr"], interaction=False,
    )
    # cr margin penalty rank = k - 2: margin ranks 3 and 2.
    # S_x (x) I_kz has rank 3*4 = 12; I_kx (x) S_z has rank 5*2 = 10.
    assert np.linalg.matrix_rank(S_blocks[0]) == 12
    assert np.linalg.matrix_rank(S_blocks[1]) == 10
    # Joint null space = null_x (x) null_z, dim 2*2 = 4 -> combined rank 16.
    assert np.linalg.matrix_rank(S_blocks[0] + S_blocks[1]) == 20 - 4
