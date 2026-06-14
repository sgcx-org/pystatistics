"""
The imputation-method contract.

Every univariate imputation method (PMM, norm, and — in Stage 3 — logreg,
polyreg, polr) conforms to a single protocol. The chained-equations sweep
(`_chain.py`) is written against this protocol and *never* names a concrete
method: it looks one up by name in the registry and calls ``impute``.

This is the seam that makes Stage 3 (categorical methods) additive rather than
invasive — a new method is a new file that registers itself; the sweep loop is
untouched.

Contract (CLAUDE.md Rule 2 — define your contracts):

    impute(y_obs, X_obs, X_mis, rng) -> imputed values for the missing rows

    Inputs (all owned/validated by the chain, not re-checked by the method):
      - y_obs : (n_obs,)   observed values of the target column
      - X_obs : (n_obs, q) predictors at the observed rows, NO intercept column;
                           fully filled (current imputation state), finite
      - X_mis : (n_mis, q) predictors at the missing rows, same columns as X_obs
      - rng   : numpy.random.Generator — the ONLY randomness source (Rule 6)

    Output:
      - (n_mis,) array of imputed values, one per missing row, in the row order
        of X_mis. The method owns the correctness of this output (Rule 2).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ImputationMethod(Protocol):
    """Protocol every imputation method must satisfy."""

    #: Stable, lowercase identifier used as the registry key and in
    #: user-facing ``method=`` arguments (e.g. ``'pmm'``, ``'norm'``).
    name: str

    #: Kind of target column this method imputes. Stage 1 ships only
    #: ``'numeric'`` methods; Stage 3 adds ``'binary'`` / ``'categorical'`` /
    #: ``'ordered'``. The design's per-column type must be compatible.
    target_kind: str

    def impute(
        self,
        y_obs: np.ndarray,
        X_obs: np.ndarray,
        X_mis: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return imputed values for the missing rows (see module docstring)."""
        ...
