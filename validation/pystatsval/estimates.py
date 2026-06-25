"""Reusable numeric summaries of fitted estimates.

One job: turn bulky estimate objects (e.g. a p x p covariance) into compact,
cross-engine-comparable scalars/vectors for a benchmark record. Storing the full
object in every record is O(p^2) and pointless for agreement checks at large p;
these summaries (Frobenius norm, log-determinant, diagonal) are enough.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# Above this dimension, the full p x p matrix is not stored in the record.
FULL_SIGMA_MAX_P = 100


def summarize_covariance(sigma: NDArray[np.floating] | None,
                         *, full_max_p: int = FULL_SIGMA_MAX_P) -> dict[str, Any]:
    """Compact, comparable summary of an estimated covariance matrix.

    Returns keys ``sigma_diag``, ``sigma_fro``, ``sigma_logdet`` (None if not
    positive-definite), and ``sigma_full`` (only when ``p <= full_max_p``). For
    ``sigma is None`` (a failed fit) every value is None.
    """
    if sigma is None:
        return {"sigma_diag": None, "sigma_fro": None,
                "sigma_logdet": None, "sigma_full": None}
    S = np.asarray(sigma, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"sigma must be square 2-D, got shape {S.shape}")
    p = S.shape[0]
    sign, logabsdet = np.linalg.slogdet(S)
    return {
        "sigma_diag": np.diag(S).tolist(),
        "sigma_fro": float(np.linalg.norm(S, "fro")),
        "sigma_logdet": (float(logabsdet) if sign > 0 else None),
        "sigma_full": (S.tolist() if p <= full_max_p else None),
    }
