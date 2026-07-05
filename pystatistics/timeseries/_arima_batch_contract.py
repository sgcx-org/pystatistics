"""Per-series failure contract for batched ARMA fits.

``arima_batch`` runs one fit per row of a (K, n) batch, on whichever
backend the user selected. This module owns the single failure contract
both backends must implement, so that backend choice can never change
failure semantics:

- A series **fails** when its backend cannot certify a valid
  *stationary* ARMA solution for it: the CPU loop's single-series
  fitter raised ``ConvergenceError``, or the GPU batch fitter returned
  an AR estimate whose polynomial has a root inside the unit circle
  (checked host-side in float64, independent of the torch build).
- If **every** series fails, the batch call raises ``ConvergenceError``
  — the same behavior a single-series ``arima(method='Whittle')`` call
  has on the same input.
- If **some** series fail, the failed rows' estimates (``ar``, ``ma``,
  ``sigma2``, ``mean``) are set to NaN and their ``converged`` flag is
  forced False, a ``UserWarning`` naming the count is emitted, and the
  message is recorded in the solution's ``warnings`` envelope. NaN
  poisons any downstream arithmetic, so a failed fit cannot be
  consumed as if it were a good one.
- If **no** series fails, inputs pass through unchanged and no warning
  is emitted.

The stationarity criterion is deliberately *not* the per-series
optimizer ``converged`` flag: on the float32 GPU path the Adam
gradient tolerance routinely stays above ``tol`` on perfectly good
fits (the documented fp32 behavior), and conversely a series can
reach gradient tolerance at a non-stationary mirror-basin optimum of
the Whittle likelihood. Validity of the returned model — all AR roots
outside the unit circle, the same check the single-series path
enforces by raising — is the contract.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ConvergenceError
from pystatistics.timeseries._whittle import _check_ar_stationarity

_FAILURE_REMEDY = (
    "A failed series usually signals the wrong order or inadequate "
    "differencing — try increasing ``d`` or reducing ``p``."
)


def nonstationary_rows(ar: NDArray) -> NDArray:
    """Flag batch rows whose AR estimate is non-stationary.

    Parameters
    ----------
    ar : NDArray
        AR coefficients, shape ``(K, p)``. Any float dtype; the root
        check runs in float64 on the host regardless of the backend
        that produced the estimates.

    Returns
    -------
    NDArray
        Boolean mask of shape ``(K,)`` — True where the AR polynomial
        has a root inside (or on) the unit circle. All-False for
        ``p == 0`` (no AR part, nothing to be non-stationary).
    """
    K = ar.shape[0]
    failed = np.zeros(K, dtype=bool)
    if ar.shape[1] == 0:
        return failed
    ar64 = np.asarray(ar, dtype=np.float64)
    for k in range(K):
        failed[k] = not _check_ar_stationarity(ar64[k])
    return failed


def enforce_batch_failure_contract(
    *,
    ar: NDArray,
    ma: NDArray,
    sigma2: NDArray,
    mean: NDArray | None,
    converged: NDArray,
    failed: NDArray,
    n_iter: int,
    backend_label: str,
) -> tuple[NDArray, NDArray, NDArray, NDArray | None, NDArray, tuple[str, ...]]:
    """Apply the shared per-series failure contract to raw batch output.

    Parameters
    ----------
    ar, ma, sigma2, mean, converged
        Raw per-series estimates from a backend, batch-leading shapes
        ``(K, p)`` / ``(K, q)`` / ``(K,)`` / ``(K,)`` or None / ``(K,)``.
        Never mutated — failed rows are NaN'd on copies.
    failed : NDArray
        Boolean mask of shape ``(K,)`` — True where the backend could
        not certify a valid stationary fit for that series.
    n_iter : int
        Iterations used (for the all-failed ``ConvergenceError``).
    backend_label : str
        Human-readable backend name for messages, e.g.
        ``'Whittle-loop-CPU'``.

    Returns
    -------
    (ar, ma, sigma2, mean, converged, warnings_tuple)
        The estimates with failed rows set to NaN and their
        ``converged`` flag forced False, plus the warning messages to
        record in the ``Result`` envelope (empty when nothing failed).

    Raises
    ------
    ConvergenceError
        If every series in the batch failed.
    """
    K = int(failed.shape[0])
    n_failed = int(np.sum(failed))

    if n_failed == 0:
        return ar, ma, sigma2, mean, converged, ()

    if n_failed == K:
        raise ConvergenceError(
            f"arima_batch ({backend_label}): all {K} series failed to "
            f"produce a valid stationary Whittle ARMA fit (non-stationary "
            f"AR optimum or optimizer failure). {_FAILURE_REMEDY}",
            iterations=int(n_iter),
            reason="all series failed: non-stationary AR or optimizer failure",
        )

    ar = ar.copy()
    ma = ma.copy()
    sigma2 = sigma2.copy()
    converged = converged.copy()
    ar[failed] = np.nan
    ma[failed] = np.nan
    sigma2[failed] = np.nan
    if mean is not None:
        mean = mean.copy()
        mean[failed] = np.nan
    converged[failed] = False

    msg = (
        f"arima_batch ({backend_label}): {n_failed} of {K} series failed "
        f"to produce a valid stationary Whittle ARMA fit; their ar/ma/"
        f"sigma2/mean estimates are set to NaN and converged=False. "
        f"Identify them with np.isnan(result.sigma2). {_FAILURE_REMEDY}"
    )
    # stacklevel: 1=here, 2=arima_batch, 3=the user's call site.
    warnings.warn(msg, UserWarning, stacklevel=3)
    return ar, ma, sigma2, mean, converged, (msg,)
