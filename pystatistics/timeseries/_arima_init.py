"""
User-supplied initial values for ARIMA optimization.

Implements R ``stats::arima``'s ``init=`` semantics for
:func:`pystatistics.timeseries.arima` (one module, one job: turning a
user init into a validated start vector):

- layout follows R's ``coef()`` order:
  ``[ar_1..ar_p, ma_1..ma_q, sar_1..sar_P, sma_1..sma_Q, mean?]``,
  where the mean slot exists only when a mean is actually estimated
  (``include_mean=True`` and ``d + D == 0``);
- ``numpy.nan`` entries are filled with defaults — zeros for AR/MA
  coefficients (R's rule) and the sample mean of the differenced
  series for the mean slot (R fills the intercept from a regression
  fit; same intent);
- AR blocks (non-seasonal and seasonal separately) must be
  stationary, mirroring R's ``arCheck`` (R errors with
  "non-stationary AR part");
- MA blocks are normalized to the invertible representative — R's
  documented ``maInvert``-on-init intent. (R's own implementation
  happens to ERROR on non-invertible MA inits because its CSS/optim
  stage diverges first; accepting and canonicalizing them is a
  strict, likelihood-equivalent improvement over the reference.)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.validation import check_array, check_1d
from pystatistics.timeseries._arima_factored import normalize_ma_coefficients
from pystatistics.timeseries._arima_likelihood import check_stationary


def prepare_init(
    init: ArrayLike,
    p: int, q: int, sp: int, sq: int,
    include_mean: bool,
    mean_default: float,
) -> NDArray:
    """Validate and canonicalize a user-supplied ``init`` vector.

    Parameters
    ----------
    init : ArrayLike
        Initial values in R ``coef()`` order:
        ``[ar_1..ar_p, ma_1..ma_q, sar_1..sar_P, sma_1..sma_Q, mean?]``.
        ``numpy.nan`` entries use the defaults described in the module
        docstring.
    p, q, sp, sq : int
        Non-seasonal and seasonal AR/MA orders (the factored orders,
        not the expanded polynomial lengths).
    include_mean : bool
        Whether a mean is estimated (i.e. after the d + D > 0
        override) — controls whether the trailing mean slot exists.
    mean_default : float
        Fill value for a ``nan`` mean slot (the sample mean of the
        differenced series). Ignored when ``include_mean`` is False.

    Returns
    -------
    NDArray
        Validated start vector in the factored layout (which equals
        the effective layout for non-seasonal models).

    Raises
    ------
    ValidationError
        On wrong length, non-finite (non-nan) entries, or a
        non-stationary AR / seasonal AR part (matching R's error).
    """
    arr = check_array(init, "init").ravel().astype(np.float64)
    check_1d(arr, "init")

    n_coef = p + q + sp + sq
    n_expected = n_coef + (1 if include_mean else 0)
    if len(arr) != n_expected:
        layout = f"[ar]*{p} + [ma]*{q} + [sar]*{sp} + [sma]*{sq}"
        if include_mean:
            layout += " + [mean]"
        raise ValidationError(
            f"init: expected length {n_expected} ({layout}), "
            f"got {len(arr)}"
        )

    filled = arr.copy()
    with np.errstate(invalid="ignore"):
        coef_nan = np.isnan(filled[:n_coef])
    filled[:n_coef][coef_nan] = 0.0
    if include_mean and np.isnan(filled[-1]):
        filled[-1] = mean_default
    if not np.all(np.isfinite(filled)):
        raise ValidationError(
            "init: entries must be finite (or nan to use the default)"
        )

    if not check_stationary(filled[:p]):
        raise ValidationError("init: non-stationary AR part")
    if not check_stationary(filled[p + q:p + q + sp]):
        raise ValidationError("init: non-stationary seasonal AR part")

    ma_norm, _ = normalize_ma_coefficients(filled[p:p + q])
    sma_norm, _ = normalize_ma_coefficients(
        filled[p + q + sp:p + q + sp + sq]
    )
    filled[p:p + q] = ma_norm
    filled[p + q + sp:p + q + sp + sq] = sma_norm
    return filled
