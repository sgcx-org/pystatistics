"""
ETS state-space forward recursion (the hot inner loop) — public import surface.

The ETS one-step recursion — one-step prediction, one-step error, and the
level/trend/season state update — is inherently SEQUENTIAL (``state[t]``
depends on ``state[t-1]``), so numpy vectorisation cannot help; the
maximum-likelihood optimiser calls it once per objective evaluation, many
times per fit.

The kernel is compiled Cython (:mod:`._ets_recursion`), built with
``-ffp-contract=off`` so it reproduces the pure-numpy reference recursion
(:func:`._ets_models._ets_recursion_reference`) BIT-FOR-BIT in fp64. The
reference remains the blessed definition; this is an implementation swap that
only changes speed. Bit-identity is enforced by
``tests/timeseries/test_ets_kernel_parity.py``. This module keeps the
historical import path (``ets_recursion_nb``) stable for :mod:`._ets_models`.

The model type — error (A/M), trend (N/A/Ad) and season (N/A/M) — is passed in
as integer/boolean flags rather than specialised per family, so one kernel
covers every ETS(error, trend, season) combination. Damping is carried
entirely by ``phi_val`` (1.0 when the trend is undamped).

References
----------
Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008).
    Forecasting with Exponential Smoothing: The State Space Approach. Springer.
"""

from __future__ import annotations

from ._ets_recursion import ets_recursion_nb  # noqa: F401

__all__ = ["ets_recursion_nb"]
