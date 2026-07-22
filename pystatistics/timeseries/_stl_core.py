"""
Fused STL inner/outer loop, matching R ``stats::stl`` — public import surface.

The whole seasonal-trend iteration — detrend, cycle-subseries loess with
one-period end extension, low-pass moving-average cascade + loess,
deseasonalise, trend loess, and the bisquare robustness outer loop — is a
single compiled driver (:func:`stl_core_nb`), mirroring R's single Fortran
routine. Running the entire loop in one compiled pass (rather than
orchestrating per-step numpy calls from Python) is what matches R's speed on
robust decompositions, where the outer loop repeats the inner passes up to
fifteen times.

The kernel is compiled Cython (:mod:`._stl_kernels`, one translation unit with
loess and the robustness step); this module keeps the historical import path
(``stl_core_nb``) for :mod:`._stl`. Clean-room from the algorithm (Cleveland,
Cleveland, McRae & Terpenning, 1990); all arithmetic in the same order as the
reference under ``-ffp-contract=off`` (acceptance gate: ``test_stl_r_parity.py``).
"""

from __future__ import annotations

from ._stl_kernels import (  # noqa: F401
    moving_average_nb,
    stl_core_nb,
)

__all__ = ["moving_average_nb", "stl_core_nb"]
