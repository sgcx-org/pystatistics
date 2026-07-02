"""Smooth-basis dispatch and GAM design assembly.

One job: turn a parametric design plus a list of ``s()`` smooth
specifications into the fitting problem's matrices —

    X_aug = [X_parametric | B_1 Z_1 | ... | B_m Z_m]

with each smooth's basis built by its mgcv-matching constructor
(:mod:`_basis_cr`, :mod:`_basis_tp`) and its sum-to-zero identifiability
constraint absorbed (:mod:`_constraints`) so ``X_aug`` is full rank by
construction — for every basis type. (4.5.x skipped the constraint; each
smooth's span contained the constant, making the design exactly singular
against the intercept. That was the root cause of the garbage-EDF defect.)

Penalties are returned in per-smooth BLOCK coordinates ``(k_j-1, k_j-1)``
together with each smooth's column range in ``X_aug``; the P-IRLS layer
stacks their square roots itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.gam._basis_cr import cr_basis
from pystatistics.gam._basis_tp import tp_basis
from pystatistics.gam._constraints import absorb_sum_to_zero
from pystatistics.gam._smooth import SmoothTerm


@dataclass(frozen=True)
class BuiltSmooth:
    """One smooth term's constructed, constrained design pieces."""

    term: SmoothTerm
    block: tuple[int, int]           # column range in X_aug
    S_block: NDArray[np.floating[Any]]   # (k-1, k-1) constrained penalty
    Z: NDArray[np.floating[Any]]     # (k, k-1) constraint reparameterisation
    s_scale: float                   # mgcv S.scale penalty normalisation


def build_design(
    X_parametric: NDArray[np.floating[Any]],
    smooth_data: dict[str, NDArray[np.floating[Any]]],
    smooths: list[SmoothTerm],
) -> tuple[NDArray[np.floating[Any]], list[BuiltSmooth]]:
    """Assemble the augmented design and per-smooth penalty blocks.

    Args:
        X_parametric: ``(n, p_lin)`` parametric design (never ``None``;
            the caller supplies the intercept-only column if needed).
        smooth_data: Variable name -> ``(n,)`` predictor array (validated
            by the caller for length/finiteness).
        smooths: Smooth term specifications.

    Returns:
        ``(X_aug, built)`` — the full-rank augmented design and one
        :class:`BuiltSmooth` per term.

    Raises:
        ValidationError: unknown basis type (the ``s()`` constructor
            guards this too — belt and braces), or basis-level validation
            failures (k vs unique-x, non-finite x).
    """
    n = X_parametric.shape[0]
    blocks_x: list[NDArray] = [np.asarray(X_parametric, dtype=np.float64)]
    built: list[BuiltSmooth] = []
    col = X_parametric.shape[1]

    for st in smooths:
        x = np.asarray(smooth_data[st.var_name], dtype=np.float64).ravel()
        if st.bs == "cr":
            B, S, s_scale = cr_basis(x, k=st.k)
        elif st.bs == "tp":
            B, S, s_scale = tp_basis(x, k=st.k)
        else:
            raise ValidationError(
                f"Unknown basis type {st.bs!r}; expected 'cr' or 'tp'"
            )
        # Sum-to-zero identifiability constraint — EVERY basis type: the
        # cr span contains the constant outright, and the tp basis carries
        # an explicit constant null-space column; either way the constraint
        # removes exactly the direction that would collide with the
        # intercept, keeping k-1 columns (linear trend survives).
        B_c, S_c, Z = absorb_sum_to_zero(B, S)
        kc = B_c.shape[1]
        blocks_x.append(B_c)
        built.append(BuiltSmooth(
            term=st, block=(col, col + kc), S_block=S_c, Z=Z,
            s_scale=s_scale,
        ))
        col += kc

    X_aug = np.hstack(blocks_x)
    return X_aug, built
