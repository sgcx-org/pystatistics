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
from pystatistics.gam._basis_cc import cc_basis
from pystatistics.gam._basis_ps import ps_basis
from pystatistics.gam._basis_md import md_tp_basis
from pystatistics.gam._basis_te import te_basis
from pystatistics.gam._constraints import absorb_sum_to_zero
from pystatistics.gam._factor_by import FactorLevelSmooth
from pystatistics.gam._smooth import IsotropicSmooth, SmoothTerm
from pystatistics.gam._tensor_smooth import TensorSmooth


def _pick_basis(
    bs: str, x: NDArray[np.floating[Any]], k: int,
) -> tuple[NDArray, NDArray, float]:
    """Dispatch to a univariate basis constructor by ``bs`` name."""
    if bs == "cr":
        return cr_basis(x, k=k)
    if bs == "tp":
        return tp_basis(x, k=k)
    if bs == "cc":
        return cc_basis(x, k=k)
    if bs == "ps":
        return ps_basis(x, k=k)
    raise ValidationError(
        f"Unknown basis type {bs!r}; expected 'cr', 'tp', 'cc', or 'ps'"
    )


@dataclass(frozen=True)
class BuiltSmooth:
    """One smooth term's constructed, constrained design pieces.

    ``S_blocks`` holds one constrained penalty per smoothing parameter: a
    single entry for an ordinary ``s()`` smooth, one per margin for a
    tensor-product ``te()``/``ti()`` smooth (all over the same ``block``).
    ``s_scales`` is aligned with ``S_blocks``.
    """

    term: Any                            # SmoothTerm | TensorSmooth
    block: tuple[int, int]               # column range in X_aug
    S_blocks: list[NDArray[np.floating[Any]]]  # constrained penalties
    Z: NDArray[np.floating[Any]]         # constraint reparameterisation
    s_scales: list[float]                # mgcv S.scale, aligned with S_blocks


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
        if isinstance(st, TensorSmooth):
            margin_x = []
            for nm in st.var_names:
                if nm not in smooth_data:
                    raise ValidationError(
                        f"smooth_data missing margin {nm!r} for {st.label}"
                    )
                margin_x.append(
                    np.asarray(smooth_data[nm], dtype=np.float64).ravel()
                )
            tb = te_basis(
                margin_x, ks=list(st.ks), bss=list(st.bss),
                interaction=st.interaction,
            )
            kc = tb.X.shape[1]
            blocks_x.append(tb.X)
            built.append(BuiltSmooth(
                term=st, block=(col, col + kc), S_blocks=tb.S_blocks,
                Z=tb.Z, s_scales=tb.s_scales,
            ))
            col += kc
            continue

        if isinstance(st, IsotropicSmooth):
            cols = []
            for nm in st.var_names:
                if nm not in smooth_data:
                    raise ValidationError(
                        f"smooth_data missing variable {nm!r} for {st.label}"
                    )
                cols.append(
                    np.asarray(smooth_data[nm], dtype=np.float64).ravel()
                )
            coords = np.column_stack(cols)
            B, S, s_scale = md_tp_basis(coords, k=st.k)
            B_c, S_c, Z = absorb_sum_to_zero(B, S)
            kc = B_c.shape[1]
            blocks_x.append(B_c)
            built.append(BuiltSmooth(
                term=st, block=(col, col + kc), S_blocks=[S_c], Z=Z,
                s_scales=[s_scale],
            ))
            col += kc
            continue

        if isinstance(st, FactorLevelSmooth):
            # One level of a factor ``by``: center the smooth of ``var_name``
            # (sum-to-zero, exactly as an ordinary smooth) and multiply the
            # centered basis by this level's 0/1 indicator, so the term acts
            # only on that level's rows. The per-level group means live in the
            # parametric design (see _factor_by.expand_factor_by), keeping this
            # centered smooth free of the constant confound. mgcv-exact.
            x = np.asarray(smooth_data[st.var_name], dtype=np.float64).ravel()
            B, S, s_scale = _pick_basis(st.bs, x, st.k)
            B_c, S_c, Z = absorb_sum_to_zero(B, S)
            ind = np.asarray(smooth_data[st.indicator], dtype=np.float64).ravel()
            B_c = B_c * ind[:, np.newaxis]
            kc = B_c.shape[1]
            blocks_x.append(B_c)
            built.append(BuiltSmooth(
                term=st, block=(col, col + kc), S_blocks=[S_c], Z=Z,
                s_scales=[s_scale],
            ))
            col += kc
            continue

        x = np.asarray(smooth_data[st.var_name], dtype=np.float64).ravel()
        B, S, s_scale = _pick_basis(st.bs, x, st.k)

        if st.by is not None:
            # Continuous varying-coefficient smooth ``by * f(x)`` (mgcv
            # ``s(x, by=z)``): keep the FULL basis — the by-multiplication
            # removes the constant confound, so no sum-to-zero centering — and
            # scale each row by the by-variable. The penalty is unchanged.
            if st.by not in smooth_data:
                raise ValidationError(
                    f"smooth_data missing by-variable {st.by!r} for s({st.var_name!r})"
                )
            z = np.asarray(smooth_data[st.by], dtype=np.float64).ravel()
            if z.shape[0] != n:
                raise ValidationError(
                    f"by-variable {st.by!r} has {z.shape[0]} obs, expected {n}"
                )
            B_c = B * z[:, np.newaxis]
            S_c = S
            Z = np.eye(B.shape[1], dtype=np.float64)
        else:
            # Sum-to-zero identifiability constraint — EVERY basis type: the
            # cr span contains the constant outright, and the tp basis carries
            # an explicit constant null-space column; either way the constraint
            # removes exactly the direction that would collide with the
            # intercept, keeping k-1 columns (linear trend survives).
            B_c, S_c, Z = absorb_sum_to_zero(B, S)
        kc = B_c.shape[1]
        blocks_x.append(B_c)
        built.append(BuiltSmooth(
            term=st, block=(col, col + kc), S_blocks=[S_c], Z=Z,
            s_scales=[s_scale],
        ))
        col += kc

    X_aug = np.hstack(blocks_x)
    return X_aug, built
