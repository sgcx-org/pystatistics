"""Tensor-product smooth bases — ``te()`` and ``ti()``, mgcv-equivalent.

A tensor-product smooth builds a multivariate smooth from 1-D marginal bases
(any of ``cr``/``tp``/``cc``/``ps``) by the ROW-WISE Kronecker product of the
marginal model matrices, with ONE penalty per margin — each embedded in the
full tensor coefficient space as ``I (x) ... (x) S_t (x) ... (x) I`` and
carrying its own smoothing parameter. This mirrors mgcv's
``tensor.prod.model.matrix`` / ``tensor.prod.penalties``:

* ``te(x, z, ...)`` — full tensor product. The marginal bases are used RAW
  (uncentred); a single sum-to-zero identifiability constraint is absorbed
  into the whole tensor (``smoothCon(te, absorb.cons=TRUE)``).
* ``ti(x, z, ...)`` — tensor-product INTERACTION. Each marginal basis is
  sum-to-zero-centred FIRST (removing its main effect and constant), then
  tensored; no further overall constraint. Used in functional-ANOVA
  decompositions ``te(x) + te(z) + ti(x, z)``.

Each margin's penalty is already divided by its mgcv ``S.scale`` inside the
marginal constructor, so the per-margin smoothing parameters are comparable
exactly as in mgcv. The several penalties share one coefficient block and
OVERLAP; the REML penalty determinant over them is taken jointly (see
``_penalty_group``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.gam._basis_cc import cc_basis
from pystatistics.gam._basis_cr import cr_basis
from pystatistics.gam._basis_ps import ps_basis
from pystatistics.gam._basis_tp import tp_basis
from pystatistics.gam._constraints import (
    absorb_sum_to_zero,
    absorb_sum_to_zero_multi,
)


def marginal_basis(
    x: NDArray[np.floating[Any]], k: int, bs: str,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], float]:
    """Dispatch to the RAW (unconstrained) 1-D basis constructor for a margin."""
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


def _row_kron(
    a: NDArray[np.floating[Any]], b: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Row-wise Kronecker product: row ``i`` is ``kron(a[i], b[i])``.

    ``(n, p), (n, q) -> (n, p*q)`` with the FIRST factor varying slowest
    (mgcv ``tensor.prod.model.matrix`` column order).
    """
    n, p = a.shape
    q = b.shape[1]
    return (a[:, :, None] * b[:, None, :]).reshape(n, p * q)


def _embed(
    S: NDArray[np.floating[Any]], left: int, right: int,
) -> NDArray[np.floating[Any]]:
    """``I_left (x) S (x) I_right`` — one margin's penalty in tensor space."""
    out = np.kron(S, np.eye(right)) if right > 1 else S
    if left > 1:
        out = np.kron(np.eye(left), out)
    return out


def _scale_penalty(
    X: NDArray[np.floating[Any]], S: NDArray[np.floating[Any]],
) -> float:
    """mgcv ``scale.penalty`` factor for one penalty against a design.

    ``S.scale = max_j (sum_i |S_ij|) / (max_i sum_j |X_ij|)^2`` — the same
    rule the 1-D constructors apply, but for a tensor smooth mgcv applies it
    to the ASSEMBLED tensor design and each Kronecker-embedded (unscaled)
    penalty, not to the marginals (verified to ~4e-15 vs ``smoothCon``).
    """
    ma_xx = float(np.max(np.abs(X).sum(axis=1)) ** 2)
    return float(np.max(np.abs(S).sum(axis=0)) / ma_xx)


@dataclass(frozen=True)
class TensorBasis:
    """A constructed tensor-product smooth (constrained)."""

    X: NDArray[np.floating[Any]]              # (n, K) constrained basis
    S_blocks: list[NDArray[np.floating[Any]]]  # per-margin penalties (K, K)
    Z: NDArray[np.floating[Any]]              # (K0, K) constraint reparam
    s_scales: list[float]                     # per-margin mgcv S.scale


def te_basis(
    margin_x: list[NDArray[np.floating[Any]]],
    ks: list[int],
    bss: list[str],
    interaction: bool,
) -> TensorBasis:
    """Build a ``te`` (``interaction=False``) or ``ti`` tensor-product smooth.

    Args:
        margin_x: One ``(n,)`` covariate array per margin.
        ks: Marginal basis dimension per margin (mgcv marginal ``k``).
        bss: Marginal basis type per margin.
        interaction: ``True`` for ``ti`` (centre each margin first, no overall
            constraint); ``False`` for ``te`` (raw margins, one overall
            sum-to-zero constraint on the tensor).

    Returns:
        A :class:`TensorBasis` with the constrained basis, one constrained
        penalty per margin, the constraint reparameterisation, and the
        per-margin ``S.scale`` factors.
    """
    X, S_blocks, s_scales = assemble_tensor(margin_x, ks, bss, interaction)
    if interaction:
        Z = np.eye(X.shape[1], dtype=np.float64)
    else:
        # te: one sum-to-zero constraint on the whole tensor basis.
        X, S_blocks, Z = absorb_sum_to_zero_multi(X, S_blocks)
    return TensorBasis(X=X, S_blocks=S_blocks, Z=Z, s_scales=s_scales)


def assemble_tensor(
    margin_x: list[NDArray[np.floating[Any]]],
    ks: list[int],
    bss: list[str],
    interaction: bool,
) -> tuple[NDArray[np.floating[Any]], list[NDArray[np.floating[Any]]],
           list[float]]:
    """Tensor basis + penalties BEFORE the overall ``te`` constraint.

    For ``ti`` this is the final smooth (margins already centred, no overall
    constraint); for ``te`` it is the ``absorb.cons=FALSE`` tensor that the
    caller then constrains. Matches ``smoothCon(..., absorb.cons=FALSE)``.
    """
    bases: list[NDArray[np.floating[Any]]] = []
    penalties: list[NDArray[np.floating[Any]]] = []
    for x, k, bs in zip(margin_x, ks, bss):
        B, S, sc = marginal_basis(x, k, bs)
        S = S * sc  # undo the marginal S.scale; mgcv rescales at tensor level
        if interaction:
            # ti: centre each margin FIRST so the tensor excludes main effects.
            B, S, _Z = absorb_sum_to_zero(B, S)
        bases.append(B)
        penalties.append(S)

    dims = [B.shape[1] for B in bases]

    # Row-wise Kronecker product of the marginal model matrices (fold left).
    X = bases[0]
    for B in bases[1:]:
        X = _row_kron(X, B)

    # Per-margin penalty embedded in the tensor coefficient space:
    #   S^(t) = I_{prod dims[:t]} (x) S_t (x) I_{prod dims[t+1:]}
    # then rescaled by mgcv's tensor-level scale.penalty rule.
    m = len(bases)
    S_blocks: list[NDArray[np.floating[Any]]] = []
    s_scales: list[float] = []
    for t in range(m):
        left = int(np.prod(dims[:t], dtype=np.int64)) if t > 0 else 1
        right = int(np.prod(dims[t + 1:], dtype=np.int64)) if t < m - 1 else 1
        S_embed = _embed(penalties[t], left, right)
        sc = _scale_penalty(X, S_embed)
        s_scales.append(sc)
        S_blocks.append(S_embed / sc)

    return X, S_blocks, s_scales
