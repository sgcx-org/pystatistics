"""Factor ``by`` variables: detection, per-level smooths, group main effect.

One job: turn a ``s(x, by=g, by_type='factor')`` term — where ``g`` is an
integer-coded grouping variable — into the design pieces mgcv builds for
``s(x, by=factor(g))``:

  * one **centered** smooth of ``x`` per level, each multiplied by that level's
    0/1 indicator and carrying its own penalty / smoothing parameter, and
  * the grouping variable's **per-level means** (a treatment-contrast main
    effect) injected into the parametric design so the model is identifiable
    and complete — the user does NOT have to add ``+ g`` by hand.

This reproduces ``mgcv::gam(y ~ s(x, by=factor(g)))`` to floating-point
arithmetic (validated against a frozen mgcv fixture: total EDF and fitted
values agree to ~5e-9). It is exact only when the parametric design carries an
intercept (mgcv always has one; ``gam``'s default does too) — enforced here.

The reported per-level smoothing parameters are on this library's internal
(centered) penalty scale; they differ from mgcv's printed ``sp`` by a constant
factor per smooth (mgcv rescales each penalty by its own ``S.scale``). The
*fit* — coefficients, fitted values, EDF — is unaffected by that rescaling, so
this is a reporting-scale difference, not a numerical one (the same situation
as the ``tp``/``ti`` bases).

The module also owns the **fail-loud guard** for an un-annotated ``by``: a
factor-looking numeric column with ``by_type=None`` is rejected rather than
silently multiplied into a meaningless varying coefficient (which is what a
naive continuous-``by`` interpretation of ``s(x, by=group)`` would produce).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.gam._smooth import IsotropicSmooth, SmoothTerm
from pystatistics.gam._tensor_smooth import TensorSmooth

# Internal smooth-data keys for the per-level 0/1 indicators. The NUL byte
# cannot occur in a user-supplied variable name, so these never collide.
_IND_PREFIX = "\x00factor_by::"


@dataclass(frozen=True)
class FactorLevelSmooth:
    """One level of a factor-``by`` smooth: a centered smooth of ``var_name``
    restricted to the rows where ``by_var`` equals ``level`` (via ``indicator``).

    ``build_design`` centers the ``var_name`` basis (sum-to-zero, like an
    ordinary smooth) and multiplies it by the ``indicator`` column, so the term
    contributes only on that level's observations. The per-level means live in
    the parametric design (see :func:`expand_factor_by`), not here.
    """

    var_name: str
    k: int
    bs: str
    indicator: str   # key into smooth_data for this level's 0/1 column
    by_var: str      # original grouping-variable name (for the display label)
    level: int       # this level's integer code


@dataclass(frozen=True)
class FactorByExpansion:
    """Result of expanding factor-``by`` terms out of a smooth list."""

    smooths: list[Any]                     # SmoothTerm | *Smooth | FactorLevelSmooth
    smooth_data: dict[str, NDArray[np.floating[Any]]]
    X_param: NDArray[np.floating[Any]]
    param_names: list[str] | None
    smooth_labels: list[str]               # aligned 1:1 with ``smooths``


def looks_like_factor(z: NDArray[np.floating[Any]]) -> bool:
    """True if ``z`` looks like a categorical factor coding, not a continuous
    covariate.

    Tight signature (chosen to minimise false positives against genuine
    continuous data): integer-valued, at least 3 distinct values, and those
    values form a contiguous run starting at 0 or 1 — the signature of
    ``pandas`` category codes (0-based) and R's ``as.integer(factor(...))``
    (1-based), the two ways a ported ``s(x, by=group)`` idiom arrives. A binary
    0/1 column is deliberately NOT flagged: it is a valid, common continuous
    ``by`` (a single-subgroup smooth).
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    if z.size == 0 or not np.all(np.isfinite(z)):
        return False
    if not np.all(np.abs(z - np.round(z)) < 1e-9):
        return False
    distinct = np.unique(np.round(z).astype(np.int64))
    k = distinct.size
    if k < 3:
        return False
    return bool(
        np.array_equal(distinct, np.arange(k))          # 0-based run
        or np.array_equal(distinct, np.arange(1, k + 1))  # 1-based run
    )


def factor_levels(z: NDArray[np.floating[Any]], by_var: str) -> NDArray[np.int64]:
    """Sorted distinct integer levels of an explicit factor ``by`` column.

    Requires integer-valued codes (a factor's levels are categorical). At least
    two levels are needed for a per-level model.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    if not np.all(np.isfinite(z)):
        raise ValidationError(
            f"factor by-variable {by_var!r} contains non-finite values"
        )
    if not np.all(np.abs(z - np.round(z)) < 1e-9):
        raise ValidationError(
            f"by_type='factor' needs integer-coded levels, but by-variable "
            f"{by_var!r} has non-integer values. Integer-code the groups "
            "(e.g. 0,1,2,...), or use by_type='continuous' for a numeric "
            "varying coefficient."
        )
    levels = np.unique(np.round(z).astype(np.int64))
    if levels.size < 2:
        raise ValidationError(
            f"by_type='factor' needs at least 2 levels, but by-variable "
            f"{by_var!r} has only {levels.size}"
        )
    return levels


def _has_intercept(X: NDArray[np.floating[Any]]) -> bool:
    """True if the constant vector lies in the column space of ``X`` (the model
    already carries an intercept, explicitly or implicitly)."""
    n = X.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    return int(np.linalg.matrix_rank(np.hstack([X, ones]))) == int(
        np.linalg.matrix_rank(X)
    )


def _guard_unannotated_by(st: SmoothTerm, z: NDArray[np.floating[Any]]) -> None:
    """Fail loud on a factor-looking ``by`` with ``by_type`` unset."""
    if not looks_like_factor(z):
        return
    distinct = np.unique(np.round(np.asarray(z, np.float64)).astype(np.int64))
    raise ValidationError(
        f"by-variable {st.by!r} for s({st.var_name!r}) looks categorical: it "
        f"takes {distinct.size} integer values {distinct.tolist()} coded as a "
        "contiguous run — the signature of a factor. pystatistics will not "
        "guess how to interpret it. Choose explicitly:\n"
        "  * by_type='factor'     -> one smooth per level, mgcv's "
        "s(x, by=factor(g)) (per-level group means are added for you)\n"
        "  * by_type='continuous' -> a single varying coefficient by*f(x), "
        "the numeric column multiplied into the basis"
    )


def expand_factor_by(
    smooths: list[Any],
    smooth_data: dict[str, NDArray[np.floating[Any]]],
    X_param: NDArray[np.floating[Any]],
    param_names: list[str] | None,
    default_label,
) -> FactorByExpansion:
    """Expand factor-``by`` smooths; pass other terms through unchanged.

    Args:
        smooths: User smooth specs (``SmoothTerm`` / tensor / isotropic).
        smooth_data: Validated ``name -> (n,)`` predictor arrays.
        X_param: ``(n, p)`` parametric design (carries the intercept).
        param_names: Names of ``X_param`` columns, or ``None``.
        default_label: ``callable(spec) -> str`` for a non-factor term's label.

    Returns:
        :class:`FactorByExpansion` with factor-``by`` terms replaced by their
        per-level :class:`FactorLevelSmooth` specs, the per-level indicator
        columns added to ``smooth_data``, the group treatment-contrast columns
        appended to ``X_param`` (with names when ``param_names`` is given), and
        a label per emitted smooth.

    Raises:
        ValidationError: an un-annotated factor-looking ``by`` (guard); a
            factor ``by`` without an intercept in the model; or a group main
            effect already collinear with ``X_param`` (the user encoded the
            grouping variable in ``X`` too).
    """
    out_smooths: list[Any] = []
    out_labels: list[str] = []
    new_data = dict(smooth_data)
    contrast_cols: list[NDArray[np.floating[Any]]] = []
    contrast_names: list[str] = []
    contrasted: set[str] = set()   # grouping vars whose main effect is injected
    saw_factor = False

    for st in smooths:
        is_plain = isinstance(st, SmoothTerm) and not isinstance(
            st, (TensorSmooth, IsotropicSmooth)
        )
        if not (is_plain and st.by is not None):
            out_smooths.append(st)
            out_labels.append(default_label(st))
            continue

        z = np.asarray(smooth_data[st.by], dtype=np.float64).ravel()

        if st.by_type is None:
            _guard_unannotated_by(st, z)          # raises if factor-looking
            out_smooths.append(st)                # else: continuous by
            out_labels.append(default_label(st))
            continue
        if st.by_type == "continuous":
            out_smooths.append(st)
            out_labels.append(default_label(st))
            continue

        # by_type == 'factor'
        saw_factor = True
        levels = factor_levels(z, st.by)
        codes = np.round(z).astype(np.int64)
        for lvl in levels:
            ind_key = f"{_IND_PREFIX}{st.by}={lvl}"
            # A level indicator depends only on the grouping column, so several
            # smooths sharing one by-variable (s(x,by=g)+s(w,by=g)) reuse the
            # same indicator rather than colliding.
            if ind_key not in new_data:
                new_data[ind_key] = (codes == lvl).astype(np.float64)
            out_smooths.append(FactorLevelSmooth(
                var_name=st.var_name, k=st.k, bs=st.bs,
                indicator=ind_key, by_var=st.by, level=int(lvl),
            ))
            out_labels.append(f"s({st.var_name}):{st.by}={lvl}")

        # Inject the grouping variable's main effect ONCE, even if several
        # smooths share it: the first level is the reference (mean carried by
        # the intercept); one treatment-contrast column per other level.
        if st.by not in contrasted:
            contrasted.add(st.by)
            for lvl in levels[1:]:
                contrast_cols.append(new_data[f"{_IND_PREFIX}{st.by}={lvl}"])
                contrast_names.append(f"{st.by}={lvl}")

    if not saw_factor:
        return FactorByExpansion(
            smooths=out_smooths, smooth_data=new_data, X_param=X_param,
            param_names=param_names, smooth_labels=out_labels,
        )

    if not _has_intercept(X_param):
        raise ValidationError(
            "by_type='factor' adds per-level group means as a treatment "
            "contrast against the model intercept, but the parametric design "
            "has no intercept column. Leave X=None (gam adds one) or include "
            "an intercept column in X."
        )

    X_new = np.hstack([X_param, np.column_stack(contrast_cols)])
    if int(np.linalg.matrix_rank(X_new)) < X_new.shape[1]:
        raise ValidationError(
            "by_type='factor' adds the grouping variable's main effect "
            "automatically, but that collides with a column already in X "
            "(the grouping variable appears to be encoded in X as well). "
            "Remove it from X, or use by_type='continuous'."
        )

    names_new = (
        None if param_names is None else list(param_names) + contrast_names
    )
    return FactorByExpansion(
        smooths=out_smooths, smooth_data=new_data, X_param=X_new,
        param_names=names_new, smooth_labels=out_labels,
    )
