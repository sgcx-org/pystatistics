"""Tensor-product smooth specifications: ``te()`` and ``ti()``.

Mirror ``mgcv::te()`` / ``mgcv::ti()``: a multivariate smooth built from 1-D
marginal bases combined by the row-wise Kronecker product, with one penalty
(and one smoothing parameter) per margin.

* ``te(x, z, ...)`` — full tensor-product smooth of two or more covariates,
  each margin on its own scale (unlike isotropic ``s(x, z)``).
* ``ti(x, z, ...)`` — tensor-product INTERACTION with the marginal main
  effects removed, for functional-ANOVA models ``te(x) + te(z) + ti(x, z)``.
"""

from __future__ import annotations

from typing import Sequence

from pystatistics.core.exceptions import ValidationError
from pystatistics.gam._smooth import s

_VALID_BASIS_TYPES = frozenset({"cr", "tp", "cc", "ps"})
_MIN_K = 3
_MAX_K = 500


def _broadcast(value, n: int, name: str) -> tuple:
    """A scalar applies to every margin; a sequence must match margin count."""
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return tuple([value] * n)
    seq = tuple(value)
    if len(seq) != n:
        raise ValidationError(
            f"{name} has {len(seq)} entries but there are {n} margins; "
            f"pass one value (applied to all) or one per margin"
        )
    return seq


class TensorSmooth:
    """Specification for a ``te()`` / ``ti()`` tensor-product smooth.

    Attributes:
        var_names: Marginal predictor names (>= 2).
        ks: Marginal basis dimension per margin (mgcv marginal ``k``).
        bss: Marginal basis type per margin (``'cr'``/``'tp'``/``'cc'``/
            ``'ps'``).
        interaction: ``True`` for ``ti`` (main effects removed), ``False``
            for ``te``.
    """

    __slots__ = ("var_names", "ks", "bss", "interaction")

    def __init__(
        self,
        var_names: Sequence[str],
        ks: Sequence[int],
        bss: Sequence[str],
        interaction: bool,
    ) -> None:
        names = tuple(var_names)
        if len(names) < 2:
            raise ValidationError(
                "a tensor smooth needs >= 2 margins; use s() for one variable"
            )
        for nm in names:
            if not isinstance(nm, str) or not nm.strip():
                raise ValidationError(
                    f"margin names must be non-empty strings, got {nm!r}"
                )
        for k in ks:
            if not isinstance(k, int) or isinstance(k, bool):
                raise ValidationError(f"k must be an integer, got {k!r}")
            if k < _MIN_K or k > _MAX_K:
                raise ValidationError(
                    f"each margin k must be in [{_MIN_K}, {_MAX_K}], got {k}"
                )
        for bs in bss:
            if bs not in _VALID_BASIS_TYPES:
                raise ValidationError(
                    f"bs must be one of {sorted(_VALID_BASIS_TYPES)}, got {bs!r}"
                )
        self.var_names = tuple(nm.strip() for nm in names)
        self.ks = tuple(int(k) for k in ks)
        self.bss = tuple(bss)
        self.interaction = bool(interaction)

    @property
    def label(self) -> str:
        head = "ti" if self.interaction else "te"
        return f"{head}({','.join(self.var_names)})"

    def __repr__(self) -> str:
        return (
            f"{'ti' if self.interaction else 'te'}"
            f"({', '.join(self.var_names)}, k={self.ks}, bs={self.bss})"
        )


def _marginal_or_tensor(var_names, k, bs, interaction):
    """One margin -> a centred 1-D smooth (mgcv ``te(x)``/``ti(x)`` reduce to
    ``s(x)``); two or more -> a tensor-product smooth."""
    n = len(var_names)
    if n == 0:
        raise ValidationError("te()/ti() need at least one variable")
    ks = _broadcast(k, n, "k")
    bss = _broadcast(bs, n, "bs")
    if n == 1:
        return s(var_names[0], k=ks[0], bs=bss[0])
    return TensorSmooth(
        var_names=var_names, ks=ks, bss=bss, interaction=interaction,
    )


def te(*var_names: str, k=5, bs: str = "cr"):
    """Tensor-product smooth ``te(x, z, ...)`` (mgcv default marginal ``k=5``).

    Args:
        *var_names: One or more marginal predictor names (a single name
            yields a centred 1-D smooth, exactly as mgcv's ``te(x)`` does).
        k: Marginal basis dimension — a scalar (all margins) or one per
            margin. Default 5, matching mgcv's ``te`` marginal default.
        bs: Marginal basis type — a scalar or one per margin.

    Returns:
        A :class:`TensorSmooth` (>= 2 margins) or :class:`SmoothTerm`
        (one margin).
    """
    return _marginal_or_tensor(var_names, k, bs, interaction=False)


def ti(*var_names: str, k=5, bs: str = "cr"):
    """Tensor-product interaction ``ti(x, z, ...)`` (main effects removed).

    Same arguments as :func:`te`; used for functional-ANOVA decompositions
    ``te(x) + te(z) + ti(x, z)`` (the ``te(x)``/``te(z)`` main-effect terms
    are often written ``ti(x)``/``ti(z)``, i.e. single-margin smooths).
    """
    return _marginal_or_tensor(var_names, k, bs, interaction=True)
