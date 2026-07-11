"""
Smooth term specification for GAM formulas.

Provides the user-facing ``s()`` constructor that mirrors R's
``mgcv::s()`` syntax for declaring smooth terms.
"""

from __future__ import annotations

from pystatistics.core.exceptions import ValidationError

_VALID_BASIS_TYPES = frozenset({"cr", "tp", "cc", "ps"})

_MIN_K = 3
_MAX_K = 500


class SmoothTerm:
    """Specification for a smooth term ``s(x)`` in a GAM formula.

    A pure, immutable-by-convention specification: variable name, basis
    type and basis dimension. All fit-derived quantities (basis matrices,
    penalties, constraint reparameterisations) live inside the fit — a
    ``SmoothTerm`` can safely be reused across multiple ``gam()`` calls
    and datasets without stale-state hazards.

    Attributes:
        var_name: Name of the predictor variable.
        k: Basis dimension (>= 3), exactly as mgcv's ``s(x, k=...)``. The
            fitted smooth contributes ``k - 1`` coefficients after its
            sum-to-zero identifiability constraint (same as mgcv).
        bs: Basis type: ``'cr'`` (cubic regression spline, default),
            ``'tp'`` (thin plate regression spline), ``'cc'`` (cyclic cubic
            regression spline, for periodic/seasonal covariates), or ``'ps'``
            (P-spline).
    """

    __slots__ = ("var_name", "k", "bs", "by")

    def __init__(
        self,
        var_name: str,
        k: int = 10,
        bs: str = "cr",
        by: str | None = None,
    ) -> None:
        """Create a smooth term specification.

        Args:
            var_name: Name of the predictor variable.
            k: Basis dimension (default 10, matching mgcv). Must be >= 3.
            bs: Basis type -- ``'cr'`` for cubic regression spline
                (default) or ``'tp'`` for thin plate regression spline.

        Raises:
            ValidationError: If ``var_name`` is empty, ``k`` is out of
                range, or ``bs`` is not a recognised basis type.
        """
        if not isinstance(var_name, str) or not var_name.strip():
            raise ValidationError(
                "var_name must be a non-empty string, "
                f"got {var_name!r}"
            )

        if not isinstance(k, int) or isinstance(k, bool):
            raise ValidationError(
                f"k must be an integer, got {type(k).__name__}"
            )
        if k < _MIN_K:
            raise ValidationError(
                f"k must be >= {_MIN_K} for a meaningful smooth, got k={k}"
            )
        if k > _MAX_K:
            raise ValidationError(
                f"k must be <= {_MAX_K}, got k={k}"
            )

        if bs not in _VALID_BASIS_TYPES:
            raise ValidationError(
                f"bs must be one of {sorted(_VALID_BASIS_TYPES)}, got {bs!r}"
            )

        if by is not None and (not isinstance(by, str) or not by.strip()):
            raise ValidationError(
                f"by must be None or a non-empty variable name, got {by!r}"
            )

        self.var_name = var_name.strip()
        self.k = k
        self.bs = bs
        self.by = by.strip() if isinstance(by, str) else None

    def __repr__(self) -> str:
        by = "" if self.by is None else f", by={self.by!r}"
        return f"s({self.var_name!r}, k={self.k}, bs={self.bs!r}{by})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SmoothTerm):
            return NotImplemented
        return (
            self.var_name == other.var_name
            and self.k == other.k
            and self.bs == other.bs
            and self.by == other.by
        )

    def __hash__(self) -> int:
        return hash((self.var_name, self.k, self.bs, self.by))


def s(var_name: str, k: int = 10, bs: str = "cr",
      by: str | None = None) -> SmoothTerm:
    """Convenience constructor for smooth terms, matching ``mgcv::s()``.

    Usage::

        from pystatistics.gam import s, gam
        result = gam(y, X, smooths=[s('x1', k=15), s('x2')])

    Args:
        var_name: Name of the predictor variable.
        k: Basis dimension (default 10).
        bs: Basis type: ``'cr'``, ``'tp'``, ``'cc'``, or ``'ps'`` (default
            ``'cr'``).
        by: Optional name of a *continuous* ``by`` variable — fits the
            varying-coefficient term ``by * f(x)`` (mgcv's ``s(x, by=z)``). The
            smooth keeps its full basis (no centering constraint) since the
            multiplication by ``by`` removes the constant confound.

    Returns:
        A :class:`SmoothTerm` specification object.
    """
    return SmoothTerm(var_name=var_name, k=k, bs=bs, by=by)
