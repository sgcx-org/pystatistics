"""
Smooth term specification for GAM formulas.

Provides the user-facing ``s()`` constructor that mirrors R's
``mgcv::s()`` syntax for declaring smooth terms.
"""

from __future__ import annotations

from pystatistics.core.exceptions import ValidationError

_VALID_BASIS_TYPES = frozenset({"cr", "tp"})

_MIN_K = 3
_MAX_K = 500


class SmoothTerm:
    """Specification for a smooth term ``s(x)`` in a GAM formula.

    This is a mutable specification object that gets consumed by the
    GAM fitter.  It stores the variable name, basis type, and number
    of basis functions, then receives its computed basis matrix and
    penalty matrix during model setup.

    # RULE 5 EXCEPTION: SmoothTerm is a mutable specification object
    # that accumulates basis information during model setup.  It does
    # not carry hidden state that affects computation -- it is purely
    # a container for user-specified configuration that gets populated
    # with derived quantities (basis matrix, penalty) during fitting.

    Attributes:
        var_name: Name of the predictor variable.
        k: Number of basis functions (>= 3).
        bs: Basis type: ``'cr'`` (cubic regression spline, default)
            or ``'tp'`` (thin plate regression spline).
        basis_matrix: Populated during fitting -- the (n, k) design
            matrix for this smooth term.  ``None`` before fitting.
        penalty_matrix: Populated during fitting -- the (k, k) penalty
            matrix for this smooth term.  ``None`` before fitting.
    """

    __slots__ = ("var_name", "k", "bs", "basis_matrix", "penalty_matrix")

    def __init__(
        self,
        var_name: str,
        k: int = 10,
        bs: str = "cr",
    ) -> None:
        """Create a smooth term specification.

        Args:
            var_name: Name of the predictor variable.
            k: Number of basis functions (default 10, matching mgcv).
                Must be >= 3.
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

        self.var_name = var_name.strip()
        self.k = k
        self.bs = bs
        self.basis_matrix = None
        self.penalty_matrix = None

    def __repr__(self) -> str:
        return f"s({self.var_name!r}, k={self.k}, bs={self.bs!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SmoothTerm):
            return NotImplemented
        return (
            self.var_name == other.var_name
            and self.k == other.k
            and self.bs == other.bs
        )

    def __hash__(self) -> int:
        return hash((self.var_name, self.k, self.bs))


def s(var_name: str, k: int = 10, bs: str = "cr") -> SmoothTerm:
    """Convenience constructor for smooth terms, matching ``mgcv::s()``.

    Usage::

        from pystatistics.gam import s, gam
        result = gam(y, X, smooths=[s('x1', k=15), s('x2')])

    Args:
        var_name: Name of the predictor variable.
        k: Number of basis functions (default 10).
        bs: Basis type: ``'cr'`` or ``'tp'`` (default ``'cr'``).

    Returns:
        A :class:`SmoothTerm` specification object.
    """
    return SmoothTerm(var_name=var_name, k=k, bs=bs)
