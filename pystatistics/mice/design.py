"""
MICEDesign: validated data wrapper for chained-equations imputation.

Wraps an ``n x p`` data matrix with NaN marking missing entries, and carries the
two pieces of per-column metadata the sweep needs:

  * ``col_kinds``   — the statistical kind of each column. Stage 1 supports only
                      ``'numeric'``; the metadata vector exists from day one so
                      Stage 3 (categorical) is a matter of relaxing one guard and
                      adding compatible methods, not restructuring the design.
  * ``methods``     — the imputation method assigned to each *incomplete* column
                      (e.g. ``'pmm'``), validated against the method registry and
                      checked for kind-compatibility at construction (Rule 2:
                      validate input at the boundary).

Immutable after construction. Follows the pystatistics Design pattern (cf.
``MVNDesign``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.mice.methods import get_method, is_registered

# Stage 1 supports numeric columns only. This single constant + the guard in
# ``_resolve_kinds`` is the entire "numeric-only" restriction; Stage 3 widens it.
_SUPPORTED_KINDS = ("numeric",)
_DEFAULT_METHOD = "pmm"


@dataclass(frozen=True)
class MICEDesign:
    """Design for MICE: data matrix + per-column kind and method metadata."""

    _data: NDArray[np.floating[Any]]
    _missing_mask: NDArray[np.bool_]
    _col_names: tuple[str, ...]
    _col_kinds: tuple[str, ...]
    _methods: tuple[str, ...]  # per column; '' for fully observed columns
    _n: int
    _p: int

    # ------------------------------------------------------------------ build
    @classmethod
    def from_array(
        cls,
        data,
        *,
        method: str = _DEFAULT_METHOD,
        methods: Mapping[Any, str] | Sequence[str] | None = None,
        column_names: Sequence[str] | None = None,
        column_kinds: Sequence[str] | None = None,
    ) -> "MICEDesign":
        """Build a MICEDesign from array-like data with NaN for missing values.

        Parameters
        ----------
        data : array-like
            2D matrix (observations x variables); NaN marks missing entries.
            Accepts numpy arrays and anything with a ``.values`` attribute
            (e.g. pandas DataFrame).
        method : str
            Default imputation method for every incomplete column (R default
            ``'pmm'``). Overridden per-column by ``methods``.
        methods : mapping or sequence, optional
            Per-column method override. Either a mapping ``{name_or_index:
            method}`` or a length-``p`` sequence. Entries for fully observed
            columns are ignored.
        column_names : sequence of str, optional
            Column names. Defaults to ``x0..x{p-1}`` (or DataFrame columns).
        column_kinds : sequence of str, optional
            Per-column statistical kind. Defaults to ``'numeric'`` for all.
            Stage 1 rejects any non-numeric kind.
        """
        data_array, inferred_names = _coerce_data(data)
        return cls._build(
            data_array,
            method=method,
            methods=methods,
            column_names=column_names if column_names is not None else inferred_names,
            column_kinds=column_kinds,
        )

    @classmethod
    def from_datasource(
        cls,
        source,
        *,
        columns: Sequence[str] | None = None,
        method: str = _DEFAULT_METHOD,
        methods: Mapping[Any, str] | Sequence[str] | None = None,
    ) -> "MICEDesign":
        """Build a MICEDesign from a DataSource (selected columns, in order)."""
        keys = list(columns) if columns is not None else list(source.keys())
        if not keys:
            raise ValidationError("DataSource has no columns")
        arrays = []
        for key in keys:
            arr = source[key]
            if hasattr(arr, "cpu"):  # torch tensor -> host
                arr = arr.cpu().numpy()
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            arrays.append(arr)
        data_array = np.hstack(arrays)
        return cls._build(
            data_array,
            method=method,
            methods=methods,
            column_names=[str(k) for k in keys],
            column_kinds=None,
        )

    @classmethod
    def _build(
        cls,
        data: NDArray,
        *,
        method: str,
        methods,
        column_names,
        column_kinds,
    ) -> "MICEDesign":
        if data.ndim != 2:
            raise ValidationError(
                f"Data must be 2D (observations x variables), got {data.ndim}D"
            )
        n, p = data.shape
        if n < 2:
            raise ValidationError(f"Need at least 2 observations, got {n}")
        if p < 1:
            raise ValidationError(f"Need at least 1 variable, got {p}")

        # NaN is expected; any other non-finite value (inf, -inf) is an error.
        non_finite = ~np.isfinite(data) & ~np.isnan(data)
        if np.any(non_finite):
            loc = np.where(non_finite)
            raise ValidationError(
                f"Data contains infinite values "
                f"(first at row {loc[0][0]}, col {loc[1][0]}). "
                f"Use NaN for missing values."
            )

        missing_mask = np.isnan(data)

        # Check this before the all-NaN-row guard: with a single column, any
        # missing cell is also an all-NaN row, and "need 2 variables" is the
        # more actionable message for that case.
        has_missing_per_col = np.any(missing_mask, axis=0)
        if np.any(has_missing_per_col) and p < 2:
            raise ValidationError(
                "Chained-equations imputation needs at least 2 variables: a "
                "single column with missing values has no predictors. Use a "
                "univariate imputation method instead."
            )

        all_nan_rows = np.all(missing_mask, axis=1)
        if np.any(all_nan_rows):
            raise ValidationError(
                f"Data contains {int(np.sum(all_nan_rows))} row(s) with all "
                f"values missing; these carry no information for imputation."
            )
        all_nan_cols = np.all(missing_mask, axis=0)
        if np.any(all_nan_cols):
            col_idx = int(np.where(all_nan_cols)[0][0])
            raise ValidationError(
                f"Column {col_idx} is completely missing; it has no observed "
                f"values to build an imputation model from."
            )

        names = _resolve_names(column_names, p)
        kinds = _resolve_kinds(column_kinds, p)
        per_col_methods = _resolve_methods(
            method, methods, names, kinds, has_missing_per_col, p
        )

        return cls(
            _data=np.ascontiguousarray(data, dtype=np.float64),
            _missing_mask=missing_mask,
            _col_names=names,
            _col_kinds=kinds,
            _methods=per_col_methods,
            _n=n,
            _p=p,
        )

    # -------------------------------------------------------------- accessors
    @property
    def data(self) -> NDArray[np.floating[Any]]:
        """Data matrix (n x p), NaN where missing."""
        return self._data

    @property
    def missing_mask(self) -> NDArray[np.bool_]:
        """Boolean (n x p) mask, True where missing."""
        return self._missing_mask

    @property
    def n(self) -> int:
        return self._n

    @property
    def p(self) -> int:
        return self._p

    @property
    def col_names(self) -> tuple[str, ...]:
        return self._col_names

    @property
    def col_kinds(self) -> tuple[str, ...]:
        return self._col_kinds

    @property
    def methods(self) -> tuple[str, ...]:
        """Per-column method name ('' for fully observed columns)."""
        return self._methods

    @property
    def incomplete_columns(self) -> tuple[int, ...]:
        """Indices of columns containing at least one missing value."""
        return tuple(int(j) for j in np.where(np.any(self._missing_mask, axis=0))[0])

    @property
    def n_missing(self) -> int:
        return int(np.sum(self._missing_mask))

    @property
    def missing_rate(self) -> float:
        return self.n_missing / (self._n * self._p)

    @property
    def has_missing(self) -> bool:
        return self.n_missing > 0

    def method_for(self, col: int) -> str:
        """Method name assigned to column ``col`` ('' if fully observed)."""
        return self._methods[col]

    def __repr__(self) -> str:
        return (
            f"MICEDesign(n={self._n}, p={self._p}, "
            f"missing_rate={self.missing_rate:.1%}, "
            f"incomplete={len(self.incomplete_columns)})"
        )


# --------------------------------------------------------------------- helpers
def _coerce_data(data) -> tuple[NDArray, list[str] | None]:
    """Coerce array-like to float64 2D, recovering DataFrame column names."""
    names = None
    if hasattr(data, "values") and hasattr(data, "columns"):
        names = [str(c) for c in data.columns]
        arr = np.asarray(data.values, dtype=np.float64)
    elif hasattr(data, "values"):
        arr = np.asarray(data.values, dtype=np.float64)
    else:
        arr = np.asarray(data, dtype=np.float64)
    return arr, names


def _resolve_names(column_names, p: int) -> tuple[str, ...]:
    if column_names is None:
        return tuple(f"x{j}" for j in range(p))
    names = [str(c) for c in column_names]
    if len(names) != p:
        raise ValidationError(
            f"column_names has length {len(names)}, expected {p}"
        )
    if len(set(names)) != len(names):
        raise ValidationError("column_names must be unique")
    return tuple(names)


def _resolve_kinds(column_kinds, p: int) -> tuple[str, ...]:
    if column_kinds is None:
        kinds = ["numeric"] * p
    else:
        kinds = [str(k) for k in column_kinds]
        if len(kinds) != p:
            raise ValidationError(
                f"column_kinds has length {len(kinds)}, expected {p}"
            )
    # Stage-1 numeric-only guard — the single point where the restriction lives.
    bad = [(j, k) for j, k in enumerate(kinds) if k not in _SUPPORTED_KINDS]
    if bad:
        j, k = bad[0]
        raise ValidationError(
            f"Column {j} has kind {k!r}, but this release supports only "
            f"numeric columns {_SUPPORTED_KINDS}. Categorical methods "
            f"(binary/categorical/ordered) are planned for a later release."
        )
    return tuple(kinds)


def _resolve_methods(
    method: str,
    methods,
    names: tuple[str, ...],
    kinds: tuple[str, ...],
    has_missing_per_col: NDArray[np.bool_],
    p: int,
) -> tuple[str, ...]:
    """Assign a validated method name to each column (per-column override)."""
    # Start with the default for every column.
    resolved = [method] * p

    if methods is not None:
        if isinstance(methods, Mapping):
            name_to_idx = {nm: j for j, nm in enumerate(names)}
            for key, meth in methods.items():
                idx = _method_key_to_index(key, name_to_idx, p)
                resolved[idx] = str(meth)
        elif isinstance(methods, Sequence) and not isinstance(methods, (str, bytes)):
            if len(methods) != p:
                raise ValidationError(
                    f"methods sequence has length {len(methods)}, expected {p}"
                )
            resolved = [str(m) for m in methods]
        else:
            raise ValidationError(
                "methods must be a mapping {column: method} or a length-p "
                f"sequence, got {type(methods).__name__}"
            )

    out: list[str] = []
    for j in range(p):
        if not has_missing_per_col[j]:
            out.append("")  # fully observed: never imputed, method irrelevant
            continue
        meth = resolved[j]
        if not is_registered(meth):
            from pystatistics.mice.methods import available_methods
            raise ValidationError(
                f"Column {names[j]!r} requests unknown method {meth!r}. "
                f"Available: {available_methods()}."
            )
        impl = get_method(meth)
        if impl.target_kind != kinds[j]:
            raise ValidationError(
                f"Column {names[j]!r} is kind {kinds[j]!r} but method "
                f"{meth!r} imputes {impl.target_kind!r} columns."
            )
        out.append(meth)
    return tuple(out)


def _method_key_to_index(key, name_to_idx: dict[str, int], p: int) -> int:
    if isinstance(key, str):
        if key not in name_to_idx:
            raise ValidationError(
                f"methods references unknown column {key!r}. "
                f"Known columns: {list(name_to_idx)}."
            )
        return name_to_idx[key]
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
        idx = int(key)
        if not (0 <= idx < p):
            raise ValidationError(
                f"methods references column index {idx} out of range [0, {p})"
            )
        return idx
    raise ValidationError(
        f"methods keys must be column names or integer indices, got "
        f"{type(key).__name__}"
    )
