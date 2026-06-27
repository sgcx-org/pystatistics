"""
MICE solution types.

``MICEParams`` is the immutable payload backends produce; ``MICESolution`` is the
user-facing wrapper exposing the ``m`` completed datasets, the per-column
imputed values, and the convergence traces used for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.result import Result, SolutionReprMixin

if TYPE_CHECKING:
    from pystatistics.mice.design import MICEDesign


@dataclass(frozen=True)
class MICEParams:
    """Immutable imputation payload computed by a backend."""

    completed: NDArray[np.floating[Any]]   # (m, n, p) — no NaN
    chain_mean: NDArray[np.floating[Any]]  # (m, maxit, n_incomplete)
    chain_var: NDArray[np.floating[Any]]   # (m, maxit, n_incomplete)
    incomplete_columns: tuple[int, ...]
    m: int
    maxit: int
    visit_sequence: tuple[int, ...]


@dataclass
class MICESolution(SolutionReprMixin):
    """User-facing MICE results: ``m`` completed datasets plus diagnostics."""

    _result: Result[MICEParams]
    _design: "MICEDesign"

    # ----------------------------------------------------------- basic shape
    @property
    def n_imputations(self) -> int:
        """Number of imputations (completed datasets)."""
        return self._result.params.m

    @property
    def max_iter(self) -> int:
        """Iterations per chain."""
        return self._result.params.maxit

    @property
    def incomplete_columns(self) -> tuple[int, ...]:
        """Indices of columns that were imputed."""
        return self._result.params.incomplete_columns

    @property
    def visit_sequence(self) -> tuple[int, ...]:
        """Column visit order used within each iteration."""
        return self._result.params.visit_sequence

    # ------------------------------------------------------ completed datasets
    def completed(self, i: int) -> NDArray[np.floating[Any]]:
        """The ``i``-th completed dataset (n x p), 0-based."""
        if not (0 <= i < self.n_imputations):
            raise ValidationError(
                f"imputation index {i} out of range [0, {self.n_imputations})"
            )
        return self._result.params.completed[i]

    def completed_datasets(self) -> list[NDArray[np.floating[Any]]]:
        """All ``m`` completed datasets as a list of (n x p) arrays."""
        return [self._result.params.completed[i] for i in range(self.n_imputations)]

    def __iter__(self) -> Iterator[NDArray[np.floating[Any]]]:
        """Iterate over the ``m`` completed datasets."""
        return iter(self.completed_datasets())

    def imputations(self, col: int) -> NDArray[np.floating[Any]]:
        """Imputed values for one column, shape (m, n_missing_in_col).

        Row ``i`` holds the values the ``i``-th imputation placed into that
        column's missing cells, in row order. Mirrors R's ``imp$imp[[var]]``.
        """
        if col not in self.incomplete_columns:
            raise ValidationError(
                f"column {col} has no missing values; nothing was imputed. "
                f"Imputed columns: {self.incomplete_columns}."
            )
        mask_col = self._design.missing_mask[:, col]
        completed = self._result.params.completed
        return np.stack([completed[i][mask_col, col] for i in range(self.n_imputations)])

    # ----------------------------------------------------------- diagnostics
    @property
    def chain_mean(self) -> NDArray[np.floating[Any]]:
        """Per-iteration mean of imputed cells, shape (m, maxit, n_incomplete)."""
        return self._result.params.chain_mean

    @property
    def chain_var(self) -> NDArray[np.floating[Any]]:
        """Per-iteration variance of imputed cells, (m, maxit, n_incomplete)."""
        return self._result.params.chain_var

    # ------------------------------------------------------------- metadata
    @property
    def info(self) -> dict[str, Any]:
        return self._result.info

    @property
    def timing(self) -> dict[str, float] | None:
        return self._result.timing

    @property
    def backend_name(self) -> str:
        return self._result.backend_name

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._result.warnings

    def summary(self) -> str:
        """Human-readable summary."""
        names = self._design.col_names
        lines = [
            "MICE: Multiple Imputation by Chained Equations",
            "=" * 60,
            f"Observations: {self._design.n}",
            f"Variables: {self._design.p}",
            f"Missing rate: {self._design.missing_rate:.1%}",
            f"Imputations: {self.n_imputations}",
            f"Iterations: {self.max_iter}",
            f"Backend: {self.backend_name}",
            "",
            "Imputed columns (method):",
            "-" * 60,
        ]
        for j in self.incomplete_columns:
            n_mis = int(np.count_nonzero(self._design.missing_mask[:, j]))
            lines.append(
                f"  {names[j]}: {n_mis} missing, method={self._design.method_for(j)!r}"
            )
        lines.append("-" * 60)
        if self.timing:
            lines.append(f"Time: {self.timing.get('total_seconds', 0):.4f}s")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MICESolution(n_imputations={self.n_imputations}, max_iter={self.max_iter}, "
            f"n={self._design.n}, p={self._design.p}, "
            f"imputed={len(self.incomplete_columns)})"
        )
