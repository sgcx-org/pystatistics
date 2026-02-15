"""
Design validation for mixed models.

MixedDesign validates and organizes the inputs for LMM/GLMM: the
response y, fixed effects matrix X, grouping variables, and random
effect specifications.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MixedDesign:
    """Validated design for a mixed model.

    Attributes:
        y: Response vector (n,).
        X: Fixed effects design matrix (n, p).
        groups: Dict of grouping factor name → group labels (n,).
        random_effects: Dict of group name → list of term names.
        random_data: Dict of variable name → data array (n,).
        n: Number of observations.
        p: Number of fixed effect columns.
    """
    y: NDArray
    X: NDArray
    groups: dict[str, NDArray]
    random_effects: dict[str, list[str]] | None
    random_data: dict[str, NDArray] | None
    n: int
    p: int

    @staticmethod
    def validate(
        y: NDArray,
        X: NDArray,
        groups: dict[str, NDArray],
        random_effects: dict[str, list[str]] | None = None,
        random_data: dict[str, NDArray] | None = None,
    ) -> 'MixedDesign':
        """Validate inputs and create a MixedDesign.

        Args:
            y: Response vector.
            X: Fixed effects design matrix. If 1-D, treated as single predictor
               (intercept should be included by the user or the solver).
            groups: Dict mapping grouping factor names to group label arrays.
            random_effects: Optional dict mapping group names to term lists.
            random_data: Optional dict mapping variable names to data arrays.

        Returns:
            Validated MixedDesign.

        Raises:
            ValueError: On invalid inputs.
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)

        if n < 3:
            raise ValueError(f"Need at least 3 observations, got {n}")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != n:
            raise ValueError(
                f"X has {X.shape[0]} rows, expected {n} (matching y)"
            )
        p = X.shape[1]

        if not groups:
            raise ValueError("At least one grouping factor required")

        # Validate each grouping factor
        groups_validated = {}
        for name, g in groups.items():
            g = np.asarray(g)
            if g.shape[0] != n:
                raise ValueError(
                    f"Group '{name}' has {g.shape[0]} elements, expected {n}"
                )
            unique = np.unique(g)
            if len(unique) < 2:
                raise ValueError(
                    f"Group '{name}' has only {len(unique)} level(s), "
                    f"need at least 2"
                )
            groups_validated[name] = g

        # Validate random_effects keys match groups
        if random_effects is not None:
            for name in random_effects:
                if name not in groups:
                    raise ValueError(
                        f"Random effect group '{name}' not found in groups dict. "
                        f"Available: {list(groups.keys())}"
                    )

        # Validate random_data
        if random_data is not None:
            for name, data in random_data.items():
                data = np.asarray(data, dtype=np.float64)
                if data.shape[0] != n:
                    raise ValueError(
                        f"Random data '{name}' has {data.shape[0]} elements, "
                        f"expected {n}"
                    )

        if not np.all(np.isfinite(y)):
            raise ValueError("y contains non-finite values (NaN or Inf)")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non-finite values (NaN or Inf)")

        return MixedDesign(
            y=y,
            X=X,
            groups=groups_validated,
            random_effects=random_effects,
            random_data=random_data,
            n=n,
            p=p,
        )
