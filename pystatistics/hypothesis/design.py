"""
HypothesisDesign: tagged union for hypothesis test inputs.

Uses factory classmethods per test type. The `test_type` field identifies
which fields are populated. Immutable after construction.

Factory logic is in _design_factories.py (extracted per Rule 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray, ArrayLike


@dataclass(frozen=True)
class HypothesisDesign:
    """
    Design for hypothesis tests.

    Uses a tagged-union approach: the `test_type` field identifies which
    fields are populated. Factory classmethods validate inputs.

    Do not construct directly; use factory classmethods.
    """
    test_type: str

    # Numeric vectors
    _x: NDArray[np.floating[Any]] | None = None
    _y: NDArray[np.floating[Any]] | None = None

    # Test configuration
    _mu: float = 0.0
    _alternative: str = "two.sided"
    _conf_level: float = 0.95
    _var_equal: bool = False
    _paired: bool = False
    _correct: bool = True

    # Contingency table
    _table: NDArray[np.floating[Any]] | None = None

    # Proportion test
    _successes: NDArray[np.floating[Any]] | None = None
    _trials: NDArray[np.floating[Any]] | None = None
    _expected_p: NDArray[np.floating[Any]] | None = None
    _rescale_p: bool = False

    # Monte Carlo
    _simulate_p_value: bool = False
    _n_monte_carlo: int = 2000
    _seed: int | None = None

    # Fisher-specific
    _compute_conf_int: bool = True

    # Wilcoxon-specific
    _exact: bool | None = None
    _compute_wilcox_ci: bool = True

    # KS-specific
    _distribution: str | None = None
    _dist_params: dict[str, float] | None = None

    # Var test
    _ratio: float = 1.0

    # Metadata
    _data_name: str = ""

    # --- Properties ---

    @property
    def x(self) -> NDArray[np.floating[Any]] | None:
        return self._x

    @property
    def y(self) -> NDArray[np.floating[Any]] | None:
        return self._y

    @property
    def table(self) -> NDArray[np.floating[Any]] | None:
        return self._table

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def alternative(self) -> str:
        return self._alternative

    @property
    def conf_level(self) -> float:
        return self._conf_level

    @property
    def var_equal(self) -> bool:
        return self._var_equal

    @property
    def paired(self) -> bool:
        return self._paired

    @property
    def correct(self) -> bool:
        return self._correct

    @property
    def simulate_p_value(self) -> bool:
        return self._simulate_p_value

    @property
    def n_monte_carlo(self) -> int:
        return self._n_monte_carlo

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def compute_conf_int(self) -> bool:
        return self._compute_conf_int

    @property
    def exact(self) -> bool | None:
        return self._exact

    @property
    def compute_wilcox_ci(self) -> bool:
        return self._compute_wilcox_ci

    @property
    def distribution(self) -> str | None:
        return self._distribution

    @property
    def dist_params(self) -> dict[str, float] | None:
        return self._dist_params

    @property
    def ratio(self) -> float:
        return self._ratio

    @property
    def successes(self) -> NDArray[np.floating[Any]] | None:
        return self._successes

    @property
    def trials(self) -> NDArray[np.floating[Any]] | None:
        return self._trials

    @property
    def expected_p(self) -> NDArray[np.floating[Any]] | None:
        return self._expected_p

    @property
    def rescale_p(self) -> bool:
        return self._rescale_p

    @property
    def data_name(self) -> str:
        return self._data_name

    # --- Factory classmethods (thin wrappers over _design_factories) ---

    @classmethod
    def for_t_test(cls, x, y=None, **kwargs):
        """Build design for t_test(). See _design_factories.build_t_test_design."""
        from pystatistics.hypothesis._design_factories import build_t_test_design
        return build_t_test_design(x, y, **kwargs)

    @classmethod
    def for_chisq_test(cls, x, y=None, **kwargs):
        """Build design for chisq_test()."""
        from pystatistics.hypothesis._design_factories import build_chisq_test_design
        return build_chisq_test_design(x, y, **kwargs)

    @classmethod
    def for_prop_test(cls, x, n, **kwargs):
        """Build design for prop_test()."""
        from pystatistics.hypothesis._design_factories import build_prop_test_design
        return build_prop_test_design(x, n, **kwargs)

    @classmethod
    def for_fisher_test(cls, x, y=None, **kwargs):
        """Build design for fisher_test()."""
        from pystatistics.hypothesis._design_factories import build_fisher_test_design
        return build_fisher_test_design(x, y, **kwargs)

    @classmethod
    def for_wilcox_test(cls, x, y=None, **kwargs):
        """Build design for wilcox_test()."""
        from pystatistics.hypothesis._design_factories import build_wilcox_test_design
        return build_wilcox_test_design(x, y, **kwargs)

    @classmethod
    def for_ks_test(cls, x, y=None, **kwargs):
        """Build design for ks_test()."""
        from pystatistics.hypothesis._design_factories import build_ks_test_design
        return build_ks_test_design(x, y, **kwargs)

    @classmethod
    def for_var_test(cls, x, y, **kwargs):
        """Build design for var_test()."""
        from pystatistics.hypothesis._design_factories import build_var_test_design
        return build_var_test_design(x, y, **kwargs)

    def __repr__(self) -> str:
        n_x = len(self._x) if self._x is not None else 0
        n_y = len(self._y) if self._y is not None else 0
        if self._table is not None:
            return (
                f"HypothesisDesign(test_type={self.test_type!r}, "
                f"table={self._table.shape})"
            )
        if n_y > 0:
            return (
                f"HypothesisDesign(test_type={self.test_type!r}, "
                f"n_x={n_x}, n_y={n_y})"
            )
        return (
            f"HypothesisDesign(test_type={self.test_type!r}, n={n_x})"
        )
