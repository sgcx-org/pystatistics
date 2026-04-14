"""
Factory functions for building HypothesisDesign instances.

Each function validates inputs for a specific test type and returns
a configured HypothesisDesign. Extracted from design.py per Rule 4
(500-line code limit).
"""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray, ArrayLike

from pystatistics.core.exceptions import ValidationError
from pystatistics.hypothesis._common import VALID_ALTERNATIVES


def _validate_alternative(alternative: str) -> str:
    """Validate and return alternative hypothesis string."""
    if alternative not in VALID_ALTERNATIVES:
        raise ValidationError(
            f"alternative must be one of {VALID_ALTERNATIVES}, got {alternative!r}"
        )
    return alternative


def _validate_conf_level(conf_level: float) -> float:
    """Validate confidence level is in (0, 1)."""
    if not (0.0 < conf_level < 1.0):
        raise ValidationError(
            f"conf_level must be in (0, 1), got {conf_level}"
        )
    return conf_level


def _to_float64_1d(x: ArrayLike, name: str = "x") -> NDArray[np.floating[Any]]:
    """Convert to 1D float64 array, removing NaN values."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    mask = ~np.isnan(arr)
    return arr[mask]


def build_t_test_design(
    x: ArrayLike,
    y: ArrayLike | None = None,
    *,
    mu: float = 0.0,
    paired: bool = False,
    var_equal: bool = False,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
):
    """Build design for t_test()."""
    from pystatistics.hypothesis.design import HypothesisDesign

    alternative = _validate_alternative(alternative)
    conf_level = _validate_conf_level(conf_level)

    x_arr = _to_float64_1d(x, "x")
    if len(x_arr) < 2:
        raise ValidationError(
            f"Need at least 2 non-missing observations in x, got {len(x_arr)}"
        )

    y_arr = None
    data_name = "x"

    if y is not None:
        y_raw = np.asarray(y, dtype=np.float64).ravel()

        if paired:
            x_raw = np.asarray(x, dtype=np.float64).ravel()
            if len(x_raw) != len(y_raw):
                raise ValidationError(
                    f"Paired t-test requires equal lengths: "
                    f"len(x)={len(x_raw)}, len(y)={len(y_raw)}"
                )
            diffs = x_raw - y_raw
            mask = ~np.isnan(diffs)
            x_arr = diffs[mask]
            if len(x_arr) < 2:
                raise ValidationError(
                    f"Need at least 2 non-missing paired differences, "
                    f"got {len(x_arr)}"
                )
            test_type = "t_paired"
            data_name = "x and y"
        else:
            y_arr = _to_float64_1d(y, "y")
            if len(y_arr) < 2:
                raise ValidationError(
                    f"Need at least 2 non-missing observations in y, "
                    f"got {len(y_arr)}"
                )
            test_type = "t_two_sample"
            data_name = "x and y"
    else:
        test_type = "t_one_sample"

    return HypothesisDesign(
        test_type=test_type,
        _x=x_arr, _y=y_arr, _mu=mu,
        _paired=paired, _var_equal=var_equal,
        _alternative=alternative, _conf_level=conf_level,
        _data_name=data_name,
    )


def build_chisq_test_design(
    x: ArrayLike,
    y: ArrayLike | None = None,
    *,
    correct: bool = True,
    p: ArrayLike | None = None,
    rescale_p: bool = False,
    simulate_p_value: bool = False,
    B: int = 2000,
    seed: int | None = None,
):
    """Build design for chisq_test()."""
    from pystatistics.hypothesis.design import HypothesisDesign

    x_arr = np.asarray(x, dtype=np.float64)

    if x_arr.ndim == 2:
        table = x_arr.copy()
        if table.shape[0] < 2 or table.shape[1] < 2:
            raise ValidationError(
                "Contingency table must have at least 2 rows and 2 columns"
            )
        if np.any(table < 0):
            raise ValidationError(
                "All entries in contingency table must be non-negative"
            )
        return HypothesisDesign(
            test_type="chisq_independence",
            _table=table, _correct=correct,
            _simulate_p_value=simulate_p_value,
            _n_monte_carlo=B, _seed=seed,
            _data_name="x",
        )

    if y is not None:
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        x_1d = x_arr.ravel()
        if len(x_1d) != len(y_arr):
            raise ValidationError(
                f"x and y must have the same length for contingency table, "
                f"got {len(x_1d)} and {len(y_arr)}"
            )
        x_cats = np.unique(x_1d)
        y_cats = np.unique(y_arr)
        table = np.zeros((len(x_cats), len(y_cats)), dtype=np.float64)
        for i, xc in enumerate(x_cats):
            for j, yc in enumerate(y_cats):
                table[i, j] = np.sum((x_1d == xc) & (y_arr == yc))

        return HypothesisDesign(
            test_type="chisq_independence",
            _table=table, _correct=correct,
            _simulate_p_value=simulate_p_value,
            _n_monte_carlo=B, _seed=seed,
            _data_name="x and y",
        )

    # 1D vector: goodness-of-fit test
    x_1d = x_arr.ravel()
    if len(x_1d) < 2:
        raise ValidationError(
            "Need at least 2 categories for goodness-of-fit test"
        )
    if np.any(x_1d < 0):
        raise ValidationError(
            "All observed counts must be non-negative"
        )

    p_arr = None
    if p is not None:
        p_arr = np.asarray(p, dtype=np.float64).ravel()
        if len(p_arr) != len(x_1d):
            raise ValidationError(
                f"length of p ({len(p_arr)}) must equal length of x ({len(x_1d)})"
            )
        if not rescale_p:
            p_sum = np.sum(p_arr)
            if abs(p_sum - 1.0) > 1e-7:
                raise ValidationError(
                    f"probabilities must sum to 1, got {p_sum:.10g}. "
                    f"Use rescale_p=True to normalize."
                )
        if np.any(p_arr <= 0):
            raise ValidationError("all probabilities must be positive")

    return HypothesisDesign(
        test_type="chisq_gof",
        _x=x_1d, _expected_p=p_arr,
        _rescale_p=rescale_p,
        _simulate_p_value=simulate_p_value,
        _n_monte_carlo=B, _seed=seed,
        _data_name="x",
    )


def build_prop_test_design(
    x: ArrayLike,
    n: ArrayLike,
    *,
    p: ArrayLike | float | None = None,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
    correct: bool = True,
):
    """Build design for prop_test()."""
    from pystatistics.hypothesis.design import HypothesisDesign

    alternative = _validate_alternative(alternative)
    conf_level = _validate_conf_level(conf_level)

    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    n_arr = np.atleast_1d(np.asarray(n, dtype=np.float64))

    if len(x_arr) != len(n_arr):
        raise ValidationError(
            f"x and n must have same length, got {len(x_arr)} and {len(n_arr)}"
        )

    if np.any(n_arr <= 0):
        raise ValidationError("All trial counts must be positive")
    if np.any(x_arr < 0) or np.any(x_arr > n_arr):
        raise ValidationError(
            "All success counts must be between 0 and n"
        )

    p_arr = None
    if p is not None:
        p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
        if len(p_arr) == 1 and len(x_arr) > 1:
            p_arr = np.repeat(p_arr, len(x_arr))
        if len(p_arr) != len(x_arr):
            raise ValidationError(
                f"length of p ({len(p_arr)}) must match x ({len(x_arr)})"
            )
        if np.any(p_arr <= 0) or np.any(p_arr >= 1):
            raise ValidationError(
                "All null proportions must be in (0, 1)"
            )

    if len(x_arr) > 1 and alternative != "two.sided":
        raise ValidationError(
            "alternative must be 'two.sided' for k > 1 groups"
        )

    data_name = "x out of n" if len(x_arr) == 1 else "x"

    return HypothesisDesign(
        test_type="prop_test",
        _successes=x_arr, _trials=n_arr,
        _expected_p=p_arr,
        _alternative=alternative, _conf_level=conf_level,
        _correct=correct, _data_name=data_name,
    )


def build_fisher_test_design(
    x: ArrayLike,
    y: ArrayLike | None = None,
    *,
    alternative: str = "two.sided",
    conf_int: bool = True,
    conf_level: float = 0.95,
    simulate_p_value: bool = False,
    B: int = 2000,
    seed: int | None = None,
):
    """Build design for fisher_test()."""
    from pystatistics.hypothesis.design import HypothesisDesign

    alternative = _validate_alternative(alternative)
    conf_level = _validate_conf_level(conf_level)

    x_arr = np.asarray(x, dtype=np.float64)

    if x_arr.ndim == 2:
        table = x_arr.copy()
    elif y is not None:
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        x_1d = x_arr.ravel()
        if len(x_1d) != len(y_arr):
            raise ValidationError(
                f"x and y must have the same length, "
                f"got {len(x_1d)} and {len(y_arr)}"
            )
        x_cats = np.unique(x_1d)
        y_cats = np.unique(y_arr)
        table = np.zeros((len(x_cats), len(y_cats)), dtype=np.float64)
        for i, xc in enumerate(x_cats):
            for j, yc in enumerate(y_cats):
                table[i, j] = np.sum((x_1d == xc) & (y_arr == yc))
    else:
        raise ValidationError(
            "Fisher test requires a 2D contingency table or two vectors"
        )

    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValidationError(
            "Contingency table must have at least 2 rows and 2 columns"
        )
    if np.any(table < 0):
        raise ValidationError(
            "All entries in contingency table must be non-negative"
        )

    nrow, ncol = table.shape
    if (nrow > 2 or ncol > 2) and alternative != "two.sided":
        raise ValidationError(
            "alternative must be 'two.sided' for r x c tables "
            "with r > 2 or c > 2"
        )

    return HypothesisDesign(
        test_type="fisher_test",
        _table=table,
        _alternative=alternative,
        _compute_conf_int=conf_int, _conf_level=conf_level,
        _simulate_p_value=simulate_p_value,
        _n_monte_carlo=B, _seed=seed,
        _data_name="x",
    )


def build_wilcox_test_design(
    x: ArrayLike,
    y: ArrayLike | None = None,
    *,
    mu: float = 0.0,
    paired: bool = False,
    exact: bool | None = None,
    correct: bool = True,
    conf_int: bool = True,
    conf_level: float = 0.95,
    alternative: str = "two.sided",
):
    """Build design for wilcox_test()."""
    from pystatistics.hypothesis.design import HypothesisDesign

    alternative = _validate_alternative(alternative)
    conf_level = _validate_conf_level(conf_level)

    x_arr = _to_float64_1d(x, "x")

    if y is not None:
        if paired:
            y_raw = np.asarray(y, dtype=np.float64).ravel()
            x_raw = np.asarray(x, dtype=np.float64).ravel()
            if len(x_raw) != len(y_raw):
                raise ValidationError(
                    f"Paired test requires equal lengths: "
                    f"len(x)={len(x_raw)}, len(y)={len(y_raw)}"
                )
            diffs = x_raw - y_raw
            mask = ~np.isnan(diffs)
            x_arr = diffs[mask]
            test_type = "wilcox_signed_rank"
            data_name = "x and y"
        else:
            y_arr = _to_float64_1d(y, "y")
            test_type = "wilcox_rank_sum"
            data_name = "x and y"
            return HypothesisDesign(
                test_type=test_type,
                _x=x_arr, _y=y_arr, _mu=mu,
                _alternative=alternative, _conf_level=conf_level,
                _correct=correct, _exact=exact,
                _compute_wilcox_ci=conf_int,
                _data_name=data_name,
            )
    else:
        test_type = "wilcox_signed_rank"
        data_name = "x"

    return HypothesisDesign(
        test_type=test_type,
        _x=x_arr, _mu=mu,
        _alternative=alternative, _conf_level=conf_level,
        _correct=correct, _exact=exact,
        _compute_wilcox_ci=conf_int,
        _data_name=data_name,
    )


def build_ks_test_design(
    x: ArrayLike,
    y: ArrayLike | None = None,
    *,
    alternative: str = "two.sided",
    distribution: str | None = None,
    **dist_params: float,
):
    """Build design for ks_test()."""
    from pystatistics.hypothesis.design import HypothesisDesign

    alternative = _validate_alternative(alternative)

    x_arr = _to_float64_1d(x, "x")
    if len(x_arr) < 1:
        raise ValidationError("Need at least 1 observation in x")

    if y is not None:
        y_arr = _to_float64_1d(y, "y")
        if len(y_arr) < 1:
            raise ValidationError("Need at least 1 observation in y")
        return HypothesisDesign(
            test_type="ks_two_sample",
            _x=x_arr, _y=y_arr,
            _alternative=alternative,
            _data_name="x and y",
        )

    return HypothesisDesign(
        test_type="ks_one_sample",
        _x=x_arr,
        _alternative=alternative,
        _distribution=distribution,
        _dist_params=dict(dist_params) if dist_params else {},
        _data_name="x",
    )


def build_var_test_design(
    x: ArrayLike,
    y: ArrayLike,
    *,
    ratio: float = 1.0,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
):
    """Build design for var_test()."""
    from pystatistics.hypothesis.design import HypothesisDesign

    alternative = _validate_alternative(alternative)
    conf_level = _validate_conf_level(conf_level)

    x_arr = _to_float64_1d(x, "x")
    y_arr = _to_float64_1d(y, "y")

    if len(x_arr) < 2:
        raise ValidationError(
            f"Need at least 2 non-missing observations in x, got {len(x_arr)}"
        )
    if len(y_arr) < 2:
        raise ValidationError(
            f"Need at least 2 non-missing observations in y, got {len(y_arr)}"
        )
    if ratio <= 0:
        raise ValidationError(
            f"ratio must be positive, got {ratio}"
        )

    return HypothesisDesign(
        test_type="var_test",
        _x=x_arr, _y=y_arr,
        _ratio=ratio,
        _alternative=alternative, _conf_level=conf_level,
        _data_name="x and y",
    )
