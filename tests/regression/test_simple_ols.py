"""Tests for ``regression.simple_ols`` — the lean univariate OLS front door.

Coverage (Rule 7):
- Normal: a known line validated against genuine R ``lm(y ~ x)`` output to
  ``rtol=1e-10`` (coef, R², adjusted R², slope Std. Error), cross-checked
  against ``scipy.stats.linregress`` and hand-computed values; positive and
  negative slope.
- Edge: minimal ``n = 3``; near-collinear points (R² ≈ 1); very small
  magnitudes; large intercept.
- Failure: length mismatch; ``n < 3``; non-finite input (x and y);
  zero-variance x; zero-variance y — each raises ``ValidationError``.

The R reference constants below were produced by ``lm(y ~ x)`` /
``summary(lm)`` on the reference dataset (R 4.x). ``scipy.stats.linregress``
agrees with R ``lm`` to full double precision on well-conditioned data, so the
linregress cross-checks are an independent second opinion on every case.
"""

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy import stats

from pystatistics.core.exceptions import ValidationError
from pystatistics.regression import SimpleOLSResult, simple_ols

# --- Reference dataset & genuine R lm() output (see module docstring) ---------
_REF_X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
_REF_Y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1, 18.2, 19.9])
_R_SLOPE = 1.9993939393939391
_R_INTERCEPT = 0.033333333333335553
_R_R_SQUARED = 0.99926987950900137
_R_ADJ_R_SQUARED = 0.99917861444762657
_R_SLOPE_SE = 0.019107736691435673

_RTOL = 1e-10


def test_reference_case_matches_r_lm():
    """Every reported quantity matches R ``lm(y ~ x)`` to rtol=1e-10."""
    r = simple_ols(_REF_X, _REF_Y)
    assert isinstance(r, SimpleOLSResult)
    assert r.slope == pytest.approx(_R_SLOPE, rel=_RTOL)
    assert r.intercept == pytest.approx(_R_INTERCEPT, rel=_RTOL)
    assert r.r_squared == pytest.approx(_R_R_SQUARED, rel=_RTOL)
    assert r.adjusted_r_squared == pytest.approx(_R_ADJ_R_SQUARED, rel=_RTOL)
    assert r.slope_se == pytest.approx(_R_SLOPE_SE, rel=_RTOL)
    assert r.n == 10


def test_reference_case_matches_linregress():
    """Cross-check the reference case against scipy.stats.linregress."""
    r = simple_ols(_REF_X, _REF_Y)
    lr = stats.linregress(_REF_X, _REF_Y)
    assert r.slope == pytest.approx(lr.slope, rel=_RTOL)
    assert r.intercept == pytest.approx(lr.intercept, rel=_RTOL)
    assert r.r_squared == pytest.approx(lr.rvalue**2, rel=_RTOL)
    assert r.slope_se == pytest.approx(lr.stderr, rel=_RTOL)


def test_exact_line_positive_slope():
    """A perfect positive line: slope/intercept exact, R² == 1, slope_se == 0."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 3.0 * x + 5.0  # slope 3, intercept 5
    r = simple_ols(x, y)
    assert r.slope == pytest.approx(3.0, rel=_RTOL)
    assert r.intercept == pytest.approx(5.0, rel=_RTOL)
    assert r.r_squared == pytest.approx(1.0, rel=_RTOL)
    assert r.adjusted_r_squared == pytest.approx(1.0, rel=_RTOL)
    assert r.slope_se == pytest.approx(0.0, abs=1e-12)


def test_negative_slope_matches_linregress():
    """Negative slope with noise agrees with scipy.stats.linregress."""
    x = np.linspace(-3.0, 4.0, 12)
    y = -2.5 * x + 1.0 + np.array(
        [0.3, -0.2, 0.1, -0.4, 0.25, 0.05, -0.15, 0.35, -0.3, 0.2, -0.1, 0.4]
    )
    r = simple_ols(x, y)
    lr = stats.linregress(x, y)
    assert r.slope < 0
    assert r.slope == pytest.approx(lr.slope, rel=_RTOL)
    assert r.intercept == pytest.approx(lr.intercept, rel=_RTOL)
    assert r.r_squared == pytest.approx(lr.rvalue**2, rel=_RTOL)
    assert r.slope_se == pytest.approx(lr.stderr, rel=_RTOL)


def test_hand_computed_values():
    """Independently hand-computed slope/intercept/R²/adj-R²/slope_se.

    x̄=2.5, ȳ=3.75; Sxx=5, Sxy=3.5 → slope=0.7, intercept=2.0.
    ŷ=[2.7,3.4,4.1,4.8]; RSS=2.30; Syy=4.75; R²=Sxy²/(Sxx·Syy)=12.25/23.75.
    adj=1−(1−R²)·3/2; slope_se=sqrt((RSS/2)/Sxx)=sqrt(0.23).
    Cross-checked against R lm(): slope 0.7, R² 0.51578947368421058,
    adj 0.27368421052631586, slope_se 0.47958315233127186.
    """
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 5.0, 4.0])
    r = simple_ols(x, y)
    assert r.slope == pytest.approx(0.7, rel=_RTOL)
    assert r.intercept == pytest.approx(2.0, rel=_RTOL)
    assert r.r_squared == pytest.approx(12.25 / 23.75, rel=_RTOL)
    assert r.adjusted_r_squared == pytest.approx(
        1.0 - (1.0 - 12.25 / 23.75) * 3.0 / 2.0, rel=_RTOL
    )
    assert r.slope_se == pytest.approx(math.sqrt((2.30 / 2.0) / 5.0), rel=_RTOL)


def test_accepts_python_lists():
    """1-D array-likes (plain lists) are accepted, not just ndarrays."""
    r = simple_ols([1, 2, 3, 4], [2.0, 4.1, 5.9, 8.2])
    lr = stats.linregress([1, 2, 3, 4], [2.0, 4.1, 5.9, 8.2])
    assert r.slope == pytest.approx(lr.slope, rel=_RTOL)
    assert r.n == 4


def test_result_is_frozen():
    """SimpleOLSResult is immutable (frozen dataclass)."""
    r = simple_ols(_REF_X, _REF_Y)
    with pytest.raises(FrozenInstanceError):
        r.slope = 0.0  # type: ignore[misc]


# --- Edge cases ---------------------------------------------------------------


def test_minimal_n_equals_three():
    """n = 3 is the smallest allowed sample; adj-R² and slope_se are defined."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 2.0])
    r = simple_ols(x, y)
    lr = stats.linregress(x, y)
    assert r.n == 3
    assert r.slope == pytest.approx(lr.slope, rel=_RTOL)
    assert r.intercept == pytest.approx(lr.intercept, rel=_RTOL)
    assert r.slope_se == pytest.approx(lr.stderr, rel=_RTOL)
    assert math.isfinite(r.adjusted_r_squared)


def test_near_collinear_high_r_squared():
    """Near-collinear data (R² ≈ 1): slope_se stays accurate — no cancellation.

    Because ``simple_ols`` computes RSS from residuals directly (not the
    algebraically-equal ``Syy − Sxy²/Sxx``, which cancels catastrophically when
    R² ≈ 1), it recovers R lm()'s slope_se to ~1e-10 here. ``scipy.stats.
    linregress``'s cancellation-prone formula is unreliable in this regime
    (it mis-estimates the slope's stderr by tens of percent), so we validate
    against R — and additionally show that ``simple_ols`` lands closer to R
    than linregress does. Dataset (x = 0..49, y = 7x + 3 + 1e-6·(−1)ⁱ) is
    bit-identical between numpy and R.
    """
    x = np.arange(50.0)
    y = 7.0 * x + 3.0 + 1e-6 * ((-1.0) ** np.arange(50))
    r_se = 9.9959951974547511e-09  # genuine R lm() slope Std. Error
    r = simple_ols(x, y)
    # Genuine R lm() output on the identical arrays:
    assert r.slope == pytest.approx(6.9999999975990397, rel=_RTOL)
    assert r.r_squared == pytest.approx(0.99999999999999989, rel=_RTOL)
    assert r.slope_se == pytest.approx(r_se, rel=1e-6)
    # The design win: our residual-based slope_se is closer to R than
    # linregress's cancellation-prone estimate.
    our_err = abs(r.slope_se - r_se)
    linregress_err = abs(stats.linregress(x, y).stderr - r_se)
    assert our_err < linregress_err


def test_small_magnitudes():
    """Very small-magnitude inputs are scale-invariant vs linregress."""
    x = np.array([1e-8, 2e-8, 3e-8, 4e-8, 5e-8])
    y = np.array([2e-9, 4.1e-9, 5.9e-9, 8.2e-9, 9.8e-9])
    r = simple_ols(x, y)
    lr = stats.linregress(x, y)
    assert r.slope == pytest.approx(lr.slope, rel=1e-9)
    assert r.intercept == pytest.approx(lr.intercept, rel=1e-6, abs=1e-18)
    assert r.r_squared == pytest.approx(lr.rvalue**2, rel=_RTOL)


def test_large_intercept():
    """A large intercept relative to the slope term is recovered accurately."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = 1.0e9 + 2.0 * x + np.array([0.5, -0.5, 0.2, -0.2, 0.1, -0.1])
    r = simple_ols(x, y)
    lr = stats.linregress(x, y)
    assert r.intercept == pytest.approx(lr.intercept, rel=_RTOL)
    assert r.slope == pytest.approx(lr.slope, rel=_RTOL)
    assert r.intercept > 1e8


# --- Failure cases (fail loud, Rule 1 / validate at boundary, Rule 2) ---------


def test_length_mismatch_raises():
    with pytest.raises(ValidationError, match="same length"):
        simple_ols([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])


def test_too_few_observations_raises():
    with pytest.raises(ValidationError, match="at least 3 observations"):
        simple_ols([1.0, 2.0], [3.0, 4.0])


def test_non_1d_input_raises():
    with pytest.raises(ValidationError, match="1-dimensional"):
        simple_ols(np.array([[1.0, 2.0, 3.0]]), np.array([1.0, 2.0, 3.0]))


def test_non_finite_x_raises():
    with pytest.raises(ValidationError, match="x contains non-finite"):
        simple_ols([1.0, np.nan, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])


def test_non_finite_x_inf_raises():
    with pytest.raises(ValidationError, match="x contains non-finite"):
        simple_ols([1.0, np.inf, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])


def test_non_finite_y_raises():
    with pytest.raises(ValidationError, match="y contains non-finite"):
        simple_ols([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, np.nan, 4.0])


def test_zero_variance_x_raises():
    with pytest.raises(ValidationError, match="x has zero variance"):
        simple_ols([2.0, 2.0, 2.0, 2.0], [1.0, 2.0, 3.0, 4.0])


def test_zero_variance_y_raises():
    with pytest.raises(ValidationError, match="y has zero variance"):
        simple_ols([1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0])
