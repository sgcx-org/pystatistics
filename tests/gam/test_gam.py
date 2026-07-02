"""Tests for the GAM module (stable augmented-QR rewrite).

Covers, per module test policy: normal cases, edge cases, failure cases —
plus explicit regression tests for the four 4.5.x defects the rewrite
fixed (unconstrained singular design / garbage EDF, tp null-space loss,
placeholder standard errors, broken REML criterion).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pystatistics.core.exceptions import ValidationError
from pystatistics.gam import GAMSolution, SmoothTerm, gam, s
from pystatistics.gam._basis_cr import cr_basis, place_knots_cr
from pystatistics.gam._basis_tp import tp_basis
from pystatistics.gam._constraints import absorb_sum_to_zero


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sine_data():
    rng = np.random.default_rng(42)
    n = 200
    x = np.sort(rng.uniform(0.0, 1.0, n))
    f = np.sin(2.0 * np.pi * x)
    y = f + rng.normal(0.0, 0.2, n)
    return x, y, f


@pytest.fixture()
def two_smooth_data():
    rng = np.random.default_rng(7)
    n = 300
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    y = np.sin(2 * np.pi * x1) + 0.5 * (x2 - 0.5) ** 2 + rng.normal(0, 0.2, n)
    return x1, x2, y


# ---------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------

class TestCrBasis:
    def test_shape_and_dimension_is_k(self):
        x = np.linspace(0.0, 1.0, 60)
        B, S, s_scale = cr_basis(x, k=10)
        assert B.shape == (60, 10)      # k columns, exactly as mgcv
        assert S.shape == (10, 10)
        assert s_scale > 0.0

    def test_penalty_symmetric_psd(self):
        x = np.linspace(0.0, 1.0, 60)
        _, S, _ = cr_basis(x, k=8)
        assert np.allclose(S, S.T)
        ev = np.linalg.eigvalsh(S)
        assert ev.min() > -1e-10
        # rank k-2 (null space: constant + linear)
        assert np.sum(ev > ev.max() * 1e-9) == 6

    def test_cardinal_interpolation_property(self):
        """b_j(knot_l) = delta_jl — the defining property of the basis."""
        from pystatistics.gam._basis_cr import _cardinal_basis
        x = np.linspace(0.0, 1.0, 40)
        knots = place_knots_cr(x, 7)
        Bk = _cardinal_basis(knots, knots)
        assert np.allclose(Bk, np.eye(7), atol=1e-10)

    def test_constant_in_span(self):
        """The unconstrained cr span contains the constant (why the
        sum-to-zero constraint is mandatory)."""
        x = np.linspace(0.0, 1.0, 50)
        B, _, _ = cr_basis(x, k=6)
        one = np.ones(50)
        coef, *_ = np.linalg.lstsq(B, one, rcond=None)
        assert np.allclose(B @ coef, one, atol=1e-10)

    def test_k_exceeding_unique_raises(self):
        x = np.repeat(np.linspace(0.0, 1.0, 5), 10)  # 5 unique, n=50
        with pytest.raises(ValidationError, match="unique"):
            cr_basis(x, k=10)

    def test_nonfinite_raises(self):
        x = np.linspace(0, 1, 20).copy()
        x[3] = np.nan
        with pytest.raises(ValidationError, match="non-finite"):
            cr_basis(x, k=5)


class TestTpBasis:
    def test_shape_and_null_space(self):
        x = np.sort(np.random.default_rng(1).uniform(0, 1, 80))
        B, S, s_scale = tp_basis(x, k=9)
        assert B.shape == (80, 9)
        assert S.shape == (9, 9)
        # penalty rank k-2, null space (constant, linear) unpenalized
        ev = np.linalg.eigvalsh(S)
        assert np.sum(ev > ev.max() * 1e-9) == 7

    def test_linear_function_in_span(self):
        """REGRESSION (4.5.x defect): tp must represent a straight line."""
        x = np.sort(np.random.default_rng(2).uniform(0, 1, 60))
        B, _, _ = tp_basis(x, k=8)
        target = 2.0 + 3.0 * x
        coef, *_ = np.linalg.lstsq(B, target, rcond=None)
        assert np.allclose(B @ coef, target, atol=1e-8)

    def test_columns_normalised(self):
        x = np.sort(np.random.default_rng(3).uniform(0, 1, 70))
        B, _, _ = tp_basis(x, k=6)
        norms = np.linalg.norm(B, axis=0)
        assert np.allclose(norms, np.sqrt(70), rtol=1e-10)


class TestConstraintAbsorption:
    def test_kills_constant_and_keeps_rank(self):
        x = np.linspace(0.0, 1.0, 50)
        B, S, _ = cr_basis(x, k=8)
        B_c, S_c, Z = absorb_sum_to_zero(B, S)
        assert B_c.shape == (50, 7)
        assert S_c.shape == (7, 7)
        # column sums are zero -> orthogonal to the intercept over the data
        assert np.allclose(B_c.sum(axis=0), 0.0, atol=1e-8)
        # design [1 | B_c] is full rank (THE 4.5.x root-cause fix)
        X_aug = np.hstack([np.ones((50, 1)), B_c])
        assert np.linalg.matrix_rank(X_aug) == 8


# ---------------------------------------------------------------------------
# SmoothTerm / s()
# ---------------------------------------------------------------------------

class TestSmoothTerm:
    def test_defaults(self):
        st = s("x")
        assert st.k == 10 and st.bs == "cr" and st.var_name == "x"

    def test_custom(self):
        st = s("age", k=15, bs="tp")
        assert (st.var_name, st.k, st.bs) == ("age", 15, "tp")

    def test_invalid_var_name(self):
        with pytest.raises(ValidationError):
            s("")

    def test_invalid_k(self):
        with pytest.raises(ValidationError):
            s("x", k=2)
        with pytest.raises(ValidationError):
            s("x", k=501)

    def test_unknown_basis_raises(self):
        with pytest.raises(ValidationError, match="bs must be"):
            s("x", bs="cc")   # cyclic basis not implemented: fail loud

    def test_is_pure_spec(self):
        """No fit-derived caches on the spec object (stale-state hazard)."""
        st = s("x")
        assert not hasattr(st, "basis_matrix")
        assert not hasattr(st, "penalty_matrix")


# ---------------------------------------------------------------------------
# Gaussian fitting — normal cases
# ---------------------------------------------------------------------------

class TestGaussianFit:
    def test_recovers_sine(self, sine_data):
        x, y, f = sine_data
        sol = gam(y, smooths=[s("x")], smooth_data={"x": x})
        assert isinstance(sol, GAMSolution)
        rmse = float(np.sqrt(np.mean((sol.fitted_values - f) ** 2)))
        assert rmse < 0.1
        assert sol.converged and sol.outer_converged

    def test_edf_sane(self, sine_data):
        """REGRESSION (4.5.x defect): EDF was garbage (even negative)."""
        x, y, _ = sine_data
        sol = gam(y, smooths=[s("x", k=20)], smooth_data={"x": x})
        assert 1.0 < sol.total_edf < 20.0
        assert np.all(np.asarray(sol.edf) > 0.0)
        # total = parametric (intercept=1) + smooth edf
        assert sol.total_edf == pytest.approx(1.0 + float(sol.edf[0]), abs=1e-8)

    def test_fixed_sp_reproducible(self, sine_data):
        x, y, _ = sine_data
        a = gam(y, smooths=[s("x")], smooth_data={"x": x}, sp=[1.5])
        b = gam(y, smooths=[s("x")], smooth_data={"x": x}, sp=[1.5])
        assert np.array_equal(a.coefficients, b.coefficients)
        assert a.lambdas[0] == 1.5

    def test_multiple_smooths(self, two_smooth_data):
        x1, x2, y = two_smooth_data
        sol = gam(y, smooths=[s("x1"), s("x2")],
                  smooth_data={"x1": x1, "x2": x2})
        assert len(sol.smooth_terms) == 2
        assert sol.edf.shape == (2,)
        assert sol.lambdas.shape == (2,)

    def test_parametric_plus_smooth(self, sine_data):
        x, y, _ = sine_data
        rng = np.random.default_rng(0)
        z = rng.normal(size=x.shape[0])
        y2 = y + 0.7 * z
        X = np.column_stack([np.ones_like(x), z])
        sol = gam(y2, X, smooths=[s("x")], smooth_data={"x": x},
                  names=["(Intercept)", "z"])
        # z coefficient recovered
        assert abs(sol.coefficients[1] - 0.7) < 0.1

    def test_smooth_coefficients_are_k_minus_1(self, sine_data):
        x, y, _ = sine_data
        sol = gam(y, smooths=[s("x", k=10)], smooth_data={"x": x})
        si = sol.smooth_terms[0]
        assert si.coef_indices[1] - si.coef_indices[0] == 9  # k-1, as mgcv
        assert sol.coefficients.shape[0] == 10  # intercept + 9

    def test_high_lambda_underfits(self, sine_data):
        x, y, _ = sine_data
        stiff = gam(y, smooths=[s("x")], smooth_data={"x": x}, sp=[1e9])
        loose = gam(y, smooths=[s("x")], smooth_data={"x": x}, sp=[1e-6])
        assert stiff.total_edf < loose.total_edf
        # lambda -> inf: smooth collapses toward the penalty null space
        # (linear, minus the constant absorbed by the constraint): edf -> ~1.
        assert float(stiff.edf[0]) == pytest.approx(1.0, abs=0.05)

    def test_standard_errors_are_real(self, sine_data):
        """REGRESSION (4.5.x defect): SEs were a placeholder sqrt(scale/n),
        identical for every coefficient."""
        x, y, _ = sine_data
        sol = gam(y, smooths=[s("x")], smooth_data={"x": x})
        se = sol.se
        assert se.shape == sol.coefficients.shape
        assert np.all(se > 0.0)
        assert np.std(se) > 1e-12  # not all identical
        # pure parametric fit: posterior SE == OLS SE closed form
        n = 100
        rng = np.random.default_rng(5)
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        yy = X @ np.array([1.0, 2.0]) + rng.normal(0, 0.5, n)
        solp = gam(yy, X)
        resid = yy - solp.fitted_values
        sigma2 = float(resid @ resid) / (n - 2)
        se_ols = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
        assert np.allclose(solp.se, se_ols, rtol=1e-8)

    def test_covariance_matches_se(self, sine_data):
        x, y, _ = sine_data
        sol = gam(y, smooths=[s("x")], smooth_data={"x": x})
        assert np.allclose(sol.se ** 2, np.diag(sol.covariance))


# ---------------------------------------------------------------------------
# Selection criteria
# ---------------------------------------------------------------------------

class TestSelection:
    def test_gcv_beats_fixed_neighbours(self, sine_data):
        """The selected lambda is a local minimum of the GCV curve."""
        x, y, _ = sine_data
        sol = gam(y, smooths=[s("x")], smooth_data={"x": x}, method="GCV")
        lam = float(sol.lambdas[0])
        for factor in (0.5, 2.0):
            other = gam(y, smooths=[s("x")], smooth_data={"x": x},
                        sp=[lam * factor])
            assert sol.gcv <= other.gcv + 1e-10

    def test_reml_gaussian(self, sine_data):
        x, y, _ = sine_data
        sol = gam(y, smooths=[s("x")], smooth_data={"x": x}, method="REML")
        assert sol.reml_score is not None and np.isfinite(sol.reml_score)
        lam = float(sol.lambdas[0])
        for factor in (0.5, 2.0):
            other = gam(y, smooths=[s("x")], smooth_data={"x": x},
                        sp=[lam * factor], method="REML")
            assert sol.reml_score <= other.reml_score + 1e-10

    def test_gcv_fit_has_no_reml_score(self, sine_data):
        x, y, _ = sine_data
        sol = gam(y, smooths=[s("x")], smooth_data={"x": x}, method="GCV")
        assert sol.reml_score is None

    def test_reml_unsupported_family_raises(self):
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0, 1, 100))
        y = rng.gamma(2.0, np.exp(0.5 * x))
        with pytest.raises(ValidationError, match="REML"):
            gam(y, smooths=[s("x")], smooth_data={"x": x},
                family="Gamma", method="REML")


# ---------------------------------------------------------------------------
# Non-Gaussian families
# ---------------------------------------------------------------------------

class TestFamilies:
    def test_poisson(self):
        rng = np.random.default_rng(4)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = rng.poisson(np.exp(1.2 + np.sin(3 * x))).astype(float)
        sol = gam(y, smooths=[s("x")], smooth_data={"x": x},
                  family="poisson")
        assert sol.converged
        assert sol.scale == 1.0
        assert 1.0 < sol.total_edf < 10.0

    def test_binomial(self):
        rng = np.random.default_rng(11)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        p = 1.0 / (1.0 + np.exp(-2.0 * np.sin(2 * np.pi * x)))
        y = rng.binomial(1, p).astype(float)
        sol = gam(y, smooths=[s("x")], smooth_data={"x": x},
                  family="binomial")
        assert sol.converged
        assert np.all((sol.fitted_values > 0) & (sol.fitted_values < 1))

    def test_binomial_separation_warns_not_crashes(self):
        """Complete separation: R-style warning + finite fit, no inf/NaN."""
        rng = np.random.default_rng(9)
        n = 60
        x = np.sort(rng.uniform(0, 1, n))
        y = (x > 0.5).astype(float)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            sol = gam(y, smooths=[s("x", k=8)], smooth_data={"x": x},
                      family="binomial")
        assert np.all(np.isfinite(sol.coefficients))
        assert any("numerically 0 or 1" in str(w.message) for w in rec)

    def test_quasipoisson_raises(self, sine_data):
        x, y, _ = sine_data
        with pytest.raises(ValidationError):
            gam(y, smooths=[s("x")], smooth_data={"x": x},
                family="quasipoisson")


# ---------------------------------------------------------------------------
# Failure / edge cases
# ---------------------------------------------------------------------------

class TestValidation:
    def test_no_smooths_no_X_raises(self):
        with pytest.raises(ValidationError):
            gam(np.array([1.0, 2.0, 3.0]))

    def test_missing_smooth_data(self, sine_data):
        x, y, _ = sine_data
        with pytest.raises(ValidationError, match="missing"):
            gam(y, smooths=[s("nope")], smooth_data={"x": x})

    def test_length_mismatch(self, sine_data):
        x, y, _ = sine_data
        with pytest.raises(ValidationError):
            gam(y, smooths=[s("x")], smooth_data={"x": x[:-5]})

    def test_invalid_method(self, sine_data):
        x, y, _ = sine_data
        with pytest.raises(ValidationError, match="method"):
            gam(y, smooths=[s("x")], smooth_data={"x": x}, method="AIC")

    def test_invalid_family(self, sine_data):
        x, y, _ = sine_data
        with pytest.raises(ValidationError):
            gam(y, smooths=[s("x")], smooth_data={"x": x}, family="beta")

    def test_nonfinite_y_raises(self, sine_data):
        x, y, _ = sine_data
        y = y.copy()
        y[0] = np.inf
        with pytest.raises(ValidationError, match="non-finite"):
            gam(y, smooths=[s("x")], smooth_data={"x": x})

    def test_sp_validation(self, sine_data):
        x, y, _ = sine_data
        with pytest.raises(ValidationError, match="sp"):
            gam(y, smooths=[s("x")], smooth_data={"x": x}, sp=[1.0, 2.0])
        with pytest.raises(ValidationError, match="sp"):
            gam(y, smooths=[s("x")], smooth_data={"x": x}, sp=[-1.0])

    def test_backend_parameter_removed(self, sine_data):
        """The GAM module is CPU-only; per library convention it exposes
        NO backend= parameter at all (removed in 4.6.0)."""
        x, y, _ = sine_data
        with pytest.raises(TypeError):
            gam(y, smooths=[s("x")], smooth_data={"x": x}, backend="gpu")

    def test_k_exceeding_unique_values_raises(self):
        rng = np.random.default_rng(1)
        x = np.repeat(np.linspace(0, 1, 6), 20)   # 6 unique, n=120
        y = rng.normal(size=120)
        with pytest.raises(ValidationError, match="unique"):
            gam(y, smooths=[s("x", k=10)], smooth_data={"x": x})

    def test_duplicated_smooth_variable_warns_rank(self, sine_data):
        """Perfect concurvity: the same variable smoothed twice."""
        x, y, _ = sine_data
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            sol = gam(
                y,
                smooths=[s("x", k=8), s("x", k=8)],
                smooth_data={"x": x},
            )
        assert any("rank deficient" in str(w.message) for w in rec)
        assert np.all(np.isfinite(sol.coefficients))


# ---------------------------------------------------------------------------
# tp end-to-end (regression: the 4.5.x tp could not fit a line)
# ---------------------------------------------------------------------------

class TestTpFit:
    def test_tp_recovers_linear_signal(self):
        rng = np.random.default_rng(3)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = 2.0 + 3.0 * x + rng.normal(0, 0.1, n)
        sol = gam(y, smooths=[s("x", k=10, bs="tp")], smooth_data={"x": x})
        rmse = float(np.sqrt(np.mean((sol.fitted_values - (2 + 3 * x)) ** 2)))
        assert rmse < 0.05
        assert sol.total_edf < 4.0  # near-linear fit, not flat, not wiggly

    def test_tp_recovers_sine(self, sine_data):
        x, y, f = sine_data
        sol = gam(y, smooths=[s("x", bs="tp")], smooth_data={"x": x})
        rmse = float(np.sqrt(np.mean((sol.fitted_values - f) ** 2)))
        assert rmse < 0.1


# ---------------------------------------------------------------------------
# Summary / solution surface
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_contents(self, sine_data):
        x, y, _ = sine_data
        sol = gam(y, smooths=[s("x")], smooth_data={"x": x})
        txt = sol.summary()
        for token in ("Family: gaussian", "s(x)", "GCV", "edf",
                      "Deviance explained"):
            assert token in txt

    def test_reml_summary_shows_reml(self, sine_data):
        x, y, _ = sine_data
        sol = gam(y, smooths=[s("x")], smooth_data={"x": x}, method="REML")
        assert "-REML" in sol.summary()

    def test_smooth_info_fields(self, sine_data):
        x, y, _ = sine_data
        sol = gam(y, smooths=[s("x", k=12)], smooth_data={"x": x})
        si = sol.smooth_terms[0]
        assert si.k == 12
        assert si.edf > 0 and si.ref_df >= si.edf - 1e-6
        assert 0.0 <= si.p_value <= 1.0
        assert si.lambda_ > 0 and si.s_scale > 0

    def test_params_frozen(self, sine_data):
        x, y, _ = sine_data
        sol = gam(y, smooths=[s("x")], smooth_data={"x": x})
        with pytest.raises(AttributeError):
            sol.params.scale = 1.0  # type: ignore[misc]

    def test_repr_not_huge(self, sine_data):
        x, y, _ = sine_data
        sol = gam(y, smooths=[s("x")], smooth_data={"x": x})
        assert len(repr(sol.params)) < 2000  # arrays are repr=False
