"""Tests for Generalized Linear Mixed Models (GLMM)."""

import numpy as np
import pytest

from pystatistics.mixed import glmm
from pystatistics.core.exceptions import ValidationError


class TestGLMMFamilyGuard:
    """glmm() must fail loud on free-dispersion families (Gaussian, Gamma):
    its Laplace likelihood fixes dispersion=1, so those families would yield a
    silently-wrong log-likelihood/AIC/variance. Gaussian users are sent to lmm().
    """

    @pytest.mark.parametrize("family", ["gaussian", "gamma"])
    def test_free_dispersion_family_raises(self, family):
        rng = np.random.default_rng(0)
        n = 80
        g = np.repeat(np.arange(8), 10)
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        y = np.abs(rng.normal(size=n)) + 0.1  # positive (valid for gamma too)
        with pytest.raises(ValidationError, match="dispersion"):
            glmm(y, X, groups={"g": g}, family=family)


class TestGLMMBinomial:
    """Tests for GLMM with binomial family."""

    def test_basic_fit(self, glmm_binomial):
        """Binomial GLMM fits and converges."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        assert result.converged
        assert len(result.coefficients) == 2

    def test_n_iter_exposed(self, glmm_binomial):
        """GLMMSolution exposes .n_iter (every iterative fit must, per the
        constitution's uniform-accessor rule)."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        assert isinstance(result.n_iter, int)
        assert result.n_iter >= 1

    def test_family_info(self, glmm_binomial):
        """Family and link recorded correctly."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        assert result.params.family_name == 'binomial'
        assert result.params.link_name == 'logit'

    def test_fixed_effects_direction(self, glmm_binomial):
        """Fixed effects should have correct sign."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        # True beta1 = 1.0 (positive)
        assert result.coefficients[1] > 0

    def test_variance_component(self, glmm_binomial):
        """Random intercept variance should be positive."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        assert len(result.var_components) == 1
        assert result.var_components[0].variance > 0

    def test_fitted_values_in_range(self, glmm_binomial):
        """Fitted values (probabilities) should be in [0, 1]."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        assert np.all(result.fitted_values >= 0)
        assert np.all(result.fitted_values <= 1)

    def test_deviance_positive(self, glmm_binomial):
        """Deviance should be positive."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        assert result.deviance > 0

    def test_wald_z_statistics(self, glmm_binomial):
        """GLMM uses Wald z-statistics, not t."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        # z = coef / se
        np.testing.assert_allclose(
            result.z_values, result.coefficients / result.standard_errors, atol=1e-10
        )

    def test_p_values_defined(self, glmm_binomial):
        """p-values should be between 0 and 1."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        for p in result.p_values:
            assert 0 <= p <= 1

    def test_summary_output(self, glmm_binomial):
        """Summary should mention family and link."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        s = result.summary()
        assert 'binomial' in s
        assert 'logit' in s

    def test_icc_logistic(self, glmm_binomial):
        """ICC on logistic scale: uses π²/3 for residual variance."""
        d = glmm_binomial
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='binomial',
        )
        icc = result.icc
        assert 'group' in icc
        assert 0 < icc['group'] < 1


class TestGLMMPoisson:
    """Tests for GLMM with Poisson family."""

    def test_basic_fit(self, glmm_poisson):
        """Poisson GLMM fits and converges."""
        d = glmm_poisson
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='poisson',
        )
        assert result.converged

    def test_family_info(self, glmm_poisson):
        """Family and link recorded correctly."""
        d = glmm_poisson
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='poisson',
        )
        assert result.params.family_name == 'poisson'
        assert result.params.link_name == 'log'

    def test_fixed_effects_direction(self, glmm_poisson):
        """Fixed effects should have correct sign."""
        d = glmm_poisson
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='poisson',
        )
        # True beta1 = 0.5 (positive)
        assert result.coefficients[1] > 0

    def test_fitted_values_positive(self, glmm_poisson):
        """Poisson fitted values (counts) should be positive."""
        d = glmm_poisson
        result = glmm(
            d['y'], d['X'],
            groups={'group': d['group']},
            family='poisson',
        )
        assert np.all(result.fitted_values > 0)


class TestGLMMOptimizerRobustness:
    """Regression guards for the Laplace optimizer's robustness.

    (1) Poisson PIRLS must not overflow when the outer optimizer probes a
        far-from-optimum β (step-halving + η-clamping in the inner loop).
    (2) The outer optimizer must not silently collapse the random-effect
        variance to ~0 at a suboptimal point — a prior L-BFGS-B overshoot to the
        θ=0 boundary that reported convergence. A derivative-free fallback
        rescues it. References are from lme4::glmer(..., nAGQ=1).
    """

    def test_poisson_wide_covariate_converges(self):
        rng = np.random.default_rng(3)
        G, per = 20, 20
        n = G * per
        g = np.repeat(np.arange(G), per)
        b = rng.normal(0, 0.5, G)
        x = rng.normal(0, 1.5, n)  # wide range -> exp(eta) would overflow raw
        y = rng.poisson(np.exp(0.5 + 0.4 * x + b[g])).astype(float)
        X = np.column_stack([np.ones(n), x])
        res = glmm(y, X, groups={"g": g}, family="poisson")
        assert res.converged
        assert np.all(np.isfinite(res.coefficients))
        assert res.coefficients[1] == pytest.approx(0.4, abs=0.1)
        assert res.var_components[0].variance > 0.05

    def test_flat_surface_variance_not_collapsed(self):
        """A logit random-intercept fit whose deviance surface has a shallow
        interior variance optimum: L-BFGS-B overshoots to θ=0 and the fine
        stationarity probe misses it through PIRLS noise, so the boundary must be
        treated as suspect and rescued by the derivative-free fallback.
        lme4::glmer(nAGQ=1) gives variance ≈ 0.174 on this exact data."""
        rng = np.random.default_rng(404)
        G, per = 25, 16
        n = G * per
        g = np.repeat(np.arange(G), per)
        b = rng.normal(0, 0.7, G)
        x = rng.normal(0, 1, n)
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(0.3 + 0.6 * x + b[g])))).astype(float)
        X = np.column_stack([np.ones(n), x])
        res = glmm(y, X, groups={"g": g}, family="binomial")
        assert res.converged
        np.testing.assert_allclose(res.var_components[0].variance, 0.174, rtol=0.1)

    def test_genuine_singular_stays_zero(self):
        """The other direction (R9/R12 true-classifier): when there is NO group
        signal the boundary rescue must NOT invent spurious variance — the
        fallback is adopted only if it strictly lowers the deviance, so a
        genuinely singular fit stays at ~0 (matching glmer isSingular)."""
        rng = np.random.default_rng(202)
        G, per = 20, 15
        n = G * per
        g = np.repeat(np.arange(G), per)
        x = rng.normal(0, 1, n)
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(0.1 + 0.8 * x)))).astype(float)
        X = np.column_stack([np.ones(n), x])
        res = glmm(y, X, groups={"g": g}, family="binomial")
        assert res.var_components[0].variance < 0.02

    def test_probit_variance_not_collapsed(self):
        from scipy.stats import norm
        from pystatistics.regression.families import Binomial
        rng = np.random.default_rng(7)
        G, per = 25, 18
        n = G * per
        g = np.repeat(np.arange(G), per)
        b = rng.normal(0, 0.6, G)
        x = rng.normal(0, 1, n)
        y = (rng.random(n) < norm.cdf(0.2 + 0.5 * x + b[g])).astype(float)
        X = np.column_stack([np.ones(n), x])
        res = glmm(y, X, groups={"g": g}, family=Binomial(link="probit"))
        # lme4::glmer(nAGQ=1): coef (0.0997, 0.5441), var 0.1384.
        assert res.converged
        np.testing.assert_allclose(res.coefficients, [0.0997, 0.5441], rtol=0.05)
        # The collapse bug drove this to ~1e-10; guard well above that.
        assert 0.08 < res.var_components[0].variance < 0.25


class TestGLMMCorrelatedFixedEffectSE:
    """Regression guard: fixed-effect SEs must be correct for CORRELATED
    predictors, not only (weighted-)orthogonal ones.

    A prior bug computed Var(β̂) as (RXᵀRX)⁻¹ instead of the correct
    (RX·RXᵀ)⁻¹. The two agree when XᵀWX is near-diagonal, so single-predictor
    fixtures masked it — but with correlated predictors the slope SEs were off
    by nearly an order of magnitude (silently wrong z/p/CIs). Reference values
    are from lme4::glmer(y ~ x1 + x2 + (1|g), family=binomial, nAGQ=1) on this
    exact (seeded) dataset.
    """

    # lme4::glmer nAGQ=1 reference (see docstring).
    R_COEF = np.array([-0.030256, 0.326583, -0.172029])
    R_SE = np.array([0.136113, 0.629364, 0.722047])

    @staticmethod
    def _make_data():
        rng = np.random.default_rng(2024)
        G, per = 30, 12
        n = G * per
        grp = np.repeat(np.arange(G), per)
        b = rng.normal(0, 0.7, G)
        x1 = rng.normal(0, 1, n)
        x2 = 0.85 * x1 + 0.15 * rng.normal(0, 1, n)  # corr(x1, x2) ~ 0.98
        eta = 0.1 + 0.5 * x1 - 0.4 * x2 + b[grp]
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
        X = np.column_stack([np.ones(n), x1, x2])
        return y, X, grp

    def test_se_matches_glmer(self):
        """SEs agree with glmer to the Laplace/optimizer tier (~5%)."""
        y, X, grp = self._make_data()
        result = glmm(y, X, groups={'g': grp}, family='binomial')
        np.testing.assert_allclose(result.coefficients, self.R_COEF, rtol=0.05)
        np.testing.assert_allclose(result.standard_errors, self.R_SE, rtol=0.05)

    def test_se_not_collapsed(self):
        """Bulletproof sentinel: the buggy formula drove the correlated-slope
        SEs below ~0.1; the correct ones are ~0.6-0.7. Guards against a
        transpose regression even if the reference constants drift."""
        y, X, grp = self._make_data()
        result = glmm(y, X, groups={'g': grp}, family='binomial')
        # slopes on x1, x2 (indices 1, 2)
        assert result.standard_errors[1] > 0.3
        assert result.standard_errors[2] > 0.3
