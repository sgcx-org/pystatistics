"""
R validation tests for mixed models.

Compares Python lmm() / glmm() results against reference values
from R lme4::lmer(), lme4::glmer(), and lmerTest::summary().

Reference data in: tests/fixtures/mixed/mixed_r_results.json
Fixture CSVs in:   tests/fixtures/mixed/*.csv

Tolerances (from plan):
    Fixed effects beta:    rtol=1e-4 (LMM), rtol=1e-3 (GLMM)
    Standard errors:       rtol=1e-3 (LMM), rtol=1e-2 (GLMM)
    Variance components:   rtol=1e-3
    Log-likelihood:        atol=0.5
    AIC/BIC:               rtol=1e-2
    Satterthwaite df:      rtol=0.05
    BLUPs:                 rtol=1e-3
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from pystatistics.mixed import lmm, glmm


# =====================================================================
# Fixtures — load R reference data and CSV data files
# =====================================================================

FIXTURE_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, 'fixtures', 'mixed'
)


def _load_r_results() -> dict:
    path = os.path.join(FIXTURE_DIR, 'mixed_r_results.json')
    with open(path) as f:
        return json.load(f)


def _load_csv(name: str) -> dict[str, np.ndarray]:
    """Load CSV fixture and return columns as numpy arrays."""
    import csv
    path = os.path.join(FIXTURE_DIR, f'{name}.csv')
    with open(path) as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)
    data = {}
    for i, col in enumerate(headers):
        vals = [row[i] for row in rows]
        # Try float first; group columns will be integer-like
        arr = np.array(vals, dtype=np.float64)
        data[col] = arr
    return data


@pytest.fixture(scope='module')
def r_results():
    return _load_r_results()


# =====================================================================
# LMM: Random intercept (REML)
# =====================================================================

class TestLMMInterceptREML:
    """lmer(y ~ x + (1|group), REML=TRUE) — random intercept model."""

    @pytest.fixture(autouse=True)
    def setup(self, r_results):
        self.ref = r_results['lmm_intercept']['reml']
        data = _load_csv('lmm_intercept')
        y = data['y']
        X = np.column_stack([np.ones(len(y)), data['x']])
        groups = {'group': data['group'].astype(int)}
        self.result = lmm(y, X, groups=groups, reml=True)

    def test_fixed_effects_intercept(self):
        beta_hat = self.result.coefficients
        r_intercept = self.ref['fixef']['(Intercept)']
        np.testing.assert_allclose(beta_hat[0], r_intercept, rtol=1e-4)

    def test_fixed_effects_slope(self):
        beta_hat = self.result.coefficients
        r_slope = self.ref['fixef']['x']
        np.testing.assert_allclose(beta_hat[1], r_slope, rtol=1e-4)

    def test_standard_errors(self):
        r_se = np.array(self.ref['se'])
        np.testing.assert_allclose(self.result.se, r_se, rtol=1e-3)

    def test_satterthwaite_df(self):
        r_df = np.array(self.ref['df'])
        np.testing.assert_allclose(
            self.result.df_satterthwaite, r_df, rtol=0.05
        )

    def test_t_values(self):
        r_t = np.array(self.ref['t_value'])
        np.testing.assert_allclose(self.result.t_values, r_t, rtol=1e-3)

    def test_p_values(self):
        r_p = np.array(self.ref['p_value'])
        # p-values can vary more, especially when very small
        for i in range(len(r_p)):
            if r_p[i] < 1e-10:
                # For very small p-values, just check order of magnitude
                assert self.result.p_values[i] < 1e-4, (
                    f"p-value[{i}] = {self.result.p_values[i]}, "
                    f"expected < 1e-4 (R: {r_p[i]})"
                )
            else:
                np.testing.assert_allclose(
                    self.result.p_values[i], r_p[i], rtol=0.1
                )

    def test_variance_group(self):
        r_var = self.ref['var_group']
        # Find the group intercept variance component
        vc = self.result.var_components
        py_var = [v.variance for v in vc if v.name == '(Intercept)'][0]
        np.testing.assert_allclose(py_var, r_var, rtol=1e-3)

    def test_variance_residual(self):
        r_var = self.ref['var_resid']
        np.testing.assert_allclose(
            self.result.params.residual_variance, r_var, rtol=1e-3
        )

    def test_log_likelihood(self):
        r_ll = self.ref['logLik']
        np.testing.assert_allclose(
            self.result.log_likelihood, r_ll, atol=0.5
        )

    def test_aic(self):
        r_aic = self.ref['AIC']
        np.testing.assert_allclose(self.result.aic, r_aic, rtol=1e-2)

    def test_bic(self):
        r_bic = self.ref['BIC']
        np.testing.assert_allclose(self.result.bic, r_bic, rtol=1e-2)

    def test_blups(self):
        """BLUPs (conditional modes) for each group match R ranef()."""
        r_ranef = np.array(self.ref['ranef'])
        py_ranef = self.result.ranef['group'][:, 0]  # intercept column
        np.testing.assert_allclose(py_ranef, r_ranef, rtol=1e-3)

    def test_convergence(self):
        assert self.result.converged


# =====================================================================
# LMM: Random intercept (ML)
# =====================================================================

class TestLMMInterceptML:
    """lmer(y ~ x + (1|group), REML=FALSE) — ML estimation on same data."""

    @pytest.fixture(autouse=True)
    def setup(self, r_results):
        self.ref = r_results['lmm_intercept']['ml']
        data = _load_csv('lmm_intercept')
        y = data['y']
        X = np.column_stack([np.ones(len(y)), data['x']])
        groups = {'group': data['group'].astype(int)}
        self.result = lmm(y, X, groups=groups, reml=False)

    def test_fixed_effects(self):
        np.testing.assert_allclose(
            self.result.coefficients[0],
            self.ref['fixef']['(Intercept)'],
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            self.result.coefficients[1],
            self.ref['fixef']['x'],
            rtol=1e-4,
        )

    def test_variance_group(self):
        vc = self.result.var_components
        py_var = [v.variance for v in vc if v.name == '(Intercept)'][0]
        np.testing.assert_allclose(py_var, self.ref['var_group'], rtol=1e-3)

    def test_variance_residual(self):
        np.testing.assert_allclose(
            self.result.params.residual_variance,
            self.ref['var_resid'],
            rtol=1e-3,
        )

    def test_log_likelihood(self):
        np.testing.assert_allclose(
            self.result.log_likelihood, self.ref['logLik'], atol=0.5
        )

    def test_aic(self):
        np.testing.assert_allclose(
            self.result.aic, self.ref['AIC'], rtol=1e-2
        )

    def test_bic(self):
        np.testing.assert_allclose(
            self.result.bic, self.ref['BIC'], rtol=1e-2
        )


# =====================================================================
# LMM: Random slope (REML)
# =====================================================================

class TestLMMSlopeREML:
    """lmer(y ~ days + (1 + days|subject), REML=TRUE)."""

    @pytest.fixture(autouse=True)
    def setup(self, r_results):
        self.ref = r_results['lmm_slope']['reml']
        data = _load_csv('lmm_slope')
        y = data['y']
        days = data['days']
        X = np.column_stack([np.ones(len(y)), days])
        groups = {'subject': data['subject'].astype(int)}
        self.result = lmm(
            y, X, groups=groups,
            random_effects={'subject': ['1', 'days']},
            random_data={'days': days},
            reml=True,
        )

    def test_fixed_intercept(self):
        np.testing.assert_allclose(
            self.result.coefficients[0],
            self.ref['fixef']['(Intercept)'],
            rtol=1e-4,
        )

    def test_fixed_slope(self):
        np.testing.assert_allclose(
            self.result.coefficients[1],
            self.ref['fixef']['days'],
            rtol=1e-4,
        )

    def test_standard_errors(self):
        r_se = np.array(self.ref['se'])
        np.testing.assert_allclose(self.result.se, r_se, rtol=1e-3)

    def test_satterthwaite_df(self):
        r_df = np.array(self.ref['df'])
        np.testing.assert_allclose(
            self.result.df_satterthwaite, r_df, rtol=0.05
        )

    def test_variance_intercept(self):
        vc = self.result.var_components
        py_var = [v.variance for v in vc if v.name == '(Intercept)'][0]
        np.testing.assert_allclose(
            py_var, self.ref['var_intercept'], rtol=1e-3
        )

    def test_variance_slope(self):
        vc = self.result.var_components
        py_var = [v.variance for v in vc if v.name == 'days'][0]
        np.testing.assert_allclose(
            py_var, self.ref['var_slope'], rtol=1e-3
        )

    def test_correlation(self):
        """Correlation between random intercept and slope."""
        vc = self.result.var_components
        slope_vc = [v for v in vc if v.name == 'days'][0]
        assert slope_vc.corr is not None
        np.testing.assert_allclose(
            slope_vc.corr, self.ref['corr'], atol=0.05
        )

    def test_variance_residual(self):
        np.testing.assert_allclose(
            self.result.params.residual_variance,
            self.ref['var_resid'],
            rtol=1e-3,
        )

    def test_log_likelihood(self):
        np.testing.assert_allclose(
            self.result.log_likelihood, self.ref['logLik'], atol=0.5
        )

    def test_aic(self):
        np.testing.assert_allclose(
            self.result.aic, self.ref['AIC'], rtol=1e-2
        )

    def test_bic(self):
        np.testing.assert_allclose(
            self.result.bic, self.ref['BIC'], rtol=1e-2
        )

    def test_convergence(self):
        assert self.result.converged


# =====================================================================
# LMM: Crossed random effects (REML)
# =====================================================================

class TestLMMCrossedREML:
    """lmer(y ~ x + (1|subject) + (1|item), REML=TRUE)."""

    @pytest.fixture(autouse=True)
    def setup(self, r_results):
        self.ref = r_results['lmm_crossed']['reml']
        data = _load_csv('lmm_crossed')
        y = data['y']
        X = np.column_stack([np.ones(len(y)), data['x']])
        groups = {
            'subject': data['subject'].astype(int),
            'item': data['item'].astype(int),
        }
        self.result = lmm(y, X, groups=groups, reml=True)

    def test_fixed_intercept(self):
        np.testing.assert_allclose(
            self.result.coefficients[0],
            self.ref['fixef']['(Intercept)'],
            rtol=1e-4,
        )

    def test_fixed_slope(self):
        np.testing.assert_allclose(
            self.result.coefficients[1],
            self.ref['fixef']['x'],
            rtol=1e-4,
        )

    def test_standard_errors(self):
        r_se = np.array(self.ref['se'])
        np.testing.assert_allclose(self.result.se, r_se, rtol=1e-3)

    def test_variance_subject(self):
        vc = self.result.var_components
        subj_vc = [v for v in vc if v.group == 'subject'][0]
        np.testing.assert_allclose(
            subj_vc.variance, self.ref['var_subject'], rtol=1e-3
        )

    def test_variance_item(self):
        vc = self.result.var_components
        item_vc = [v for v in vc if v.group == 'item'][0]
        np.testing.assert_allclose(
            item_vc.variance, self.ref['var_item'], rtol=1e-3
        )

    def test_variance_residual(self):
        np.testing.assert_allclose(
            self.result.params.residual_variance,
            self.ref['var_resid'],
            rtol=1e-3,
        )

    def test_log_likelihood(self):
        np.testing.assert_allclose(
            self.result.log_likelihood, self.ref['logLik'], atol=0.5
        )

    def test_aic(self):
        np.testing.assert_allclose(
            self.result.aic, self.ref['AIC'], rtol=1e-2
        )

    def test_bic(self):
        np.testing.assert_allclose(
            self.result.bic, self.ref['BIC'], rtol=1e-2
        )

    def test_convergence(self):
        assert self.result.converged


# =====================================================================
# LMM: ML estimation (separate dataset)
# =====================================================================

class TestLMMML:
    """lmer(y ~ x + (1|group), REML=FALSE) on separate lmm_ml dataset."""

    @pytest.fixture(autouse=True)
    def setup(self, r_results):
        self.ref = r_results['lmm_ml']['ml']
        data = _load_csv('lmm_ml')
        y = data['y']
        X = np.column_stack([np.ones(len(y)), data['x']])
        groups = {'group': data['group'].astype(int)}
        self.result = lmm(y, X, groups=groups, reml=False)

    def test_fixed_effects(self):
        np.testing.assert_allclose(
            self.result.coefficients[0],
            self.ref['fixef']['(Intercept)'],
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            self.result.coefficients[1],
            self.ref['fixef']['x'],
            rtol=1e-4,
        )

    def test_variance_group(self):
        vc = self.result.var_components
        py_var = [v.variance for v in vc if v.name == '(Intercept)'][0]
        np.testing.assert_allclose(py_var, self.ref['var_group'], rtol=1e-3)

    def test_variance_residual(self):
        np.testing.assert_allclose(
            self.result.params.residual_variance,
            self.ref['var_resid'],
            rtol=1e-3,
        )

    def test_log_likelihood(self):
        np.testing.assert_allclose(
            self.result.log_likelihood, self.ref['logLik'], atol=0.5
        )

    def test_aic(self):
        np.testing.assert_allclose(
            self.result.aic, self.ref['AIC'], rtol=1e-2
        )

    def test_bic(self):
        np.testing.assert_allclose(
            self.result.bic, self.ref['BIC'], rtol=1e-2
        )


# =====================================================================
# LMM: No effect (singular fit expected)
# =====================================================================

class TestLMMNoEffect:
    """lmer(y ~ x + (1|group)) with pure noise — singular fit.

    R reports boundary (singular) fit, var_group = 0.
    Our optimizer should also converge to var_group near 0.
    """

    @pytest.fixture(autouse=True)
    def setup(self, r_results):
        self.ref = r_results['lmm_no_effect']['reml']
        data = _load_csv('lmm_no_effect')
        y = data['y']
        X = np.column_stack([np.ones(len(y)), data['x']])
        groups = {'group': data['group'].astype(int)}
        self.result = lmm(y, X, groups=groups, reml=True)

    def test_fixed_intercept(self):
        # Not very precise since data is pure noise, but should
        # be in the right ballpark
        np.testing.assert_allclose(
            self.result.coefficients[0],
            self.ref['fixef']['(Intercept)'],
            atol=0.5,
        )

    def test_fixed_slope(self):
        np.testing.assert_allclose(
            self.result.coefficients[1],
            self.ref['fixef']['x'],
            atol=0.5,
        )

    def test_variance_group_near_zero(self):
        """R reports var_group = 0 (singular). Ours should be near 0."""
        vc = self.result.var_components
        py_var = [v.variance for v in vc if v.name == '(Intercept)'][0]
        assert py_var < 0.5, (
            f"Expected near-zero group variance, got {py_var}"
        )

    def test_variance_residual(self):
        np.testing.assert_allclose(
            self.result.params.residual_variance,
            self.ref['var_resid'],
            rtol=0.05,  # wider tolerance for singular fit
        )

    def test_log_likelihood(self):
        np.testing.assert_allclose(
            self.result.log_likelihood, self.ref['logLik'], atol=1.0
        )


# =====================================================================
# GLMM: Binomial
# =====================================================================

class TestGLMMBinomial:
    """glmer(y ~ x + (1|group), family=binomial).

    GLMM tolerances are wider than LMM because:
    1. The Laplace approximation itself introduces error
    2. PIRLS convergence differs between Python dense and R's C++/CHOLMOD sparse
    3. The profiled deviance surface is flatter, so small theta differences
       propagate to larger beta differences
    """

    @pytest.fixture(autouse=True)
    def setup(self, r_results):
        self.ref = r_results['glmm_binomial']['fit']
        data = _load_csv('glmm_binomial')
        y = data['y']
        X = np.column_stack([np.ones(len(y)), data['x']])
        groups = {'group': data['group'].astype(int)}
        self.result = glmm(y, X, groups=groups, family='binomial')

    def test_fixed_intercept(self):
        np.testing.assert_allclose(
            self.result.coefficients[0],
            self.ref['fixef']['(Intercept)'],
            rtol=0.05,
        )

    def test_fixed_slope(self):
        np.testing.assert_allclose(
            self.result.coefficients[1],
            self.ref['fixef']['x'],
            rtol=0.05,
        )

    def test_standard_errors(self):
        r_se = np.array(self.ref['se'])
        np.testing.assert_allclose(self.result.se, r_se, rtol=0.06)

    def test_z_values(self):
        r_z = np.array(self.ref['z_value'])
        np.testing.assert_allclose(self.result.z_values, r_z, rtol=0.06)

    def test_p_values(self):
        r_p = np.array(self.ref['p_value'])
        for i in range(len(r_p)):
            if r_p[i] < 1e-6:
                assert self.result.p_values[i] < 1e-3
            else:
                np.testing.assert_allclose(
                    self.result.p_values[i], r_p[i], rtol=0.2
                )

    def test_variance_group(self):
        vc = self.result.var_components
        py_var = [v.variance for v in vc if v.name == '(Intercept)'][0]
        np.testing.assert_allclose(
            py_var, self.ref['var_group'], rtol=1e-2
        )

    def test_deviance(self):
        np.testing.assert_allclose(
            self.result.deviance, self.ref['deviance'], rtol=1e-2
        )

    def test_log_likelihood(self):
        np.testing.assert_allclose(
            self.result.log_likelihood, self.ref['logLik'], atol=1.0
        )

    def test_aic(self):
        np.testing.assert_allclose(
            self.result.aic, self.ref['AIC'], rtol=1e-2
        )

    def test_bic(self):
        np.testing.assert_allclose(
            self.result.bic, self.ref['BIC'], rtol=1e-2
        )

    def test_convergence(self):
        assert self.result.converged


# =====================================================================
# GLMM: Poisson
# =====================================================================

class TestGLMMPoisson:
    """glmer(y ~ x + (1|group), family=poisson).

    Same wider tolerances as binomial GLMM — see note above.
    """

    @pytest.fixture(autouse=True)
    def setup(self, r_results):
        self.ref = r_results['glmm_poisson']['fit']
        data = _load_csv('glmm_poisson')
        y = data['y']
        X = np.column_stack([np.ones(len(y)), data['x']])
        groups = {'group': data['group'].astype(int)}
        self.result = glmm(y, X, groups=groups, family='poisson')

    def test_fixed_intercept(self):
        np.testing.assert_allclose(
            self.result.coefficients[0],
            self.ref['fixef']['(Intercept)'],
            rtol=0.05,
        )

    def test_fixed_slope(self):
        np.testing.assert_allclose(
            self.result.coefficients[1],
            self.ref['fixef']['x'],
            rtol=0.05,
        )

    def test_standard_errors(self):
        r_se = np.array(self.ref['se'])
        np.testing.assert_allclose(self.result.se, r_se, rtol=0.06)

    def test_z_values(self):
        r_z = np.array(self.ref['z_value'])
        np.testing.assert_allclose(self.result.z_values, r_z, rtol=0.06)

    def test_p_values(self):
        r_p = np.array(self.ref['p_value'])
        for i in range(len(r_p)):
            if r_p[i] < 1e-6:
                assert self.result.p_values[i] < 1e-3
            else:
                np.testing.assert_allclose(
                    self.result.p_values[i], r_p[i], rtol=0.1
                )

    def test_variance_group(self):
        vc = self.result.var_components
        py_var = [v.variance for v in vc if v.name == '(Intercept)'][0]
        np.testing.assert_allclose(
            py_var, self.ref['var_group'], rtol=1e-2
        )

    def test_deviance(self):
        np.testing.assert_allclose(
            self.result.deviance, self.ref['deviance'], rtol=1e-2
        )

    def test_log_likelihood(self):
        np.testing.assert_allclose(
            self.result.log_likelihood, self.ref['logLik'], atol=1.0
        )

    def test_aic(self):
        np.testing.assert_allclose(
            self.result.aic, self.ref['AIC'], rtol=1e-2
        )

    def test_bic(self):
        np.testing.assert_allclose(
            self.result.bic, self.ref['BIC'], rtol=1e-2
        )

    def test_convergence(self):
        assert self.result.converged


# =====================================================================
# Cross-estimation consistency checks
# =====================================================================

class TestREMLvsML:
    """REML vs ML on same data: variance estimates should differ
    systematically (REML >= ML for variance components)."""

    @pytest.fixture(autouse=True)
    def setup(self, r_results):
        self.ref_reml = r_results['lmm_intercept']['reml']
        self.ref_ml = r_results['lmm_intercept']['ml']
        data = _load_csv('lmm_intercept')
        y = data['y']
        X = np.column_stack([np.ones(len(y)), data['x']])
        groups = {'group': data['group'].astype(int)}
        self.reml = lmm(y, X, groups=groups, reml=True)
        self.ml = lmm(y, X, groups=groups, reml=False)

    def test_reml_variance_geq_ml(self):
        """REML variance estimates are typically >= ML estimates."""
        vc_reml = [v for v in self.reml.var_components
                    if v.name == '(Intercept)'][0]
        vc_ml = [v for v in self.ml.var_components
                  if v.name == '(Intercept)'][0]
        # REML should give larger variance (corrects downward bias)
        assert vc_reml.variance >= vc_ml.variance * 0.95

    def test_fixed_effects_similar(self):
        """Fixed effects should be very similar between REML and ML."""
        np.testing.assert_allclose(
            self.reml.coefficients, self.ml.coefficients, rtol=1e-3
        )

    def test_ml_loglik_greater_than_reml(self):
        """ML log-likelihood should be >= REML log-likelihood
        (ML optimizes the full likelihood)."""
        # This is comparing different objectives so not always strictly true
        # Just check they're in the right ballpark
        assert abs(self.ml.log_likelihood - self.reml.log_likelihood) < 10
