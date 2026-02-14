"""
Validate PyStatistics GLM against R reference results.

Each GLM fixture CSV + R results JSON pair becomes a parametrized test.
Tests are skipped if no GLM R fixtures are found.

Tolerances:
    - Well-conditioned cases: rtol=1e-8
    - Near-separated binomial: rtol=1e-4 (IRLS sensitive to FP differences)
"""

import json
import pytest
import numpy as np
from pathlib import Path
from functools import lru_cache

from pystatistics import DataSource
from pystatistics.regression import Design, fit, GLMSolution

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _discover_glm_fixtures():
    """Find all GLM fixture names that have both CSV and R results."""
    fixtures = []
    for r_file in sorted(FIXTURES_DIR.glob("glm_*_r_results.json")):
        name = r_file.stem.replace("_r_results", "")
        csv_file = FIXTURES_DIR / f"{name}.csv"
        meta_file = FIXTURES_DIR / f"{name}_meta.json"
        if csv_file.exists() and meta_file.exists():
            fixtures.append(name)
    return fixtures


GLM_FIXTURE_NAMES = _discover_glm_fixtures()


@lru_cache(maxsize=None)
def _load_glm_fixture(name):
    """Load CSV data, R results, and metadata. Cached."""
    csv_path = FIXTURES_DIR / f"{name}.csv"
    r_path = FIXTURES_DIR / f"{name}_r_results.json"
    meta_path = FIXTURES_DIR / f"{name}_meta.json"

    with open(r_path) as f:
        r_results = json.load(f)

    with open(meta_path) as f:
        meta = json.load(f)

    family = meta['family']

    ds = DataSource.from_file(csv_path)
    x_cols = [c for c in ds.metadata['columns'] if c != 'y']
    design = Design.from_datasource(ds, x=x_cols, y='y')
    result = fit(design, family=family, backend='cpu')

    return result, r_results, meta


def _get_rtol(name):
    """Get relative tolerance for a fixture.

    Near-separated binomial needs wider tolerance.
    Default 1e-7 gives comfortable margin above machine epsilon.
    """
    if 'separated' in name:
        return 1e-4
    # Default: 1e-7 validates correctness to 7 significant digits
    return 1e-7


def _get_se_rtol(name):
    """Get tolerance for standard errors and derived quantities.

    Standard errors are computed from the QR R-matrix of the final
    IRLS iteration. Small differences in IRLS convergence paths can
    cause larger SE discrepancies than in coefficients, especially
    for Poisson data near boundaries (zeros, near-separation).
    """
    if 'separated' in name:
        return 1e-3
    if 'poisson_zeros' in name:
        # IRLS converges along slightly different path near μ≈0;
        # this shifts the final weights and QR by ~1e-5 in SE
        return 1e-4
    return 1e-7


@pytest.mark.parametrize("fixture_name", GLM_FIXTURE_NAMES)
class TestGLMRValidation:
    """Validate GLM results against R's glm.fit()."""

    def test_converged(self, fixture_name):
        result, r_results, _ = _load_glm_fixture(fixture_name)
        assert result.converged == r_results['converged'], (
            f"Python converged={result.converged} but R converged={r_results['converged']}"
        )

    def test_coefficients(self, fixture_name):
        result, r_results, _ = _load_glm_fixture(fixture_name)
        rtol = _get_rtol(fixture_name)
        r_coef = np.array(r_results['coefficients'])
        np.testing.assert_allclose(
            result.coefficients, r_coef, rtol=rtol,
            err_msg=f"Coefficients differ from R ({fixture_name})"
        )

    def test_deviance(self, fixture_name):
        result, r_results, _ = _load_glm_fixture(fixture_name)
        rtol = _get_rtol(fixture_name)
        np.testing.assert_allclose(
            result.deviance, r_results['deviance'], rtol=rtol,
            err_msg=f"Deviance differs from R ({fixture_name})"
        )

    def test_null_deviance(self, fixture_name):
        result, r_results, _ = _load_glm_fixture(fixture_name)
        rtol = _get_rtol(fixture_name)
        np.testing.assert_allclose(
            result.null_deviance, r_results['null_deviance'], rtol=rtol,
            err_msg=f"Null deviance differs from R ({fixture_name})"
        )

    def test_aic(self, fixture_name):
        result, r_results, _ = _load_glm_fixture(fixture_name)
        rtol = _get_rtol(fixture_name)
        np.testing.assert_allclose(
            result.aic, r_results['aic'], rtol=rtol,
            err_msg=f"AIC differs from R ({fixture_name})"
        )

    def test_standard_errors(self, fixture_name):
        result, r_results, _ = _load_glm_fixture(fixture_name)
        rtol = _get_se_rtol(fixture_name)
        r_se = np.array(r_results['standard_errors'])
        np.testing.assert_allclose(
            result.standard_errors, r_se, rtol=rtol,
            err_msg=f"Standard errors differ from R ({fixture_name})"
        )

    def test_test_statistics(self, fixture_name):
        result, r_results, _ = _load_glm_fixture(fixture_name)
        rtol = _get_se_rtol(fixture_name)
        r_stats = np.array(r_results['test_statistics'])
        np.testing.assert_allclose(
            result.test_statistics, r_stats, rtol=rtol,
            err_msg=f"Test statistics differ from R ({fixture_name})"
        )

    def test_p_values(self, fixture_name):
        result, r_results, _ = _load_glm_fixture(fixture_name)
        # p-values need wider tolerance because the CDF amplifies
        # small differences in the test statistic, especially in
        # the extreme tails (very small p-values). SE differences
        # of ~1e-5 can cascade to p-value differences of ~1e-3
        # via the normal/t CDF. Use 10x the SE tolerance as baseline.
        rtol = max(10 * _get_se_rtol(fixture_name), 1e-5)
        r_pv = np.array(r_results['p_values'])
        np.testing.assert_allclose(
            result.p_values, r_pv, rtol=rtol,
            err_msg=f"P-values differ from R ({fixture_name})"
        )

    def test_fitted_values(self, fixture_name):
        result, r_results, _ = _load_glm_fixture(fixture_name)
        rtol = _get_rtol(fixture_name)
        r_fitted = np.array(r_results['fitted_values'])
        np.testing.assert_allclose(
            result.fitted_values, r_fitted, rtol=rtol,
            err_msg=f"Fitted values differ from R ({fixture_name})"
        )

    def test_linear_predictor(self, fixture_name):
        result, r_results, _ = _load_glm_fixture(fixture_name)
        rtol = _get_rtol(fixture_name)
        r_lp = np.array(r_results['linear_predictor'])
        np.testing.assert_allclose(
            result.linear_predictor, r_lp, rtol=rtol,
            err_msg=f"Linear predictor differs from R ({fixture_name})"
        )

    def test_residuals_deviance(self, fixture_name):
        result, r_results, _ = _load_glm_fixture(fixture_name)
        rtol = _get_rtol(fixture_name)
        r_dev_resid = np.array(r_results['residuals_deviance'])
        np.testing.assert_allclose(
            result.residuals_deviance, r_dev_resid, rtol=rtol,
            err_msg=f"Deviance residuals differ from R ({fixture_name})"
        )

    def test_residuals_pearson(self, fixture_name):
        result, r_results, _ = _load_glm_fixture(fixture_name)
        rtol = _get_rtol(fixture_name)
        r_pearson = np.array(r_results['residuals_pearson'])
        np.testing.assert_allclose(
            result.residuals_pearson, r_pearson, rtol=rtol,
            err_msg=f"Pearson residuals differ from R ({fixture_name})"
        )

    def test_residuals_response(self, fixture_name):
        result, r_results, _ = _load_glm_fixture(fixture_name)
        rtol = _get_rtol(fixture_name)
        r_response = np.array(r_results['residuals_response'])
        np.testing.assert_allclose(
            result.residuals_response, r_response, rtol=rtol,
            err_msg=f"Response residuals differ from R ({fixture_name})"
        )

    def test_dispersion(self, fixture_name):
        result, r_results, meta = _load_glm_fixture(fixture_name)
        rtol = _get_rtol(fixture_name)
        np.testing.assert_allclose(
            result.dispersion, r_results['dispersion'], rtol=rtol,
            err_msg=f"Dispersion differs from R ({fixture_name})"
        )

    def test_df_residual(self, fixture_name):
        result, r_results, _ = _load_glm_fixture(fixture_name)
        assert result.df_residual == r_results['df_residual']

    def test_rank(self, fixture_name):
        result, r_results, _ = _load_glm_fixture(fixture_name)
        assert result.rank == r_results['rank']

    def test_returns_glmsolution(self, fixture_name):
        result, _, _ = _load_glm_fixture(fixture_name)
        assert isinstance(result, GLMSolution)
