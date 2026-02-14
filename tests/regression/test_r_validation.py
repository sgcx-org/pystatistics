"""
Validate PyStatistics regression against R reference results.

Each fixture CSV + R results JSON pair becomes a parametrized test.
Tests are skipped if no R fixtures are found (run generate_fixtures.py
and run_r_validation.R first).
"""

import json
import pytest
import numpy as np
from pathlib import Path
from functools import lru_cache

from pystatistics import DataSource
from pystatistics.regression import Design, fit

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _discover_fixtures():
    """Find all fixture names that have both CSV and R results."""
    fixtures = []
    for r_file in sorted(FIXTURES_DIR.glob("*_r_results.json")):
        name = r_file.stem.replace("_r_results", "")
        csv_file = FIXTURES_DIR / f"{name}.csv"
        if csv_file.exists():
            fixtures.append(name)
    return fixtures


FIXTURE_NAMES = _discover_fixtures()


@lru_cache(maxsize=None)
def _load_fixture(name):
    """Load CSV data, R results, and metadata. Cached to avoid re-fitting."""
    csv_path = FIXTURES_DIR / f"{name}.csv"
    r_path = FIXTURES_DIR / f"{name}_r_results.json"
    meta_path = FIXTURES_DIR / f"{name}_meta.json"

    with open(r_path) as f:
        r_results = json.load(f)

    is_ill_conditioned = False
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            if meta.get('condition_number', 1.0) > 1e4:
                is_ill_conditioned = True

    ds = DataSource.from_file(csv_path)
    x_cols = [c for c in ds.metadata['columns'] if c != 'y']
    design = Design.from_datasource(ds, x=x_cols, y='y')
    result = fit(design)

    return result, r_results, is_ill_conditioned


def _tolerances(is_ill_conditioned):
    if is_ill_conditioned:
        return dict(rtol=1e-4, atol=1e-6)
    return dict(rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not FIXTURE_NAMES, reason="No R fixtures found. Run generate_fixtures.py and run_r_validation.R first.")
class TestRValidation:
    """Validate each fixture against R reference."""

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_coefficients(self, fixture_name):
        result, r, ill = _load_fixture(fixture_name)
        tol = _tolerances(ill)
        np.testing.assert_allclose(
            result.coefficients, r['coefficients'], **tol
        )

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_r_squared(self, fixture_name):
        result, r, ill = _load_fixture(fixture_name)
        tol = _tolerances(ill)
        np.testing.assert_allclose(result.r_squared, r['r_squared'], **tol)

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_adjusted_r_squared(self, fixture_name):
        result, r, ill = _load_fixture(fixture_name)
        tol = _tolerances(ill)
        np.testing.assert_allclose(
            result.adjusted_r_squared, r['adj_r_squared'], **tol
        )

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_residual_std_error(self, fixture_name):
        result, r, ill = _load_fixture(fixture_name)
        tol = _tolerances(ill)
        np.testing.assert_allclose(
            result.residual_std_error, r['sigma'], **tol
        )

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_standard_errors(self, fixture_name):
        result, r, ill = _load_fixture(fixture_name)
        tol = _tolerances(ill)
        np.testing.assert_allclose(
            result.standard_errors, r['standard_errors'], **tol
        )

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_t_statistics(self, fixture_name):
        result, r, ill = _load_fixture(fixture_name)
        tol = _tolerances(ill)
        np.testing.assert_allclose(
            result.t_statistics, r['t_statistics'], **tol
        )

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_p_values(self, fixture_name):
        result, r, ill = _load_fixture(fixture_name)
        tol = _tolerances(ill)
        np.testing.assert_allclose(
            result.p_values, r['p_values'], **tol
        )

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_residuals(self, fixture_name):
        result, r, ill = _load_fixture(fixture_name)
        tol = _tolerances(ill)
        np.testing.assert_allclose(
            result.residuals, r['residuals_all'], **tol
        )

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_fitted_values(self, fixture_name):
        result, r, ill = _load_fixture(fixture_name)
        tol = _tolerances(ill)
        np.testing.assert_allclose(
            result.fitted_values, r['fitted_all'], **tol
        )

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_rss(self, fixture_name):
        result, r, ill = _load_fixture(fixture_name)
        tol = _tolerances(ill)
        np.testing.assert_allclose(result.rss, r['rss'], **tol)

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_tss(self, fixture_name):
        result, r, ill = _load_fixture(fixture_name)
        tol = _tolerances(ill)
        np.testing.assert_allclose(result.tss, r['tss'], **tol)

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_df_residual(self, fixture_name):
        result, r, _ = _load_fixture(fixture_name)
        assert result.df_residual == r['df_residual']
