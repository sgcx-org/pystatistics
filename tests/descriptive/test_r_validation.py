"""
Parametrised R validation tests for descriptive statistics.

Compares pystatistics results against R reference values for each
desc_* fixture. Tests are auto-discovered from fixture files.

Run R validation:
    python tests/fixtures/generate_descriptive_fixtures.py
    Rscript tests/fixtures/run_r_descriptive_validation.R
    pytest tests/descriptive/test_r_validation.py -v
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from pystatistics.descriptive import describe, cor, cov, quantile, summary

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r_to_array(values) -> np.ndarray:
    """
    Convert an R JSON value (scalar or array) to a numpy array.

    R's jsonlite writes NA/NaN as the strings "NA" or "NaN" and
    null as JSON null (Python None). This converts all of those to np.nan.
    When p=1, R may write a scalar instead of a 1-element array.
    """
    # Handle scalar values (R unboxes length-1 vectors)
    if not isinstance(values, list):
        if values is None or (isinstance(values, str) and values.upper() in ('NA', 'NAN')):
            return np.array([np.nan], dtype=np.float64)
        return np.array([float(values)], dtype=np.float64)

    result = []
    for v in values:
        if v is None or (isinstance(v, str) and v.upper() in ('NA', 'NAN')):
            result.append(np.nan)
        else:
            result.append(float(v))
    return np.array(result, dtype=np.float64)


def _assert_allclose_with_nan(actual, expected, **tol):
    """Assert arrays are close, handling NaN positions (both must be NaN)."""
    nan_mask = np.isnan(expected)
    if np.any(nan_mask):
        assert np.all(np.isnan(actual[nan_mask])), \
            f"Expected NaN at positions {np.where(nan_mask)} but got {actual[nan_mask]}"
        non_nan = ~nan_mask
        if np.any(non_nan):
            np.testing.assert_allclose(actual[non_nan], expected[non_nan], **tol)
    else:
        np.testing.assert_allclose(actual, expected, **tol)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_fixtures() -> list[str]:
    """Find all desc_* fixtures that have R results."""
    names = []
    for f in sorted(FIXTURES_DIR.glob("desc_*_r_results.json")):
        name = f.stem.replace("_r_results", "")
        csv = FIXTURES_DIR / f"{name}.csv"
        if csv.exists():
            names.append(name)
    return names


FIXTURE_NAMES = _discover_fixtures()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _load_fixture(name: str):
    """Load fixture data, R results, and metadata."""
    csv_path = FIXTURES_DIR / f"{name}.csv"
    meta_path = FIXTURES_DIR / f"{name}_meta.json"
    results_path = FIXTURES_DIR / f"{name}_r_results.json"

    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    with open(meta_path) as f:
        meta = json.load(f)

    with open(results_path) as f:
        r_results = json.load(f)

    return data, r_results, meta


def _get_tolerances(name: str):
    """Get tolerances appropriate for the fixture."""
    if 'extreme' in name:
        return dict(rtol=1e-8, atol=1e-6)
    if 'constant' in name:
        return dict(rtol=1e-8, atol=1e-10)
    return dict(rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
class TestRValidation:
    """Validate descriptive statistics against R reference values."""

    def test_mean(self, fixture_name):
        """Column means match R colMeans()."""
        data, r, meta = _load_fixture(fixture_name)

        if r['has_nan']:
            result = describe(data, use='complete.obs', backend='cpu')
            expected = _r_to_array(r['mean_complete'])
        else:
            result = describe(data, use='everything', backend='cpu')
            expected = _r_to_array(r['mean_everything'])

        _assert_allclose_with_nan(result.mean, expected, **_get_tolerances(fixture_name))

    def test_variance(self, fixture_name):
        """Column variances match R var() (Bessel-corrected)."""
        data, r, meta = _load_fixture(fixture_name)

        if r['has_nan']:
            result = describe(data, use='complete.obs', backend='cpu')
            expected = _r_to_array(r['var_complete'])
        else:
            result = describe(data, use='everything', backend='cpu')
            expected = _r_to_array(r['var_everything'])

        _assert_allclose_with_nan(result.variance, expected, **_get_tolerances(fixture_name))

    def test_sd(self, fixture_name):
        """Column SDs match R sd()."""
        data, r, meta = _load_fixture(fixture_name)

        if r['has_nan']:
            result = describe(data, use='complete.obs', backend='cpu')
            expected = _r_to_array(r['sd_complete'])
        else:
            result = describe(data, use='everything', backend='cpu')
            expected = _r_to_array(r['sd_everything'])

        _assert_allclose_with_nan(result.sd, expected, **_get_tolerances(fixture_name))

    def test_covariance(self, fixture_name):
        """Covariance matrix matches R cov()."""
        data, r, meta = _load_fixture(fixture_name)
        p = meta['p']

        if r['has_nan']:
            result = cov(data, use='complete.obs', backend='cpu')
            expected = _r_to_array(r['cov_complete']).reshape(p, p, order='F')
        else:
            result = cov(data, use='everything', backend='cpu')
            expected = _r_to_array(r['cov_everything']).reshape(p, p, order='F')

        _assert_allclose_with_nan(
            result.covariance_matrix, expected, **_get_tolerances(fixture_name)
        )

    def test_cor_pearson(self, fixture_name):
        """Pearson correlation matches R cor(method='pearson')."""
        data, r, meta = _load_fixture(fixture_name)
        p = meta['p']

        if r['has_nan']:
            result = cor(data, method='pearson', use='pairwise.complete.obs', backend='cpu')
            key = 'cor_pearson_pairwise'
        else:
            result = cor(data, method='pearson', use='everything', backend='cpu')
            key = 'cor_pearson_everything'

        expected = _r_to_array(r[key]).reshape(p, p, order='F')
        _assert_allclose_with_nan(
            result.correlation_pearson, expected, **_get_tolerances(fixture_name)
        )

    def test_cor_spearman(self, fixture_name):
        """Spearman correlation matches R cor(method='spearman')."""
        data, r, meta = _load_fixture(fixture_name)
        p = meta['p']

        if r['has_nan']:
            key = 'cor_spearman_pairwise'
        else:
            key = 'cor_spearman_everything'

        if key not in r or r[key] is None:
            pytest.skip(f"R did not compute {key} for {fixture_name}")

        result = cor(
            data, method='spearman',
            use='pairwise.complete.obs' if r['has_nan'] else 'everything',
            backend='cpu',
        )
        expected = _r_to_array(r[key]).reshape(p, p, order='F')
        _assert_allclose_with_nan(
            result.correlation_spearman, expected, **_get_tolerances(fixture_name)
        )

    def test_cor_kendall(self, fixture_name):
        """Kendall correlation matches R cor(method='kendall')."""
        data, r, meta = _load_fixture(fixture_name)
        p = meta['p']

        if r['has_nan']:
            key = 'cor_kendall_pairwise'
        else:
            key = 'cor_kendall_everything'

        if key not in r or r[key] is None:
            pytest.skip(f"R did not compute {key} for {fixture_name}")

        result = cor(
            data, method='kendall',
            use='pairwise.complete.obs' if r['has_nan'] else 'everything',
            backend='cpu',
        )
        expected = _r_to_array(r[key]).reshape(p, p, order='F')

        # Kendall has some numerical variation, use slightly relaxed tol
        tol = dict(rtol=1e-8, atol=1e-10)
        if 'extreme' in fixture_name:
            tol = dict(rtol=1e-6, atol=1e-6)

        _assert_allclose_with_nan(result.correlation_kendall, expected, **tol)

    @pytest.mark.parametrize("qtype", range(1, 10))
    def test_quantiles(self, fixture_name, qtype):
        """All 9 quantile types match R quantile()."""
        data, r, meta = _load_fixture(fixture_name)
        p = meta['p']
        key = f'quantiles_type{qtype}'

        if key not in r:
            pytest.skip(f"No {key} in R results for {fixture_name}")

        if r['has_nan']:
            result = quantile(data, type=qtype, use='complete.obs', backend='cpu')
        else:
            result = quantile(data, type=qtype, use='everything', backend='cpu')

        # R: apply(data, 2, quantile) gives 5 probs x p cols, flattened column-major
        expected = _r_to_array(r[key]).reshape(5, p, order='F')
        _assert_allclose_with_nan(result.quantiles, expected, **_get_tolerances(fixture_name))

    def test_summary(self, fixture_name):
        """Summary table matches R summary()."""
        data, r, meta = _load_fixture(fixture_name)
        p = meta['p']

        if r['has_nan']:
            result = summary(data, use='complete.obs', backend='cpu')
        else:
            result = summary(data, use='everything', backend='cpu')

        # R: apply(data, 2, summary) gives 6 rows x p cols, flattened column-major
        expected = _r_to_array(r['summary']).reshape(6, p, order='F')
        _assert_allclose_with_nan(result.summary_table, expected, **_get_tolerances(fixture_name))

    def test_skewness(self, fixture_name):
        """Skewness matches R e1071::skewness(type=2)."""
        data, r, meta = _load_fixture(fixture_name)

        if 'skewness' not in r or r['skewness'] is None:
            pytest.skip("No skewness in R results")

        if r['has_nan']:
            result = describe(data, use='complete.obs', backend='cpu')
        else:
            result = describe(data, use='everything', backend='cpu')

        expected = _r_to_array(r['skewness'])
        _assert_allclose_with_nan(result.skewness, expected, **_get_tolerances(fixture_name))

    def test_kurtosis(self, fixture_name):
        """Kurtosis matches R e1071::kurtosis(type=2)."""
        data, r, meta = _load_fixture(fixture_name)

        if 'kurtosis' not in r or r['kurtosis'] is None:
            pytest.skip("No kurtosis in R results")

        if r['has_nan']:
            result = describe(data, use='complete.obs', backend='cpu')
        else:
            result = describe(data, use='everything', backend='cpu')

        expected = _r_to_array(r['kurtosis'])
        _assert_allclose_with_nan(result.kurtosis, expected, **_get_tolerances(fixture_name))
