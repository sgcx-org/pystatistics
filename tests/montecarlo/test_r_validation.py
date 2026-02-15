"""
Parametrised R validation tests for Monte Carlo module.

Compares pystatistics bootstrap/permutation results against R reference
values from the boot package. Tests are auto-discovered from fixture files.

Key tolerance notes:
- t0 (observed statistic): tight tolerance (deterministic, same data)
- bias / SE: moderate tolerance (stochastic — R and Python use different
  RNGs, so the bootstrap samples differ. With large R the law of large
  numbers brings them close.)
- CI endpoints: moderate tolerance (functions of stochastic replicates)
- permutation p-values: wide tolerance (stochastic)

Run R validation:
    python tests/fixtures/generate_montecarlo_fixtures.py
    Rscript tests/fixtures/run_r_montecarlo_validation.R
    pytest tests/montecarlo/test_r_validation.py -v
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from pystatistics.montecarlo import boot, boot_ci, permutation_test

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _load_meta(name: str) -> dict:
    """Load fixture metadata."""
    path = FIXTURES_DIR / f"{name}_meta.json"
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=32)
def _load_r_results(name: str) -> dict:
    """Load R reference results."""
    path = FIXTURES_DIR / f"{name}_r_results.json"
    with open(path) as f:
        return json.load(f)


def _boot_statistic_from_meta(meta: dict):
    """Build the statistic function from fixture metadata."""
    desc = meta.get("description", "").lower()

    if "variance" in desc:
        def stat_fn(data, indices):
            return np.array([np.var(data[indices], ddof=1)])
        return stat_fn
    elif "median" in desc:
        def stat_fn(data, indices):
            return np.array([np.median(data[indices])])
        return stat_fn
    else:
        # Default: mean
        def stat_fn(data, indices):
            return np.array([np.mean(data[indices])])
        return stat_fn


def _mean_diff(x, y):
    """Permutation statistic: difference in means."""
    return np.mean(x) - np.mean(y)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_boot_fixtures() -> list[str]:
    """Find all mc_boot_* fixtures (excluding _ci_ variants)."""
    names = []
    for f in sorted(FIXTURES_DIR.glob("mc_boot_*_r_results.json")):
        name = f.stem.replace("_r_results", "")
        meta_path = FIXTURES_DIR / f"{name}_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("type") == "boot":
                names.append(name)
    return names


def _discover_ci_fixtures() -> list[str]:
    """Find all mc_boot_ci_* fixtures."""
    names = []
    for f in sorted(FIXTURES_DIR.glob("mc_boot_ci_*_r_results.json")):
        name = f.stem.replace("_r_results", "")
        meta_path = FIXTURES_DIR / f"{name}_meta.json"
        if meta_path.exists():
            names.append(name)
    return names


def _discover_perm_fixtures() -> list[str]:
    """Find all mc_perm_* fixtures."""
    names = []
    for f in sorted(FIXTURES_DIR.glob("mc_perm_*_r_results.json")):
        name = f.stem.replace("_r_results", "")
        meta_path = FIXTURES_DIR / f"{name}_meta.json"
        if meta_path.exists():
            names.append(name)
    return names


BOOT_FIXTURES = _discover_boot_fixtures()
CI_FIXTURES = _discover_ci_fixtures()
PERM_FIXTURES = _discover_perm_fixtures()


# ---------------------------------------------------------------------------
# Bootstrap tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name", BOOT_FIXTURES)
class TestBootstrapRValidation:
    """Validate bootstrap results against R's boot package."""

    def test_t0(self, fixture_name):
        """Observed statistic (t0) matches R exactly (deterministic)."""
        meta = _load_meta(fixture_name)
        r = _load_r_results(fixture_name)

        data = np.array(meta["data"])
        stat_fn = _boot_statistic_from_meta(meta)

        result = boot(
            data, stat_fn, R=meta["R"],
            sim=meta["sim"], stype=meta.get("stype", "i"),
            seed=meta["seed"],
        )

        # t0 is deterministic — same data, same statistic
        assert result.t0[0] == pytest.approx(
            r["t0"], rel=1e-10, abs=1e-15,
        ), f"t0 mismatch for {fixture_name}"

    def test_bias(self, fixture_name):
        """Bootstrap bias estimate is close to R's (stochastic tolerance)."""
        meta = _load_meta(fixture_name)
        r = _load_r_results(fixture_name)

        data = np.array(meta["data"])
        stat_fn = _boot_statistic_from_meta(meta)

        result = boot(
            data, stat_fn, R=meta["R"],
            sim=meta["sim"], stype=meta.get("stype", "i"),
            seed=meta["seed"],
        )

        # Different RNGs, so bias won't match exactly. Use moderate tolerance.
        # With R=999-1999, bias estimates have non-trivial MC variance.
        assert result.bias[0] == pytest.approx(
            r["bias"], abs=0.3,
        ), f"Bias mismatch for {fixture_name}"

    def test_se(self, fixture_name):
        """Bootstrap SE is close to R's (stochastic tolerance)."""
        meta = _load_meta(fixture_name)
        r = _load_r_results(fixture_name)

        data = np.array(meta["data"])
        stat_fn = _boot_statistic_from_meta(meta)

        result = boot(
            data, stat_fn, R=meta["R"],
            sim=meta["sim"], stype=meta.get("stype", "i"),
            seed=meta["seed"],
        )

        # SE should be in the right ballpark (within 20% relative)
        assert result.se[0] == pytest.approx(
            r["se"], rel=0.20,
        ), f"SE mismatch for {fixture_name}"

    def test_replicate_count(self, fixture_name):
        """Number of replicates matches."""
        meta = _load_meta(fixture_name)
        r = _load_r_results(fixture_name)

        data = np.array(meta["data"])
        stat_fn = _boot_statistic_from_meta(meta)

        result = boot(
            data, stat_fn, R=meta["R"],
            sim=meta["sim"], stype=meta.get("stype", "i"),
            seed=meta["seed"],
        )

        assert result.R == r["R"]
        assert result.t.shape == (r["R"], 1)


# ---------------------------------------------------------------------------
# Bootstrap CI tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name", CI_FIXTURES)
class TestBootstrapCIRValidation:
    """Validate bootstrap CI against R's boot.ci()."""

    def test_ci_endpoints(self, fixture_name):
        """CI endpoints are close to R's boot.ci() (stochastic tolerance)."""
        meta = _load_meta(fixture_name)
        r = _load_r_results(fixture_name)

        data = np.array(meta["data"])
        stat_fn = _boot_statistic_from_meta(meta)

        result = boot(
            data, stat_fn, R=meta["R"],
            sim=meta["sim"], stype=meta.get("stype", "i"),
            seed=meta["seed"],
        )

        ci_types = meta["ci_types"]
        conf_level = meta["conf_level"]

        for ci_type in ci_types:
            if ci_type not in r.get("ci", {}):
                continue

            r_ci = r["ci"][ci_type]
            if r_ci is None:
                continue

            ci_result = boot_ci(
                result, type=ci_type, conf=conf_level,
            )
            py_ci = ci_result.ci[ci_type]

            # CI endpoints are stochastic — use moderate tolerance.
            # With R=2999, typical MC variance in CI endpoints is ~0.1-0.5.
            #
            # Use both relative and absolute tolerance. For CI bounds
            # near zero, absolute tolerance matters more.
            for i, label in enumerate(["lower", "upper"]):
                assert py_ci[0, i] == pytest.approx(
                    r_ci[i], abs=0.5, rel=0.10,
                ), f"{ci_type} CI {label} mismatch for {fixture_name}"

    def test_ci_ordering(self, fixture_name):
        """CI lower bound < upper bound for all types."""
        meta = _load_meta(fixture_name)

        data = np.array(meta["data"])
        stat_fn = _boot_statistic_from_meta(meta)

        result = boot(
            data, stat_fn, R=meta["R"],
            sim=meta["sim"], stype=meta.get("stype", "i"),
            seed=meta["seed"],
        )

        for ci_type in meta["ci_types"]:
            ci_result = boot_ci(
                result, type=ci_type, conf=meta["conf_level"],
            )
            ci = ci_result.ci[ci_type]
            assert ci[0, 0] < ci[0, 1], \
                f"{ci_type} CI not ordered for {fixture_name}"

    def test_ci_contains_t0(self, fixture_name):
        """CI should generally contain t0 for well-behaved data."""
        meta = _load_meta(fixture_name)

        data = np.array(meta["data"])
        stat_fn = _boot_statistic_from_meta(meta)

        result = boot(
            data, stat_fn, R=meta["R"],
            sim=meta["sim"], stype=meta.get("stype", "i"),
            seed=meta["seed"],
        )

        # Percentile CI with 95% confidence should contain t0 for normal data
        ci_result = boot_ci(result, type="perc", conf=meta["conf_level"])
        ci = ci_result.ci["perc"]

        # This is not guaranteed for highly biased estimators, but should hold
        # for the mean/variance/median with our test data
        assert ci[0, 0] < result.t0[0] < ci[0, 1], \
            f"Percentile CI does not contain t0 for {fixture_name}"


# ---------------------------------------------------------------------------
# Permutation tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name", PERM_FIXTURES)
class TestPermutationRValidation:
    """Validate permutation test against R reference values."""

    def test_observed_stat(self, fixture_name):
        """Observed statistic matches R exactly (deterministic)."""
        meta = _load_meta(fixture_name)
        r = _load_r_results(fixture_name)

        x = np.array(meta["x"])
        y = np.array(meta["y"])

        result = permutation_test(
            x, y, _mean_diff, R=meta["R"],
            alternative=meta["alternative"],
            seed=meta["seed"],
        )

        # Observed statistic is deterministic
        assert result.observed_stat == pytest.approx(
            r["observed_stat"], rel=1e-10, abs=1e-15,
        ), f"Observed stat mismatch for {fixture_name}"

    def test_p_value(self, fixture_name):
        """Permutation p-value is close to R's (stochastic tolerance)."""
        meta = _load_meta(fixture_name)
        r = _load_r_results(fixture_name)

        x = np.array(meta["x"])
        y = np.array(meta["y"])

        result = permutation_test(
            x, y, _mean_diff, R=meta["R"],
            alternative=meta["alternative"],
            seed=meta["seed"],
        )

        # P-values are stochastic. With R=9999, typical MC variance in
        # p-value is ≈ sqrt(p*(1-p)/R). For p≈0.001, this is ~0.003.
        # For p≈0.5, this is ~0.005. Use abs=0.05 as safe bound.
        if r["p_value"] < 0.01:
            # Very small p-values: both should be very small
            assert result.p_value < 0.05, \
                f"p-value too large for {fixture_name}"
        elif r["p_value"] > 0.10:
            # Non-significant: both should be non-significant
            assert result.p_value > 0.01, \
                f"p-value too small for {fixture_name}"

        # Also check absolute proximity
        assert result.p_value == pytest.approx(
            r["p_value"], abs=0.05,
        ), f"p-value mismatch for {fixture_name}"

    def test_perm_distribution_properties(self, fixture_name):
        """Permutation distribution has expected properties."""
        meta = _load_meta(fixture_name)
        r = _load_r_results(fixture_name)

        x = np.array(meta["x"])
        y = np.array(meta["y"])

        result = permutation_test(
            x, y, _mean_diff, R=meta["R"],
            alternative=meta["alternative"],
            seed=meta["seed"],
        )

        # Permutation distribution should be centered near zero
        # (under H0, shuffling destroys any real difference)
        assert abs(np.mean(result.perm_stats)) == pytest.approx(
            0.0, abs=0.3,
        ), f"Perm distribution not centered for {fixture_name}"

        # SD of permutation distribution should be similar to R's
        assert np.std(result.perm_stats) == pytest.approx(
            r["perm_stats_sd"], rel=0.15,
        ), f"Perm distribution SD mismatch for {fixture_name}"

    def test_alternative_stored(self, fixture_name):
        """Alternative hypothesis is stored correctly."""
        meta = _load_meta(fixture_name)

        x = np.array(meta["x"])
        y = np.array(meta["y"])

        result = permutation_test(
            x, y, _mean_diff, R=meta["R"],
            alternative=meta["alternative"],
            seed=meta["seed"],
        )

        assert result.alternative == meta["alternative"]
