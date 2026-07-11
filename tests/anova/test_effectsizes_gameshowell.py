"""
Tests for the VA-11 additions: omega-squared / partial omega-squared effect
sizes, pairwise Cohen's d on post-hoc comparisons, and the Games-Howell
(unequal-variance) post-hoc test.

R references (committed fixture ``anova_effectsizes_gameshowell.json``):
    - omega^2 / partial omega^2 : effectsize::omega_squared
    - Games-Howell              : base-R ptukey/qtukey (canonical formula)
    - pairwise Cohen's d        : effectsize::cohens_d (pooled SD)
on a 4-group one-way design with unequal n and unequal variances (n=85).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pystatistics.anova import anova_oneway, anova_posthoc
from pystatistics.core.exceptions import ValidationError

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(scope="module")
def es_data():
    d = json.loads((_FIXTURES / "anova_effectsizes_gameshowell.json").read_text())
    y = np.asarray(d["y"], dtype=float)
    g = np.asarray(d["g"])
    return y, g, d


class TestOmegaSquared:
    def test_omega_squared_matches_effectsize(self, es_data):
        y, g, d = es_data
        sol = anova_oneway(y, g)
        assert abs(sol.omega_squared["group"] - d["omega2"]) < 1e-9

    def test_partial_omega_equals_omega_for_oneway(self, es_data):
        y, g, d = es_data
        sol = anova_oneway(y, g)
        # For a one-way design partial omega^2 == omega^2 (effectsize does this).
        assert abs(sol.partial_omega_squared["group"]
                   - sol.omega_squared["group"]) < 1e-12
        assert abs(sol.partial_omega_squared["group"] - d["partial_omega2"]) < 1e-9

    def test_omega_below_eta(self, es_data):
        y, g, d = es_data
        sol = anova_oneway(y, g)
        # omega^2 is the less-biased (smaller) sibling of eta^2.
        assert sol.omega_squared["group"] < sol.eta_squared["group"]
        assert abs(sol.eta_squared["group"] - d["eta2"]) < 1e-8


class TestGamesHowell:
    def test_matches_base_r(self, es_data):
        y, g, d = es_data
        sol = anova_oneway(y, g)
        gh = anova_posthoc(sol, method="games-howell")
        assert gh.method == "games-howell"
        ref = d["games_howell"]
        assert len(gh.comparisons) == len(ref)
        for c in gh.comparisons:
            diff, se, df, p, lo, hi = ref[f"{c.group2}-{c.group1}"]
            assert abs(c.diff - diff) < 1e-4
            assert abs(c.se - se) < 1e-4
            assert abs(c.p_value - p) < 1e-4
            assert abs(c.ci_lower - lo) < 1e-4
            assert abs(c.ci_upper - hi) < 1e-4

    def test_ignores_pooled_error_term(self, es_data):
        """Games-Howell reports no single pooled MSE/df (per-pair Welch)."""
        y, g, _ = es_data
        sol = anova_oneway(y, g)
        gh = anova_posthoc(sol, method="games-howell")
        params = gh._result.params
        assert np.isnan(params.mse)
        assert params.df_error == -1

    def test_requires_two_per_group(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g = np.array(["A", "A", "A", "A", "B"])  # B has a single obs
        sol = anova_oneway(y, g)
        with pytest.raises(ValidationError, match="2 observations"):
            anova_posthoc(sol, method="games-howell")

    def test_differs_from_tukey_under_heteroscedasticity(self, es_data):
        """With unequal variances, Games-Howell CIs differ from Tukey's."""
        y, g, _ = es_data
        sol = anova_oneway(y, g)
        gh = {(c.group1, c.group2): c for c in
              anova_posthoc(sol, method="games-howell").comparisons}
        tk = {(c.group1, c.group2): c for c in
              anova_posthoc(sol, method="tukey").comparisons}
        # at least one pair has a materially different SE
        assert any(abs(gh[k].se - tk[k].se) > 1e-3 for k in gh)


class TestPairwiseCohensD:
    def test_cohens_d_matches_effectsize(self, es_data):
        y, g, d = es_data
        sol = anova_oneway(y, g)
        for method in ("tukey", "bonferroni", "games-howell"):
            ph = anova_posthoc(sol, method=method)
            for c in ph.comparisons:
                ref = d["cohens_d"][f"{c.group2}-{c.group1}"]
                assert c.cohens_d is not None
                assert abs(c.cohens_d - ref) < 1e-6, (method, c.group1, c.group2)

    def test_cohens_d_sign_follows_diff(self, es_data):
        y, g, _ = es_data
        sol = anova_oneway(y, g)
        for c in anova_posthoc(sol, method="tukey").comparisons:
            assert np.sign(c.cohens_d) == np.sign(c.diff)
