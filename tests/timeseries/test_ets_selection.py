"""
Tests for ETS "Z"-wildcard automatic model selection and R parity.

Covers the candidate-set rules, selection behaviour, the honest
cross-engine verification against R forecast::ets (fixture-driven; see
tests/fixtures/generate_ets_r_reference.R), damped-fit convergence, the
log-likelihood reporting convention, and boundary validation of the
public ets() entry point.  Recursion/fitting-engine unit tests live in
test_ets.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from pystatistics.core.exceptions import ValidationError
from pystatistics.timeseries._ets_select import ets

import json
from pathlib import Path

_ETS_FIXTURE = Path(__file__).parent.parent / "fixtures" / "ets_r_reference.json"


def _r_reference():
    with open(_ETS_FIXTURE) as fh:
        return json.load(fh)


class TestZZZCandidateSet:
    """The enumerated candidate set mirrors forecast::ets exactly."""

    def test_full_zzz_seasonal_positive_has_15_candidates(self):
        ref = _r_reference()
        x = np.asarray(ref["selection"]["airpassengers"]["x"])
        sol = ets(x, model="ZZZ", period=12)
        sel = sol.info["selection"]
        names = {c["model"] for c in sel["candidates"]}
        assert len(sel["candidates"]) == 15
        # A-error x multiplicative-season trio excluded (restrict=TRUE).
        skipped_models = {s["model"] for s in sel["skipped"]}
        assert {"ANM", "AAdM", "AAM"} <= skipped_models
        assert "ETS(M,A,M)" in names and "ETS(A,N,N)" in names

    def test_zzn_has_6_candidates(self):
        ref = _r_reference()
        x = np.asarray(ref["selection"]["airpassengers"]["x"])
        sol = ets(x, model="ZZN", period=12)
        sel = sol.info["selection"]
        names = {c["model"] for c in sel["candidates"]}
        assert names == {
            "ETS(A,N,N)", "ETS(A,Ad,N)", "ETS(A,A,N)",
            "ETS(M,N,N)", "ETS(M,Ad,N)", "ETS(M,A,N)",
        }

    def test_nonseasonal_period1_zzz(self):
        ref = _r_reference()
        x = np.asarray(ref["selection"]["nile"]["x"])
        sol = ets(x, model="ZZZ", period=1)
        assert len(sol.info["selection"]["candidates"]) == 6

    def test_negative_data_drops_multiplicative_error(self):
        ref = _r_reference()
        x = np.asarray(ref["selection"]["diff_nile"]["x"])
        sol = ets(x, model="ZZZ", period=1)
        sel = sol.info["selection"]
        assert all(c["model"].startswith("ETS(A") for c in sel["candidates"])
        assert any("strictly positive" in s["reason"] for s in sel["skipped"])

    def test_damped_true_restricts_trend_candidates(self):
        ref = _r_reference()
        x = np.asarray(ref["selection"]["nile"]["x"])
        sol = ets(x, model="ZZN", period=1, damped=True)
        names = {c["model"] for c in sol.info["selection"]["candidates"]}
        assert names == {"ETS(A,Ad,N)", "ETS(M,Ad,N)"}

    def test_damped_false_excludes_damped(self):
        ref = _r_reference()
        x = np.asarray(ref["selection"]["nile"]["x"])
        sol = ets(x, model="ZZN", period=1, damped=False)
        names = {c["model"] for c in sol.info["selection"]["candidates"]}
        assert names == {"ETS(A,N,N)", "ETS(A,A,N)",
                         "ETS(M,N,N)", "ETS(M,A,N)"}


class TestZZZSelection:
    """Selection agrees with R where the engines' optima agree, and is
    always internally consistent with the disclosed candidate table."""

    @pytest.mark.parametrize("name", ["usaccdeaths", "nile", "wwwusage",
                                      "diff_nile"])
    def test_selection_matches_r(self, name):
        """Datasets where pystatistics and forecast::ets agree exactly.

        (The engine optimises the same parameter space as R, but the
        selected model can still differ where R's Nelder-Mead stalls
        short of a candidate's optimum — see timeseries/_ets_select.py;
        those datasets are exercised by the two tests below.)
        """
        case = _r_reference()["selection"][name]
        sol = ets(np.asarray(case["x"]), model=case["model_arg"],
                  period=case["period"])
        assert sol.spec.name == case["method"]

    @pytest.mark.parametrize("name", ["airpassengers", "co2", "lynx",
                                      "airpassengers_zzn",
                                      "airpassengers_azz",
                                      "airpassengers_mzz"])
    def test_selection_is_argmin_of_disclosed_table(self, name):
        case = _r_reference()["selection"][name]
        sol = ets(np.asarray(case["x"]), model=case["model_arg"],
                  period=case["period"])
        sel = sol.info["selection"]
        best = min(sel["candidates"], key=lambda c: c[sel["ic"]])
        assert sel["selected"] == best["model"]
        assert sel["selected"] == sol.spec.name

    @pytest.mark.parametrize("name", ["airpassengers", "co2", "lynx",
                                      "airpassengers_zzn",
                                      "airpassengers_azz",
                                      "airpassengers_mzz"])
    def test_divergent_selection_dominates_r_choice(self, name):
        """Where the selection differs from R, it must be because our
        engine found a better optimum, never a worse criterion value:
        the selected model's AICc (converted to R's log-likelihood
        convention) must beat the AICc R reported for *its* selection."""
        case = _r_reference()["selection"][name]
        sol = ets(np.asarray(case["x"]), model=case["model_arg"],
                  period=case["period"])
        n = case["n"]
        const = 0.5 * n * (np.log(n / (2.0 * np.pi)) - 1.0)
        aicc_r_convention = sol.aicc + 2.0 * const
        assert aicc_r_convention <= case["aicc"] + 0.01

    @pytest.mark.parametrize("name", ["airpassengers", "usaccdeaths",
                                      "co2", "nile", "wwwusage", "lynx",
                                      "diff_nile", "airpassengers_zzn",
                                      "airpassengers_azz",
                                      "airpassengers_mzz"])
    def test_selection_dominates_r_under_r_own_likelihood(self, name):
        """The honest cross-engine verification, pinned as a regression
        test so no future change silently 'corrects' our selection back
        to a worse one.

        The fixture stores, for each dataset, the AICc of *our* selected
        fit's parameters evaluated by R's own likelihood code
        (forecast:::pegelsresid.C, transplant harness self-validated in
        generate_ets_r_reference.R) and R's admissible() verdict.
        Refitting the same model spec in R would only re-run R's
        Nelder-Mead — the optimiser being compared — so the fixture
        numbers are the only fair yardstick.  Our pick must be admissible
        and score at least as well as R's pick under R's own AICc; on the
        six divergent datasets it must be strictly better.

        If the selection legitimately changes, regenerate the fixture
        (generate_ets_py_params.py then generate_ets_r_reference.R) and
        re-verify rather than relaxing this test.
        """
        case = _r_reference()["selection"][name]
        sol = ets(np.asarray(case["x"]), model=case["model_arg"],
                  period=case["period"])
        # Drift guard: the stored cross-scoring belongs to THIS selection.
        assert sol.spec.name == case["py_pick"]
        assert bool(case["py_pick_admissible"])
        assert case["py_pick_aicc_in_r"] <= case["aicc"] + 0.01
        if sol.spec.name != case["method"]:
            assert case["py_pick_aicc_in_r"] < case["aicc"] - 1e-6

    @pytest.mark.parametrize("name", ["airpassengers", "usaccdeaths",
                                      "nile", "lynx"])
    def test_selected_params_in_usual_region(self, name):
        """Fitted smoothing parameters respect R's usual region and the
        seasonal normalisation (the aligned parameter space)."""
        case = _r_reference()["selection"][name]
        sol = ets(np.asarray(case["x"]), model=case["model_arg"],
                  period=case["period"])
        eps = 1e-4
        assert eps <= sol.alpha <= 1.0 - eps
        if sol.beta is not None:
            assert eps <= sol.beta <= sol.alpha
        if sol.gamma is not None:
            assert eps <= sol.gamma <= 1.0 - sol.alpha
        if sol.phi is not None:
            assert 0.8 <= sol.phi <= 0.98
        if sol.init_season is not None:
            target = 0.0 if sol.spec.season == "A" else float(sol.spec.period)
            assert np.sum(sol.init_season) == pytest.approx(target, abs=1e-8)

    def test_default_model_is_zzz(self):
        """ets(y) now auto-selects, matching forecast::ets's default."""
        case = _r_reference()["selection"]["nile"]
        sol = ets(np.asarray(case["x"]))
        assert "selection" in sol.info
        assert sol.info["selection"]["requested"] == "ZZZ"

    def test_selection_uses_requested_ic(self):
        case = _r_reference()["selection"]["nile"]
        x = np.asarray(case["x"])
        sol = ets(x, model="ZZN", ic="bic")
        sel = sol.info["selection"]
        assert sel["ic"] == "bic"
        best = min(sel["candidates"], key=lambda c: c["bic"])
        assert sel["selected"] == best["model"]

    def test_fully_specified_model_bypasses_selection(self):
        """No 'Z' -> fit exactly what was asked, no selection metadata."""
        case = _r_reference()["selection"]["nile"]
        sol = ets(np.asarray(case["x"]), model="ANN")
        assert sol.spec.name == "ETS(A,N,N)"
        assert "selection" not in sol.info


class TestDampedConvergence:
    """Damped-trend fits reach their optimum and say so (regression for
    the phi optimiser stall: R's initparam starts phi at 0.9782, 99% of
    the way to the 0.98 bound, where the logit transform saturates and
    the numerical phi-gradient vanishes; large damped fits then exhausted
    scipy's default evaluation budget and reported converged=False)."""

    def test_co2_madm_converges_and_reaches_r_optimum(self):
        """The 4.6.2 failure case: co2 MAdM stalled unconverged at R's
        value. It must now converge AND match or beat R's own damped
        optimum (the fixture's co2 selection IS R's ETS(M,Ad,M))."""
        case = _r_reference()["selection"]["co2"]
        assert case["method"] == "ETS(M,Ad,M)"  # fixture sanity
        sol = ets(np.asarray(case["x"]), model="MAdM", period=12)
        assert sol.converged
        assert 0.8 <= sol.phi <= 0.98
        n = case["n"]
        const = 0.5 * n * (np.log(n / (2.0 * np.pi)) - 1.0)
        assert sol.aicc + 2.0 * const <= case["aicc"] + 0.01

    @pytest.mark.parametrize("name,model,period", [
        ("airpassengers", "MAdM", 12),
        ("airpassengers", "AAdA", 12),
        ("usaccdeaths", "MAdM", 12),
        ("lynx", "MAdN", 1),
        ("nile", "AAdN", 1),
        ("wwwusage", "AAdN", 1),
    ])
    def test_damped_reference_fits_converge(self, name, model, period):
        case = _r_reference()["selection"][name]
        sol = ets(np.asarray(case["x"]), model=model, period=period)
        assert sol.converged
        assert 0.8 <= sol.phi <= 0.98

    def test_free_phi_dominates_fixed_phi_probes(self):
        """A free-phi damped fit optimises over the whole (0.8, 0.98) box,
        so it must match or beat a fit with phi fixed at any probe point.
        The 4.6.2 stall failed exactly this: co2 MAdM with free phi
        (aicc 172.59) was worse than the same model with phi fixed at
        0.98 (aicc 171.23)."""
        case = _r_reference()["selection"]["co2"]
        x = np.asarray(case["x"])
        free = ets(x, model="MAdM", period=12)
        for probe in (0.9, 0.98):
            fixed = ets(x, model="MAdM", period=12, phi=probe)
            # phi is one fewer estimated parameter when fixed; compare
            # log-likelihoods, which share the same convention.
            assert free.log_likelihood >= fixed.log_likelihood - 0.01


class TestZZZFailures:
    """Explicit requests that cannot be honoured fail loud."""

    def test_multiplicative_error_wildcard_on_negative_data(self):
        y = np.array([1.0, -2.0, 3.0, -1.0, 2.0, 0.5, 1.5, -0.5] * 3)
        with pytest.raises(ValidationError, match="strictly positive"):
            ets(y, model="MZZ", period=1)

    def test_explicit_seasonal_with_period_one(self):
        y = np.arange(30, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="period >= 2"):
            ets(y, model="ZZA", period=1)

    def test_explicit_seasonal_with_period_over_24(self):
        y = np.arange(120, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="period <= 24"):
            ets(y, model="ZZA", period=25)

    def test_wildcard_seasonal_with_period_over_24_warns_and_drops(self):
        rng = np.random.default_rng(8)
        y = rng.normal(100.0, 5.0, 200)
        sol = ets(y, model="ZZZ", period=30)
        assert any("period 30" in w for w in sol.warnings)
        assert all("N)" in c["model"] for c in sol.info["selection"]["candidates"])

    def test_damped_false_with_explicit_ad_string(self):
        y = np.arange(30, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="damped"):
            ets(y, model="AAdZ", period=1, damped=False)

    def test_invalid_ic(self):
        y = np.arange(30, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="ic"):
            ets(y, model="ZZN", ic="hqic")


class TestLogLikelihoodConvention:
    """The documented full-Gaussian vs R-concentrated constant.

    ll_pystat = ll_R + 0.5 * n * [log(n / (2*pi)) - 1]; verified against
    stored forecast::ets fits where both optimisers find the same optimum.
    """

    @pytest.mark.parametrize("name", ["nile_ann", "nile_aan"])
    def test_constant_offset_vs_r(self, name):
        case = _r_reference()["fixed"][name]
        x = np.asarray(case["x"])
        sol = ets(x, model=case["model_arg"], period=case["period"])
        n = case["n"]
        const = 0.5 * n * (np.log(n / (2.0 * np.pi)) - 1.0)
        assert abs((sol.log_likelihood - case["loglik"]) - const) < 0.01

    def test_constant_value_n100(self):
        """The documented example: n=100 -> constant 88.36."""
        const = 0.5 * 100 * (np.log(100 / (2.0 * np.pi)) - 1.0)
        assert abs(const - 88.3647) < 1e-3

    @pytest.mark.parametrize("name", ["airpassengers_aaa",
                                      "airpassengers_mam",
                                      "usaccdeaths_ana"])
    def test_seasonal_loglik_at_least_r_optimum(self, name):
        """The engine optimises the same parameter space as R but with
        L-BFGS-B instead of R's Nelder-Mead (which stalls on these
        seasonal fits — see _ets_select.py), so after removing the
        convention constant its log-likelihood should not fall
        meaningfully below R's optimum."""
        case = _r_reference()["fixed"][name]
        x = np.asarray(case["x"])
        sol = ets(x, model=case["model_arg"], period=case["period"])
        n = case["n"]
        const = 0.5 * n * (np.log(n / (2.0 * np.pi)) - 1.0)
        assert sol.log_likelihood - const >= case["loglik"] - 0.5


class TestWildcardValidation:
    """Boundary validation of the public ets() (adversarial-review fixes)."""

    def test_float_period_raises(self):
        y = np.arange(40, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="period.*integer"):
            ets(y, period=12.0)

    def test_string_period_raises(self):
        y = np.arange(40, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="period.*integer"):
            ets(y, period="12")

    def test_bool_period_raises(self):
        y = np.arange(40, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="period.*integer"):
            ets(y, period=True)

    def test_empty_series_raises(self):
        with pytest.raises(ValidationError, match="at least 3 observations"):
            ets(np.array([]))

    def test_two_dimensional_series_raises(self):
        y = np.arange(40, dtype=float).reshape(20, 2)
        with pytest.raises(ValidationError, match="1-D"):
            ets(y, model="ZZN")

    def test_two_dimensional_series_raises_fully_specified(self):
        y = np.arange(40, dtype=float).reshape(20, 2)
        with pytest.raises(ValidationError, match="1-D"):
            ets(y, model="ANN")

    def test_column_vector_accepted(self):
        """(n, 1) input is unambiguous and accepted."""
        y = (np.arange(30, dtype=float) + 1.0).reshape(-1, 1)
        sol = ets(y, model="ANN")
        assert sol.n_obs == 30

    def test_damped_true_with_trend_n_wildcard_raises(self):
        """model='ZNN' + damped=True is a forbidden combination (as in R),
        not a silent no-op."""
        y = np.arange(40, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="trend component"):
            ets(y, model="ZNN", damped=True)

    def test_period_over_24_fully_specified_raises(self):
        y = np.arange(120, dtype=float) + 1.0
        with pytest.raises(ValidationError, match="period <= 24"):
            ets(y, model="MAM", period=25)

    def test_tiny_series_aicc_selection_raises_with_remedy(self):
        """n=3: every candidate fits but AICc is infinite -> a clear
        ValidationError naming the remedy, not a false ConvergenceError."""
        with pytest.raises(ValidationError, match="finite aicc"):
            ets(np.array([1.0, 2.0, 3.0]))

    def test_tiny_series_selectable_with_aic(self):
        """The remedy works: ic='aic' selects at n=3."""
        sol = ets(np.array([1.0, 2.0, 3.0]), ic="aic")
        assert "selection" in sol.info

    def test_n5_selects_under_default_aicc(self):
        """n=5 is the smallest series the default ets(y) can select on."""
        sol = ets(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))
        assert np.isfinite(sol.aicc)
