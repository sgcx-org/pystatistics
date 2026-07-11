"""Tests for the GLM-family analytic smoothing-parameter gradient.

Per module test policy — normal cases: the analytic gradient matches central
finite differences of the ACTUAL criterion (evaluated through the inner
P-IRLS fit) across families, links (canonical AND non-canonical) and
criteria; the analytic search lands on the same optimum the finite-difference
search found; selection cost no longer scales as ``2m+1`` inner fits per
outer step. Edge cases: rank-deficient design (concurvity), near-separation
(mu clamps active), lambda extremes, single smooth, singular Newton systems
(warned Fisher fallback), multimodal inner fits (branch resolution).
Failure cases: fail-loud paths raise the documented exceptions.
Determinism: identical inputs give bit-identical gradients.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pystatistics.core.exceptions import ConvergenceError
from pystatistics.gam._basis import build_design
from pystatistics.gam._criteria import (
    gcv_score,
    initial_log_lambdas,
    reml_score,
    ubre_score,
)
from pystatistics.gam._edf import influence_matrix, total_edf
from pystatistics.gam._gradient_glm import (
    _implicit_derivatives,
    gcv_gradient_glm,
    reml_gradient_glm,
    ubre_gradient_glm,
)
from pystatistics.gam._pirls import (
    PenaltyRoot,
    PirlsFit,
    fit_fixed_lambda,
    make_penalty_roots,
)
from pystatistics.gam._smooth import s
from pystatistics.regression.families import (
    Binomial,
    GammaFamily,
    Gaussian,
    NegativeBinomial,
    Poisson,
    resolve_family,
)

_TOL_INNER = 1e-13
_MAX_ITER = 400


def _family(key):
    return {
        "poisson": lambda: resolve_family("poisson"),
        "poisson-sqrt": lambda: Poisson(link="sqrt"),
        "binomial": lambda: resolve_family("binomial"),
        "binomial-probit": lambda: Binomial(link="probit"),
        "binomial-cloglog": lambda: Binomial(link="cloglog"),
        "Gamma-log": lambda: GammaFamily(link="log"),
        "Gamma-inverse": lambda: GammaFamily(),
        "gaussian-log": lambda: Gaussian(link="log"),
        "nb": lambda: NegativeBinomial(theta=3.0),
    }[key]()


def _problem(family_key, m=2, n=350, seed=7):
    """Deterministic (family-appropriate) test problem with m cr smooths."""
    rng = np.random.default_rng(seed)
    x1 = np.sort(rng.uniform(0.0, 1.0, n))
    x2 = rng.uniform(0.0, 1.0, n)
    f = 1.4 * np.sin(2 * np.pi * x1)
    if m >= 2:
        f = f + np.cos(2 * np.pi * x2)
    fam = _family(family_key)
    if family_key == "poisson-sqrt":
        # sqrt link: the true eta = sqrt(mu) must stay well positive
        # (eta < 0 leaves the link's domain and P-IRLS cannot converge).
        y = rng.poisson((3.0 + f) ** 2).astype(float)
    elif family_key.startswith("poisson"):
        y = rng.poisson(np.exp(f - f.mean() + 1.0)).astype(float)
    elif family_key.startswith("binomial"):
        p = 1.0 / (1.0 + np.exp(-2.0 * f))
        y = rng.binomial(1, p).astype(float)
    elif family_key.startswith("Gamma"):
        mu = np.exp(0.8 * f + 1.0)
        y = rng.gamma(shape=4.0, scale=mu / 4.0)
    elif family_key == "gaussian-log":
        y = np.exp(0.6 * f) * (1.0 + rng.normal(0.0, 0.08, n))
    elif family_key == "nb":
        mu = np.exp(f - f.mean() + 1.0)
        y = rng.negative_binomial(3.0, 3.0 / (3.0 + mu)).astype(float)
    else:  # pragma: no cover - config error in the test itself
        raise ValueError(family_key)
    smooths = [s("x1", k=10, bs="cr")]
    data = {"x1": x1}
    if m >= 2:
        smooths.append(s("x2", k=8, bs="cr"))
        data["x2"] = x2
    X_aug, built = build_design(np.ones((n, 1)), data, smooths)
    roots = make_penalty_roots([b.S_blocks[0] for b in built],
                               [b.block for b in built])
    return y, X_aug, roots, fam


def _criterion(rho, y, X, roots, fam, method):
    lam = np.exp(rho)
    fit = fit_fixed_lambda(y, X, roots, lam, fam, _TOL_INNER, _MAX_ITER)
    n = y.shape[0]
    if method == "REML":
        return reml_score(fit, y, X, fam, roots, lam)
    edf = total_edf(influence_matrix(fit.R, fit.R_x, fit.piv, fit.rank))
    if method == "UBRE":
        return ubre_score(fit.deviance, n, edf, 1.0)
    return gcv_score(fit.deviance, n, edf)


def _analytic(rho, y, X, roots, fam, method):
    lam = np.exp(rho)
    fit = fit_fixed_lambda(y, X, roots, lam, fam, _TOL_INNER, _MAX_ITER)
    if method == "REML":
        return reml_gradient_glm(fit, roots, lam, y, X, fam)
    if method == "UBRE":
        return ubre_gradient_glm(fit, roots, lam, y, X, fam)
    edf = total_edf(influence_matrix(fit.R, fit.R_x, fit.piv, fit.rank))
    return gcv_gradient_glm(fit, roots, lam, y, X, fam, edf)


def _fd(rho, y, X, roots, fam, method, eps=1e-5):
    g = np.empty_like(rho)
    for j in range(len(rho)):
        rp, rm = rho.copy(), rho.copy()
        rp[j] += eps
        rm[j] -= eps
        g[j] = (_criterion(rp, y, X, roots, fam, method)
                - _criterion(rm, y, X, roots, fam, method)) / (2 * eps)
    return g


def _assert_gradient_matches(y, X, roots, fam, method,
                             offsets=(0.0, -3.0, 2.5), rtol=1e-4):
    rho0 = initial_log_lambdas(X, roots)
    for off in offsets:
        rho = rho0 + np.asarray(off, dtype=float)
        ga = _analytic(rho, y, X, roots, fam, method)
        gf = _fd(rho, y, X, roots, fam, method)
        # Relative error with an absolute floor tied to the gradient's own
        # scale: a coordinate whose true gradient is ~0 (near its optimum)
        # must not be judged against pure FD noise.
        floor = rtol * max(float(np.max(np.abs(gf))), 1e-6)
        denom = np.maximum(np.abs(gf), floor)
        rel = np.abs(ga - gf) / denom
        assert rel.max() < rtol, (method, off, ga, gf)


# ---------------------------------------------------------------------------
# Normal cases: gradient vs criterion finite differences
# ---------------------------------------------------------------------------

class TestGradientMatchesFiniteDifference:
    """Canonical links (Fisher == Newton) and non-canonical links (where the
    full-Newton implicit derivative is REQUIRED — a Fisher-weight shortcut is
    off by up to several percent on probit / Gamma-log)."""

    @pytest.mark.parametrize("family_key,method", [
        ("poisson", "UBRE"),
        ("poisson", "REML"),
        ("poisson-sqrt", "UBRE"),
        ("binomial", "UBRE"),
        ("binomial", "REML"),
        ("binomial-probit", "UBRE"),
        ("binomial-probit", "REML"),
        ("binomial-cloglog", "UBRE"),
        ("binomial-cloglog", "REML"),
        ("Gamma-log", "GCV"),
        ("Gamma-inverse", "GCV"),
        ("gaussian-log", "GCV"),
        ("nb", "UBRE"),
        ("nb", "REML"),
    ])
    def test_two_smooths(self, family_key, method):
        y, X, roots, fam = _problem(family_key, m=2)
        _assert_gradient_matches(y, X, roots, fam, method)

    def test_single_smooth(self):
        y, X, roots, fam = _problem("poisson", m=1)
        _assert_gradient_matches(y, X, roots, fam, "UBRE")

    def test_lambda_extremes(self):
        """Near the search bounds (rho0 +/- 8): the gradient must stay exact
        where the optimizer probes hardest."""
        y, X, roots, fam = _problem("poisson", m=2)
        _assert_gradient_matches(y, X, roots, fam, "REML",
                                 offsets=(-8.0, 8.0))

    def test_anisotropic_lambda(self):
        """Strongly anisotropic rho (one smooth near-interpolating, the
        other near-null): cross-smooth terms in the gradient must hold, not
        just isotropic shifts of the start point."""
        y, X, roots, fam = _problem("poisson", m=2)
        _assert_gradient_matches(y, X, roots, fam, "REML",
                                 offsets=([4.0, -4.0], [-4.0, 4.0]))
        _assert_gradient_matches(y, X, roots, fam, "UBRE",
                                 offsets=([4.0, -4.0],))

    def test_deterministic(self):
        """Identical inputs -> bit-identical gradient (Rule 6: the internal
        central differences use a fixed, documented step)."""
        y, X, roots, fam = _problem("poisson", m=2)
        rho = initial_log_lambdas(X, roots)
        g1 = _analytic(rho, y, X, roots, fam, "REML")
        g2 = _analytic(rho, y, X, roots, fam, "REML")
        assert np.array_equal(g1, g2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestGradientEdgeCases:

    @pytest.mark.parametrize("method", ["UBRE", "REML"])
    def test_rank_deficient_concurvity(self, method):
        """s(x1) + s(x2) with x2 == x1: a column is dropped by the pivoted
        solve; the gradient must agree with FD on the kept coordinates."""
        rng = np.random.default_rng(11)
        n = 300
        x1 = np.sort(rng.uniform(0.0, 1.0, n))
        f = 1.4 * np.sin(2 * np.pi * x1)
        y = rng.poisson(np.exp(f - f.mean() + 1.0)).astype(float)
        fam = resolve_family("poisson")
        X_aug, built = build_design(
            np.ones((n, 1)), {"x1": x1, "x2": x1.copy()},
            [s("x1", k=10, bs="cr"), s("x2", k=10, bs="cr")])
        roots = make_penalty_roots([b.S_blocks[0] for b in built],
                                   [b.block for b in built])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # rank-deficiency warning
            _assert_gradient_matches(y, X_aug, roots, fam, method,
                                     offsets=(0.0, 2.0))

    def test_near_separation_mu_clamps(self):
        """Quasi-separable binomial: fitted probabilities hit the domain
        clamp; the gradient must still match FD (the criterion both sides
        of the FD step sees the same clamped fits)."""
        rng = np.random.default_rng(3)
        n = 200
        x1 = np.sort(rng.uniform(0.0, 1.0, n))
        y = (x1 > 0.5).astype(float)
        flip = rng.choice(n, 4, replace=False)
        y[flip] = 1.0 - y[flip]
        fam = resolve_family("binomial")
        X_aug, built = build_design(np.ones((n, 1)), {"x1": x1},
                                    [s("x1", k=10, bs="cr")])
        roots = make_penalty_roots([b.S_blocks[0] for b in built],
                                   [b.block for b in built])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # separation warning
            _assert_gradient_matches(y, X_aug, roots, fam, "UBRE",
                                     offsets=(0.0, -4.0))


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

class TestGradientFailureCases:

    def test_pirls_nonfinite_working_response_raises_convergence_error(self):
        """The _pirls fail-loud path must raise the DOCUMENTED
        ConvergenceError — before the core exceptions fix it raised
        TypeError (ConvergenceError demanded an iterations argument the
        message-only call sites never passed)."""
        from pystatistics.gam._pirls import reduce_wls

        X = np.ones((4, 2))
        w = np.ones(4)
        z = np.array([1.0, 2.0, np.inf, 3.0])
        with pytest.raises(ConvergenceError, match="non-finite"):
            reduce_wls(X, w, z)

    def test_convergence_error_message_only_construction(self):
        """Core contract: ConvergenceError is constructible with a message
        alone (iterations optional, None when not applicable) and still
        accepts the full diagnostic signature."""
        e = ConvergenceError("just a message")
        assert e.iterations is None
        e2 = ConvergenceError("msg", 42, final_change=0.1,
                              reason="max_iterations", threshold=1e-8)
        assert e2.iterations == 42

    def test_singular_newton_system_warns_and_falls_back(self):
        """A numerically singular X'WnX + S_lambda must WARN and fall back
        to the always-defined Fisher implicit solve — never feed inf/nan to
        the optimizer (LAPACK getrf only WARNS on a zero pivot, so the
        module gates on finiteness) and never abort a fit mgcv completes
        (adversarially verified reachable: binomial-probit n=60, where
        mgcv's own optimum sits exactly where the Newton eigenvalues
        collapse)."""
        n, p = 6, 2
        X = np.ones((n, p))                      # duplicate columns
        y = np.linspace(0.5, 1.5, n)
        fam = resolve_family("poisson")
        # Zero penalty (rank-0 root) so S_lambda cannot rescue the
        # singularity; the fake fit CLAIMS full rank so no column is dropped.
        root = PenaltyRoot(rows=np.zeros((1, p)), rank=0, block=(0, p),
                           logdet_pos=0.0, group=0)
        fit = PirlsFit(
            beta=np.array([0.1, 0.1]), mu=np.exp(X @ [0.1, 0.1]),
            eta=X @ [0.1, 0.1], w=np.ones(n), deviance=1.0, penalty=0.0,
            R=np.eye(p), R_x=np.eye(p), piv=np.arange(p), rank=p,
            n_iter=1, converged=True,
        )
        with pytest.warns(UserWarning, match="Fisher"):
            out = _implicit_derivatives(fit, [root], np.array([1.0]), y, X,
                                        fam)
        deta = out[-1]
        assert np.all(np.isfinite(deta))

    def test_reml_nonpd_newton_hessian_warns_and_falls_back(self,
                                                            monkeypatch):
        """A non-positive-definite Newton Hessian in the non-canonical REML
        determinant must WARN and fall back to the Fisher determinant —
        finite score, never a crash or an unexplained -inf. (Forced by
        injecting negative Newton weights; the PD test is a Cholesky
        attempt, NOT a slogdet sign — an even-dimensional negative-definite
        matrix has positive determinant.)"""
        import pystatistics.gam._gradient_glm as gg

        y, X, roots, fam = _problem("binomial-probit", m=1, n=80)
        lam = np.array([1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fit_fixed_lambda(y, X, roots, lam, fam, _TOL_INNER,
                                   _MAX_ITER)
        real = gg._eta_derivatives

        def negated(family, yy, eta):
            u, omega, w_n, om_n = real(family, yy, eta)
            return u, omega, -np.abs(w_n) - 1.0, om_n  # force indefinite

        monkeypatch.setattr(gg, "_eta_derivatives", negated)
        with pytest.warns(UserWarning, match="not positive definite"):
            v = gg.reml_logdet_glm(fit, roots, lam, y, X, fam)
        assert np.isfinite(v)
        # ... and the gradient makes the SAME decision (Fisher fallback),
        # staying finite rather than differentiating a different criterion.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = gg.reml_gradient_glm(fit, roots, lam, y, X, fam)
        assert np.all(np.isfinite(g))

    def test_reml_score_matches_mgcv_noncanonical_fixed_sp(self):
        """The REML SCORE at fixed sp pins mgcv's Newton-weight Laplace
        determinant on a non-canonical link (probit): the Fisher
        determinant was 0.034 off here — adversarially proven to be
        exactly 0.5*(log|A_Fisher| - log|A_Newton|)."""
        from pystatistics.gam import gam

        y, x1, x2 = self._e2e_probit_data()
        sol = gam(y, smooths=[s("x1", k=10, bs="cr"), s("x2", k=8, bs="cr")],
                  smooth_data={"x1": x1, "x2": x2},
                  family=Binomial(link="probit"), method="REML",
                  sp=[37.993517, 107.51899])
        # mgcv reml at its selected sp on this exact data: 147.09843235
        assert abs(sol.reml_score - 147.09843235) < 1e-4, sol.reml_score

    @staticmethod
    def _e2e_probit_data():
        rng = np.random.default_rng(20260709)
        n = 300
        x1 = np.sort(rng.uniform(0.0, 1.0, n))
        x2 = rng.uniform(0.0, 1.0, n)
        f = 1.4 * np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x2)
        rng.poisson(np.exp(f - f.mean() + 1.0))  # keep RNG stream aligned
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-2.0 * f))).astype(float)
        return y, x1, x2

    def test_singular_newton_region_completes_via_public_gam(self):
        """REGRESSION (adversarial review, confirmed major): binomial-probit
        data whose small-lambda optimum drives the Newton system singular
        (mgcv fits it fine: sp~[2e-7, 5e-6]) must complete through public
        gam() — gracefully degraded/warned, never an uncaught crash."""
        from pystatistics.gam import gam

        rng = np.random.default_rng(14)
        n = 60
        x1 = np.sort(rng.uniform(0.0, 1.0, n))
        x2 = rng.uniform(0.0, 1.0, n)
        f = 2.5 * (np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x2))
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-f))).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = gam(y, smooths=[s("x1", k=12, bs="cr"),
                                  s("x2", k=10, bs="cr")],
                      smooth_data={"x1": x1, "x2": x2},
                      family=Binomial(link="probit"), method="GCV")
        assert np.all(np.isfinite(sol.fitted_values))

    def test_multimodal_inner_fit_reports_search_branch(self):
        """REGRESSION (adversarial review, confirmed major): at near-zero
        penalty the inner P-IRLS problem is multimodal — the warm-chained
        search tracks the deep branch (GCV ~4.4, matching mgcv's 4.84 on
        this exact data) while a fresh refit at the same lambdas lands on a
        shallow branch (GCV ~38) — which was silently reported. The final
        fit must continue the search's winning branch."""
        from pystatistics.gam import gam

        rng = np.random.default_rng(16)
        n = 60
        x1 = np.sort(rng.uniform(0.0, 1.0, n))
        x2 = rng.uniform(0.0, 1.0, n)
        f = 5.0 * (np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x2))
        y = np.exp(0.6 * f) * (1.0 + rng.normal(0.0, 0.3, n))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = gam(y, smooths=[s("x1", k=12, bs="cr"),
                                  s("x2", k=10, bs="cr")],
                      smooth_data={"x1": x1, "x2": x2},
                      family=Gaussian(link="log"), method="GCV")
        # Deep branch: GCV ~4.4 (mgcv: 4.84). The silent-wrong regression
        # reported 38.2; anything above ~6 means the branch was lost again.
        assert sol.gcv < 6.0, sol.gcv


# ---------------------------------------------------------------------------
# Selection equivalence + cost
# ---------------------------------------------------------------------------

class TestSelectionEquivalence:
    """select_lambdas (analytic jac) lands on the same optimum a
    finite-difference L-BFGS-B search finds — the estimates are unchanged;
    only the route to the optimum got cheaper."""

    @pytest.mark.parametrize("family_key,method", [
        ("poisson", "REML"),
        ("binomial", "GCV"),      # GCV.Cp semantics -> UBRE internally
        ("Gamma-log", "GCV"),
    ])
    def test_same_optimum_as_fd_search(self, family_key, method):
        from scipy.optimize import minimize

        from pystatistics.gam import _criteria as crit

        y, X, roots, fam = _problem(family_key, m=2, n=400)

        crit_key = method
        if method != "REML":
            crit_key = "UBRE" if fam.dispersion_is_fixed else "GCV"

        def obj(rho):
            return _criterion(rho, y, X, roots, fam, crit_key)

        r0 = initial_log_lambdas(X, roots)
        bounds = [(v - 15.0, v + 15.0) for v in r0]
        ref = max(abs(obj(r0)), 1e-300)
        fd = minimize(lambda r: obj(r) / ref, r0, method="L-BFGS-B",
                      bounds=bounds,
                      options={"maxiter": 200, "ftol": 1e-12,
                               "gtol": 1e-9, "eps": 1e-4})
        lam_fd = np.exp(fd.x)

        lam_an, converged, _mu = crit.select_lambdas(
            y, X, roots, fam, method, 1e-8, 200)
        assert converged
        rel = np.abs(lam_an - lam_fd) / (np.abs(lam_fd) + 1e-300)
        assert rel.max() < 5e-3, (family_key, method, lam_an, lam_fd)

    @pytest.mark.parametrize("family_key,method,edf_ref,sp_ref", [
        # mgcv 1.9-3 references computed on EXACTLY the data _e2e_data()
        # generates (exported to CSV, fitted with
        # gam(y ~ s(x1,k=10,bs='cr') + s(x2,k=8,bs='cr'), method=REML/GCV.Cp)).
        ("poisson", "REML", 13.55076174, (63.922836, 78.870338)),
        ("poisson", "GCV", 11.96412854, (176.11368, 161.06868)),
        ("binomial", "REML", 9.74656876, (12.828351, 35.272794)),
        ("binomial", "GCV", 9.14744118, (13.308508, 78.480786)),
        # NON-canonical REML: pins the Newton-weight Laplace determinant
        # (the Fisher determinant selected edf ~8e-3 away from mgcv here).
        ("binomial-probit", "REML", 9.77192330, (37.993517, 107.51899)),
    ])
    def test_gam_end_to_end_matches_mgcv(self, family_key, method,
                                         edf_ref, sp_ref):
        """Public gam() free selection pins mgcv's selected smoothness for
        GLM families — the regression this file exists to prevent is a
        gradient bug silently shifting selection."""
        from pystatistics.gam import gam

        y, x1, x2 = self._e2e_data(family_key.split("-")[0])
        sol = gam(y, smooths=[s("x1", k=10, bs="cr"), s("x2", k=8, bs="cr")],
                  smooth_data={"x1": x1, "x2": x2},
                  family=_family(family_key) if "-" in family_key
                  else family_key,
                  method=method)
        assert sol.outer_converged
        assert abs(sol.total_edf - edf_ref) < 2e-3, sol.total_edf
        sp_rel = np.abs(np.asarray(sol.lambdas) - sp_ref) / np.asarray(sp_ref)
        assert sp_rel.max() < 5e-3, sol.lambdas

    @staticmethod
    def _e2e_data(family_key):
        rng = np.random.default_rng(20260709)
        n = 300
        x1 = np.sort(rng.uniform(0.0, 1.0, n))
        x2 = rng.uniform(0.0, 1.0, n)
        f = 1.4 * np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x2)
        y_pois = rng.poisson(np.exp(f - f.mean() + 1.0)).astype(float)
        y_bin = rng.binomial(1, 1.0 / (1.0 + np.exp(-2.0 * f))).astype(float)
        return (y_pois if family_key == "poisson" else y_bin), x1, x2

    def test_nb_estimated_theta_gcv_refused(self):
        """nb() with estimated theta under method='GCV' must be REFUSED:
        profiling the UBRE score over theta is structurally degenerate (the
        NB deviance shrinks monotonically as theta -> 0, so the profiled
        optimum collapses to theta ~ 0.04 on well-behaved theta=3 data).
        Fail loud, never a silently degenerate dispersion."""
        from pystatistics.core.exceptions import ValidationError
        from pystatistics.gam import gam

        rng = np.random.default_rng(17)
        n = 300
        x1 = np.sort(rng.uniform(0.0, 1.0, n))
        f = 1.4 * np.sin(2 * np.pi * x1)
        mu = np.exp(f - f.mean() + 1.0)
        y = rng.negative_binomial(3.0, 3.0 / (3.0 + mu)).astype(float)
        with pytest.raises(ValidationError, match="REML"):
            gam(y, smooths=[s("x1", k=10, bs="cr")],
                smooth_data={"x1": x1}, family="nb", method="GCV")
        # ... while a FIXED theta remains perfectly valid under GCV/UBRE.
        sol = gam(y, smooths=[s("x1", k=10, bs="cr")],
                  smooth_data={"x1": x1},
                  family=NegativeBinomial(theta=3.0), method="GCV")
        assert sol.outer_converged

    def test_inner_fit_count_does_not_scale_as_2m_plus_1(self, monkeypatch):
        """The 4.6.x finite-difference search needed (2m+1) inner fits per
        outer step (~46 at m=3 on this problem); the analytic path needs one
        per step (~13). Guard the regression with generous headroom."""
        from pystatistics.gam import _criteria as crit

        rng = np.random.default_rng(5)
        n = 400
        xs = [np.sort(rng.uniform(0.0, 1.0, n)),
              rng.uniform(0.0, 1.0, n), rng.uniform(0.0, 1.0, n)]
        f = (1.5 * np.sin(3 * np.pi * xs[0]) + np.cos(2 * np.pi * xs[1])
             - 0.8 * np.sin(2 * np.pi * xs[2]))
        y = rng.poisson(np.exp(f - f.mean() + 1.2)).astype(float)
        fam = resolve_family("poisson")
        data = {f"x{i + 1}": xs[i] for i in range(3)}
        X_aug, built = build_design(
            np.ones((n, 1)), data,
            [s(f"x{i + 1}", k=10, bs="cr") for i in range(3)])
        roots = make_penalty_roots([b.S_blocks[0] for b in built],
                                   [b.block for b in built])

        calls = {"n": 0}
        real_fit = crit.fit_fixed_lambda

        def counting_fit(*args, **kwargs):
            calls["n"] += 1
            return real_fit(*args, **kwargs)

        monkeypatch.setattr(crit, "fit_fixed_lambda", counting_fit)
        _, converged, _mu = crit.select_lambdas(
            y, X_aug, roots, fam, "REML", 1e-8, 200)
        assert converged
        assert calls["n"] <= 30, calls["n"]
