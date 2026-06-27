"""Characterization of the CPU ``polr`` slope-ridge behaviour.

A secondary claim attached to the issue #8 GPU collapse was that the *CPU* slope
ridge (``methods/polr._slope_ridge``) over-shrinks the dominant slope on
strongly-imbalanced / quasi-separated ordinals, producing over-dispersed
imputations relative to R ``mice``. That claim was investigated end-to-end
against an R ``mice`` oracle on two real surveys (GSS, CSES) and does **not**
hold — there is no over-shrinkage bug, and lowering the ridge makes the
agreement with R *worse*, not better. What the receipts established, and what
this module pins so the conclusion is not silently re-litigated:

  * **In-sample marginal fidelity.** The ridge penalises the *slopes* only; the
    thresholds are unpenalised, so their score equations are exactly zero at the
    optimum. Hence the average predicted category probability over the fitted
    rows equals the observed marginal — even under (quasi-)complete separation,
    where the slope is large. A regression that penalised the thresholds, or
    otherwise distorted the fit, would break this identity.

  * **Posterior-draw stability.** On both well-identified and separated ordinals
    the observed information (hence ``vcov``) stays positive definite and the
    natural-coordinate posterior draw is tight around the point estimate — the
    drawn slopes stay within a small factor of the MLE and the drawn
    dominant-category probability has tiny spread. The over-dispersion is NOT a
    near-singular-``vcov`` draw blow-up.

  * **Finite, bounded fit under separation.** The ridge's actual job: keep the
    proportional-odds fit finite and the information PD when a continuous
    predictor (nearly) perfectly orders a sparse extreme category. The slope is
    large (the data are separated) but bounded, not divergent.

  * **Agreement with the well-posed answer.** On a well-identified ordinal with
    MCAR missingness — the regime where pystatistics, R, and the truth all
    agree (CSES matched R to three decimals) — the imputed-category marginal
    tracks the observed marginal closely.

The genuine pystatistics/R differences on separated MNAR survey columns are
structural (R's per-column collinearity pruning, mice's own ridge, and R's
*improper* point-estimate ``polr`` draw) and do not favour either engine; they
are not a defect in this ridge. See also ``test_gpu_polr_separation.py`` for the
primary (GPU) fix.
"""

from __future__ import annotations

import numpy as np

from pystatistics.ordinal import polr
from pystatistics.ordinal._likelihood import _cumulative_probs_vectorized
from pystatistics.regression.families import LogitLink
from pystatistics.mice.methods._draw import mvn_draw
from pystatistics.mice.methods.polr import PolrMethod, _slope_ridge

# Large but finite ceiling for the separated slope (the legitimate fit is ~50;
# an unregularised divergence would be orders of magnitude larger).
_FINITE_BOUND = 1.0e3


def _well_identified_ordinal(seed: int, n: int = 4000, K: int = 4):
    """A proper proportional-odds DGP: ``latent = 0.9 x1 - 0.6 x2 + logistic``
    cut into K balanced levels. Slopes are modest and every category is well
    populated, so the fit is well-identified (the CSES-like regime)."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.standard_normal(n), rng.standard_normal(n)
    latent = 0.9 * x1 - 0.6 * x2 + rng.logistic(size=n)
    cuts = np.quantile(latent, np.linspace(0, 1, K + 1)[1:-1])
    y = np.digitize(latent, cuts).astype(np.intp)
    return y, np.column_stack([x1, x2]), K


def _separated_ordinal(seed: int, n: int = 4000, K: int = 4):
    """A continuous predictor that (nearly) perfectly orders the target, with a
    sparse extreme category (top cut at the 0.97 quantile) — the GSS failure
    signature. The proportional-odds MLE is unbounded; the ridge keeps it
    finite. ``z`` is an inert second predictor."""
    rng = np.random.default_rng(seed)
    x, z = rng.standard_normal(n), rng.standard_normal(n)
    cuts = np.quantile(x, [0.30, 0.60, 0.97])[: K - 1]
    y = np.digitize(x, cuts).astype(np.intp)
    for lv in range(K):  # guarantee every level is present among observed
        if not np.any(y == lv):
            y[lv] = lv
    return y, np.column_stack([x, z]), K


def _fit(y, X):
    """polr fit with the exact MICE slope ridge."""
    return polr(y, X, l2=_slope_ridge(X.astype(np.float64)))


def _predicted_marginal(fit, X, K):
    """Average predicted category probability over the rows of ``X``."""
    eta = X @ np.asarray(fit.coefficients)
    probs = _cumulative_probs_vectorized(
        np.asarray(fit.threshold_values), eta, LogitLink(), K)
    return probs.mean(axis=0)


def _observed_marginal(y, K):
    return np.bincount(y, minlength=K) / y.size


def _draw_stats(fit, X, K, dom, n_draws: int = 500):
    """Posterior-draw spread of the dominant predicted prop and the max |slope|."""
    alpha = np.asarray(fit.threshold_values)
    beta = np.asarray(fit.coefficients)
    vcov = np.asarray(fit.vcov)
    mean = np.concatenate([alpha, beta])
    rng = np.random.default_rng(0)
    doms, smax = [], []
    for _ in range(n_draws):
        th = mvn_draw(mean, vcov, rng)
        a_s, b_s = th[: alpha.size], th[alpha.size:]
        pm = _cumulative_probs_vectorized(a_s, X @ b_s, LogitLink(), K).mean(axis=0)
        doms.append(pm[dom])
        smax.append(np.abs(b_s).max())
    min_eig = float(np.linalg.eigvalsh(vcov).min())
    return min_eig, np.array(doms), np.array(smax), float(np.abs(beta).max())


class TestInSampleMarginalFidelity:
    """The fit reproduces the observed marginal in-sample, in BOTH regimes —
    the threshold-stationarity property the ridge must not break."""

    def test_well_identified(self):
        y, X, K = _well_identified_ordinal(seed=0)
        pm = _predicted_marginal(_fit(y, X), X, K)
        np.testing.assert_allclose(pm, _observed_marginal(y, K), atol=5e-3)

    def test_separated_still_faithful(self):
        """Holds to ~1e-8 even though the separated slope is ~50: the thresholds
        are unpenalised, so the marginal identity survives separation."""
        y, X, K = _separated_ordinal(seed=0)
        pm = _predicted_marginal(_fit(y, X), X, K)
        np.testing.assert_allclose(pm, _observed_marginal(y, K), atol=5e-3)


class TestDrawStability:
    """The natural-coordinate posterior draw is well-conditioned and tight — the
    over-dispersion is not a near-singular-vcov draw blow-up."""

    def test_well_identified(self):
        y, X, K = _well_identified_ordinal(seed=0)
        fit = _fit(y, X)
        dom = int(np.argmax(_observed_marginal(y, K)))
        min_eig, doms, smax, bmax = _draw_stats(fit, X, K, dom)
        assert min_eig > 0.0                       # vcov positive definite
        assert doms.std() < 0.05                   # tight dominant-prop draws
        assert smax.max() < 2.0 * bmax             # drawn slopes near the MLE

    def test_separated(self):
        y, X, K = _separated_ordinal(seed=0)
        fit = _fit(y, X)
        dom = int(np.argmax(_observed_marginal(y, K)))
        min_eig, doms, smax, bmax = _draw_stats(fit, X, K, dom)
        assert min_eig > 0.0
        assert doms.std() < 0.05
        assert smax.max() < 2.0 * bmax


class TestSeparatedFitFiniteAndBounded:
    """Under (quasi-)separation the ridge keeps the fit finite, bounded, and the
    information PD — its actual purpose."""

    def test_fit_finite_bounded_pd(self):
        y, X, K = _separated_ordinal(seed=0)
        fit = _fit(y, X)                            # must not raise
        beta = np.asarray(fit.coefficients)
        assert np.all(np.isfinite(beta))
        assert np.abs(beta).max() < _FINITE_BOUND
        assert float(np.linalg.eigvalsh(np.asarray(fit.vcov)).min()) > 0.0


class TestWellIdentifiedMatchesWellPosedAnswer:
    """On a well-identified ordinal with MCAR missingness — where pystatistics,
    R, and the truth coincide — the imputed marginal tracks the observed
    marginal (the R-agreement regime, validated against R on CSES)."""

    def test_mcar_imputed_marginal_tracks_observed(self):
        y, X, K = _well_identified_ordinal(seed=0)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y))
        n_mis = int(0.3 * len(y))
        mis, obs = idx[:n_mis], idx[n_mis:]
        imp = PolrMethod().impute(
            y[obs], X[obs], X[mis], np.random.default_rng(1))
        p_imp = np.bincount(np.rint(imp).astype(int), minlength=K) / imp.size
        tv = 0.5 * float(np.abs(p_imp - _observed_marginal(y, K)).sum())
        assert tv < 0.05, f"TV(imputed, observed) = {tv:.4f}"
