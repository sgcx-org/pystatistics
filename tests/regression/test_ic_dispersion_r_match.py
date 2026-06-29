"""
Information-criteria correctness against R, for the 4.3.2 fixes.

Two model-fit statistics had diverged from R while coefficients/SEs/p-values
matched:

1. Gamma and Gaussian BIC mis-penalized the ML-estimated dispersion. Their AIC
   correctly counts the dispersion (σ² / shape) as a free parameter, but BIC
   re-penalized it with the AIC constant (2) instead of log(n), leaving BIC off
   by exactly (log(n) - 2). Fixed by recording the AIC's parameter count in
   ``ic_param_count`` so BIC penalizes the dispersion with log(n) too.

2. Binomial deviance / log-likelihood clipped fitted mu to [1e-10, 1-1e-10],
   coarser than R's binomial()$dev.resids (machine epsilon). For fitted
   probabilities between eps and 1e-10 this under-counted the deviance (and the
   AIC/BIC that inherit it). Fixed by tightening the bound to machine epsilon.

Reference values were produced by R 4.x (glm / AIC / BIC / binomial()) and are
frozen as literals below. The datasets are embedded so the tests need no network
and no R at run time.
"""

import numpy as np
import pytest

from pystatistics.regression import fit, GammaFamily, Gaussian, Binomial
from pystatistics.regression.families import Poisson


# =====================================================================
# Bug 1 — Gamma / Gaussian BIC counts the estimated dispersion at log(n)
# =====================================================================

# Deterministic 30-row dataset (positive continuous response). Used for both the
# Gamma(log) and Gaussian fits. Frozen alongside the R reference statistics.
_X1 = [0.3047, -1.04, 0.7505, 0.9406, -1.951, -1.3022, 0.1278, -0.3162,
       -0.0168, -0.853, 0.8794, 0.7778, 0.066, 1.1272, 0.4675, -0.8593,
       0.3688, -0.9589, 0.8785, -0.0499, -0.1849, -0.6809, 1.2225, -0.1545,
       -0.4283, -0.3521, 0.5323, 0.3654, 0.4127, 0.4308]
_X2 = [2.1416, -0.4064, -0.5122, -0.8138, 0.616, 1.129, -0.1139, -0.8402,
       -0.8245, 0.6506, 0.7433, 0.5432, -0.6655, 0.2322, 0.1167, 0.2187,
       0.8714, 0.2236, 0.6789, 0.0676, 0.2891, 0.6313, -1.4572, -0.3197,
       -0.4704, -0.6389, -0.2751, 1.4949, -0.8658, 0.9683]
_Y = [0.1476, 1.1585, 3.587, 1.9251, 0.9539, 0.1754, 0.6636, 1.736, 1.2308,
      1.2592, 2.1835, 1.186, 0.5689, 1.3652, 2.2721, 1.4681, 0.8663, 0.1573,
      0.4547, 1.7721, 0.8596, 0.6482, 1.5164, 0.6111, 2.4026, 1.9886, 1.1202,
      1.0733, 2.5031, 3.4483]


def _disp_design():
    X = np.column_stack([np.ones(len(_X1)), _X1, _X2])
    return X, np.asarray(_Y, dtype=float)


# R: glm(y ~ x1 + x2, family = Gamma(link="log")); AIC(); BIC()
_R_GAMMA_DEV = 12.5262209700819
_R_GAMMA_AIC = 72.8011030741606
_R_GAMMA_BIC = 78.4058926008092

# R: glm(y ~ x1 + x2, family = gaussian()); AIC(); BIC()
_R_GAUSS_DEV = 18.0840497569805
_R_GAUSS_AIC = 77.9513001683726
_R_GAUSS_BIC = 83.5560896950212


class TestGammaDispersionIC:
    """Gamma BIC must count the estimated shape with log(n), like R."""

    def test_aic_matches_r(self):
        X, y = _disp_design()
        r = fit(X, y, family=GammaFamily(link='log'))
        assert r.aic == pytest.approx(_R_GAMMA_AIC, rel=0, abs=1e-6)

    def test_bic_matches_r(self):
        X, y = _disp_design()
        r = fit(X, y, family=GammaFamily(link='log'))
        # Pre-4.3.2 this was off by exactly (log(n) - 2).
        assert r.bic == pytest.approx(_R_GAMMA_BIC, rel=0, abs=1e-6)

    def test_deviance_matches_r(self):
        X, y = _disp_design()
        r = fit(X, y, family=GammaFamily(link='log'))
        assert r.deviance == pytest.approx(_R_GAMMA_DEV, rel=0, abs=1e-6)

    def test_bic_counts_dispersion_param(self):
        """BIC = AIC - 2k + k·log(n) with k = rank + 1 (dispersion included)."""
        X, y = _disp_design()
        r = fit(X, y, family=GammaFamily(link='log'))
        n = X.shape[0]
        k = r.rank + 1
        expected = r.aic - 2.0 * k + k * np.log(n)
        assert r.bic == pytest.approx(expected, rel=0, abs=1e-9)
        # And NOT the buggy count that omits the dispersion.
        k_bad = r.rank
        bad = r.aic - 2.0 * k_bad + k_bad * np.log(n)
        assert abs(r.bic - bad) == pytest.approx(np.log(n) - 2.0, abs=1e-9)


class TestGaussianDispersionIC:
    """Gaussian BIC must count the estimated σ² with log(n), like R."""

    def test_aic_matches_r(self):
        X, y = _disp_design()
        r = fit(X, y, family=Gaussian())
        assert r.aic == pytest.approx(_R_GAUSS_AIC, rel=0, abs=1e-6)

    def test_bic_matches_r(self):
        X, y = _disp_design()
        r = fit(X, y, family=Gaussian())
        # Pre-4.3.2 this was off by exactly (log(n) - 2).
        assert r.bic == pytest.approx(_R_GAUSS_BIC, rel=0, abs=1e-6)

    def test_bic_counts_dispersion_param(self):
        X, y = _disp_design()
        r = fit(X, y, family=Gaussian())
        n = X.shape[0]
        k = r.rank + 1
        expected = r.aic - 2.0 * k + k * np.log(n)
        assert r.bic == pytest.approx(expected, rel=0, abs=1e-9)


class TestDispersionParamCount:
    """The n_ic_dispersion_params contract: 1 for ML-estimated dispersion."""

    def test_gaussian_counts_one(self):
        assert Gaussian().n_ic_dispersion_params == 1

    def test_gamma_counts_one(self):
        assert GammaFamily().n_ic_dispersion_params == 1

    def test_binomial_counts_zero(self):
        assert Binomial().n_ic_dispersion_params == 0

    def test_poisson_counts_zero(self):
        assert Poisson().n_ic_dispersion_params == 0

    def test_fixed_dispersion_bic_uses_rank(self):
        """Poisson BIC must still penalize exactly ``rank`` parameters."""
        # Small deterministic Poisson fit.
        x = np.linspace(-1.0, 1.0, 25)
        X = np.column_stack([np.ones(25), x])
        # counts increasing in x, deterministic
        y = np.round(np.exp(0.5 + 0.8 * x)).astype(float)
        r = fit(X, y, family='poisson')
        n = X.shape[0]
        expected = r.aic - 2.0 * r.rank + r.rank * np.log(n)
        assert r.bic == pytest.approx(expected, rel=0, abs=1e-9)


# =====================================================================
# Bug 2 — Binomial deviance / log-likelihood clip mu at machine epsilon
# =====================================================================

# Explicit (y, mu, wt) vectors with fitted probabilities in (eps, 1e-10): there
# R's binomial()$dev.resids uses mu directly, so the machine-epsilon clip matches
# R exactly while the old 1e-10 clip does not.
_BIN_Y = np.array([1.0, 0.0, 1.0, 0.0])
_BIN_MU = np.array([1e-12, 1.0 - 1e-12, 1e-15, 0.4])
_BIN_WT = np.ones(4)

# R: binomial()$dev.resids(y, mu, wt); sum(...)
_R_BIN_DEVIANCE = 180.62333274499716
# R: sum(wt * (y*log(mu) + (1-y)*log(1-mu)))
_R_BIN_LOGLIK = -90.311666372498578
# What the old 1e-10 clip produced (NOT R) — guards against regressing the bound.
_OLD_COARSE_DEVIANCE = 139.17675666169401


class TestBinomialDevianceClip:
    """Binomial deviance / log-likelihood match R's machine-eps clip."""

    def test_deviance_matches_r(self):
        d = Binomial().deviance(_BIN_Y, _BIN_MU, _BIN_WT)
        assert d == pytest.approx(_R_BIN_DEVIANCE, rel=0, abs=1e-9)

    def test_deviance_not_coarse_clip(self):
        """The fix must move off the old 1e-10 bound for extreme mu."""
        d = Binomial().deviance(_BIN_Y, _BIN_MU, _BIN_WT)
        assert abs(d - _OLD_COARSE_DEVIANCE) > 1.0

    def test_log_likelihood_matches_r(self):
        ll = Binomial().log_likelihood(_BIN_Y, _BIN_MU, _BIN_WT, 1.0)
        assert ll == pytest.approx(_R_BIN_LOGLIK, rel=0, abs=1e-9)

    def test_loglik_consistent_with_deviance(self):
        """For binary data the saturated log-lik is 0, so dev = -2·loglik."""
        b = Binomial()
        d = b.deviance(_BIN_Y, _BIN_MU, _BIN_WT)
        ll = b.log_likelihood(_BIN_Y, _BIN_MU, _BIN_WT, 1.0)
        assert d == pytest.approx(-2.0 * ll, rel=0, abs=1e-9)

    def test_eps_bound_is_machine_epsilon(self):
        """A point exactly at the eps floor uses -2·log(eps), matching R."""
        eps = float(np.finfo(np.float64).eps)
        d = Binomial().deviance(
            np.array([1.0]), np.array([eps / 2.0]), np.array([1.0])
        )
        assert d == pytest.approx(-2.0 * np.log(eps), rel=0, abs=1e-9)
