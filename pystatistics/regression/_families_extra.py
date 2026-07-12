"""
Additional GLM families beyond the core set in ``families.py``.

- ``QuasiPoisson`` / ``QuasiBinomial`` — the overdispersion families: identical
  IRLS fit (hence identical coefficients) to Poisson / Binomial, but the
  dispersion is *estimated* from the Pearson statistic rather than fixed at 1, so
  standard errors are scaled by ``sqrt(phi_hat)``. Like R's ``quasipoisson()`` /
  ``quasibinomial()`` they define no proper likelihood, so AIC is undefined (NaN).
- ``InverseGaussian`` — variance ``V(mu) = mu^3``; canonical link ``1/mu^2``.

Registered into ``families._FAMILY_CLASSES`` under R's names.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatistics.regression.families import (
    Family, Link, Binomial, Poisson,
)
from pystatistics.regression._links_extra import InverseSquaredLink


class QuasiPoisson(Poisson):
    """Quasi-Poisson family: Poisson mean-variance with estimated dispersion.

    Coefficients match ``poisson``; standard errors are inflated by the estimated
    overdispersion ``phi_hat`` (Pearson / df). No proper likelihood → AIC is NaN,
    matching R's ``quasipoisson()``.
    """

    @property
    def name(self) -> str:
        return 'quasipoisson'

    @property
    def dispersion_is_fixed(self) -> bool:
        return False

    @property
    def dispersion_estimator(self) -> str:
        return 'pearson'

    def log_likelihood(
        self, y: NDArray, mu: NDArray, wt: NDArray, dispersion: float
    ) -> float:
        # Quasi-families have no true likelihood (R returns NA for AIC).
        return float('nan')

    def aic(
        self, y: NDArray, mu: NDArray, wt: NDArray, rank: int, dispersion: float
    ) -> float:
        return float('nan')


class QuasiBinomial(Binomial):
    """Quasi-binomial family: binomial mean-variance with estimated dispersion.

    Coefficients match ``binomial``; standard errors are inflated by the estimated
    overdispersion. No proper likelihood → AIC is NaN, matching R's
    ``quasibinomial()``.
    """

    @property
    def name(self) -> str:
        return 'quasibinomial'

    @property
    def dispersion_is_fixed(self) -> bool:
        return False

    @property
    def dispersion_estimator(self) -> str:
        return 'pearson'

    def log_likelihood(
        self, y: NDArray, mu: NDArray, wt: NDArray, dispersion: float
    ) -> float:
        return float('nan')

    def aic(
        self, y: NDArray, mu: NDArray, wt: NDArray, rank: int, dispersion: float
    ) -> float:
        return float('nan')


class InverseGaussian(Family):
    """Inverse-Gaussian family. Default link: ``1/mu^2`` (canonical).

    V(mu) = mu^3; unit deviance ``(y - mu)^2 / (y mu^2)``. Used for positive,
    right-skewed continuous data with variance rising as mu^3. Matches R's
    ``inverse.gaussian()``.
    """

    @property
    def name(self) -> str:
        return 'inverse-gaussian'

    def _default_link(self) -> Link:
        return InverseSquaredLink()

    def variance(self, mu: NDArray) -> NDArray:
        return np.maximum(mu, 1e-10) ** 3

    def initialize(self, y: NDArray, weights: NDArray | None = None) -> NDArray:
        # Positive mean; R uses mustart = y.
        return np.maximum(y, 1e-10)

    def deviance(self, y: NDArray, mu: NDArray, wt: NDArray) -> float:
        # R inverse.gaussian dev.resids: wt * (y - mu)^2 / (y * mu^2)
        mu = np.maximum(mu, 1e-10)
        y_safe = np.maximum(y, 1e-10)
        return float(np.sum(wt * (y - mu) ** 2 / (y_safe * mu ** 2)))

    @property
    def dispersion_is_fixed(self) -> bool:
        return False

    @property
    def dispersion_estimator(self) -> str:
        # R's summary.glm reports the Pearson dispersion for inverse.gaussian.
        return 'pearson'

    @property
    def n_ic_dispersion_params(self) -> int:
        # The dispersion (1/lambda) is ML-estimated and counted by AIC/BIC.
        return 1

    def log_likelihood(
        self, y: NDArray, mu: NDArray, wt: NDArray, dispersion: float
    ) -> float:
        # Inverse-Gaussian density with lambda = 1/dispersion:
        #   f = sqrt(lambda / (2 pi y^3)) exp(-lambda (y-mu)^2 / (2 mu^2 y))
        # loglik = 0.5 sum wt (log(lambda) - log(2 pi) - 3 log y)
        #          - lambda * dev / 2,  with dev = sum wt (y-mu)^2/(y mu^2).
        if not (np.isfinite(dispersion) and dispersion > 0):
            return float('nan')
        y_safe = np.maximum(y, 1e-10)
        lam = 1.0 / dispersion
        sw = float(np.sum(wt))
        dev = self.deviance(y, mu, wt)
        ll = 0.5 * (sw * np.log(lam) - sw * np.log(2 * np.pi)
                    - 3.0 * float(np.sum(wt * np.log(y_safe)))) - 0.5 * lam * dev
        return float(ll)

    def aic(
        self, y: NDArray, mu: NDArray, wt: NDArray, rank: int, dispersion: float
    ) -> float:
        """AIC matching R's ``inverse.gaussian()$aic``.

        R evaluates the AIC log-likelihood at the ML dispersion ``dev/sum(wt)``
        (not the moment estimate used for standard errors) and counts the
        dispersion as a free parameter (``+2``); ``glm.fit`` then adds
        ``2*rank``.
        """
        sw = float(np.sum(wt))
        dev = self.deviance(y, mu, wt)
        disp_mle = dev / sw
        ll = self.log_likelihood(y, mu, wt, disp_mle)
        return -2.0 * ll + 2.0 + 2.0 * rank
