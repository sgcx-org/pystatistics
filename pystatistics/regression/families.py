"""
GLM family and link function specifications.

Each Family defines:
- A variance function V(μ) relating variance to the mean
- A default link function g(μ) mapping the mean to the linear predictor
- A deviance function for assessing model fit
- A log-likelihood function for AIC computation
- An initialization function for IRLS starting values

Each Link defines:
- g(μ) → η  (link)
- g⁻¹(η) → μ  (inverse link)
- dμ/dη  (derivative of inverse link, for IRLS weights)

References:
    McCullagh, P., & Nelder, J. A. (1989). Generalized Linear Models (2nd ed.)
    R Core Team. stats::family, stats::make.link
"""

from __future__ import annotations

from pystatistics.core.exceptions import ValidationError

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


# Machine epsilon for float64 (~2.22e-16). Used as the lower/upper bound when
# clipping binomial fitted probabilities in the deviance and log-likelihood, so
# that those statistics match R's binomial()$dev.resids / linkinv, which bound
# mu to .Machine$double.eps. log(eps) is finite, so there is no overflow risk.
_FLOAT64_EPS = float(np.finfo(np.float64).eps)


# =====================================================================
# Link functions
# =====================================================================

class Link(ABC):
    """Abstract link function g(μ) mapping mean to linear predictor."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def link(self, mu: NDArray) -> NDArray:
        """g(μ) → η."""
        ...

    @abstractmethod
    def linkinv(self, eta: NDArray) -> NDArray:
        """g⁻¹(η) → μ."""
        ...

    @abstractmethod
    def mu_eta(self, eta: NDArray) -> NDArray:
        """dμ/dη = (g⁻¹)'(η)."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class IdentityLink(Link):
    """Identity link: g(μ) = μ. Default for Gaussian family."""

    @property
    def name(self) -> str:
        return 'identity'

    def link(self, mu: NDArray) -> NDArray:
        return mu.copy()

    def linkinv(self, eta: NDArray) -> NDArray:
        return eta.copy()

    def mu_eta(self, eta: NDArray) -> NDArray:
        return np.ones_like(eta)


class LogitLink(Link):
    """Logit link: g(μ) = log(μ/(1-μ)). Default for Binomial family."""

    @property
    def name(self) -> str:
        return 'logit'

    def link(self, mu: NDArray) -> NDArray:
        # NUMERICAL GUARD: prevents log(0) and division by zero in logit transform
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return np.log(mu / (1 - mu))

    def linkinv(self, eta: NDArray) -> NDArray:
        # Clip to prevent overflow in exp
        eta = np.clip(eta, -500, 500)
        return 1.0 / (1.0 + np.exp(-eta))

    def mu_eta(self, eta: NDArray) -> NDArray:
        eta = np.clip(eta, -500, 500)
        p = 1.0 / (1.0 + np.exp(-eta))
        # NUMERICAL GUARD: prevents zero derivative in IRLS weights
        return np.maximum(p * (1.0 - p), 1e-10)


class LogLink(Link):
    """Log link: g(μ) = log(μ). Default for Poisson family."""

    @property
    def name(self) -> str:
        return 'log'

    def link(self, mu: NDArray) -> NDArray:
        # NUMERICAL GUARD: prevents log(0) in log link function
        return np.log(np.maximum(mu, 1e-10))

    def linkinv(self, eta: NDArray) -> NDArray:
        # Clip to prevent overflow
        eta = np.clip(eta, -500, 500)
        return np.exp(eta)

    def mu_eta(self, eta: NDArray) -> NDArray:
        eta = np.clip(eta, -500, 500)
        return np.exp(eta)


class InverseLink(Link):
    """Inverse link: g(μ) = 1/μ. Default for Gamma family."""

    @property
    def name(self) -> str:
        return 'inverse'

    def link(self, mu: NDArray) -> NDArray:
        # NUMERICAL GUARD: prevents division by zero in inverse link
        return 1.0 / np.maximum(mu, 1e-10)

    def linkinv(self, eta: NDArray) -> NDArray:
        # NUMERICAL GUARD: prevents division by zero in inverse link
        return 1.0 / np.maximum(eta, 1e-10)

    def mu_eta(self, eta: NDArray) -> NDArray:
        # NUMERICAL GUARD: prevents division by zero in derivative
        return -1.0 / np.maximum(eta ** 2, 1e-20)


class ProbitLink(Link):
    """Probit link: g(μ) = Φ⁻¹(μ). Alternative for Binomial family."""

    @property
    def name(self) -> str:
        return 'probit'

    def link(self, mu: NDArray) -> NDArray:
        from scipy.stats import norm
        # NUMERICAL GUARD: prevents Inf from ppf(0) or ppf(1)
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return norm.ppf(mu)

    def linkinv(self, eta: NDArray) -> NDArray:
        from scipy.stats import norm
        return norm.cdf(eta)

    def mu_eta(self, eta: NDArray) -> NDArray:
        from scipy.stats import norm
        # NUMERICAL GUARD: prevents zero derivative in IRLS weights
        return np.maximum(norm.pdf(eta), 1e-10)


# =====================================================================
# Link name → class mapping
# =====================================================================

from pystatistics.regression._links_extra import (  # noqa: E402
    CLogLogLink, CauchitLink, SqrtLink, InverseSquaredLink,
)

_LINK_CLASSES: dict[str, type[Link]] = {
    'identity': IdentityLink,
    'logit': LogitLink,
    'log': LogLink,
    'inverse': InverseLink,
    'probit': ProbitLink,
    'cloglog': CLogLogLink,
    'cauchit': CauchitLink,
    'sqrt': SqrtLink,
    'inverse-squared': InverseSquaredLink,
}


def _resolve_link(link: str | Link | None, default: Link) -> Link:
    """Resolve a link argument to a Link instance."""
    if link is None:
        return default
    if isinstance(link, Link):
        return link
    if isinstance(link, str):
        cls = _LINK_CLASSES.get(link.lower())
        if cls is None:
            valid = ', '.join(sorted(_LINK_CLASSES.keys()))
            raise ValidationError(f"Unknown link: {link!r}. Valid links: {valid}")
        return cls()
    raise ValidationError(f"link must be str or Link, got {type(link).__name__}")


# =====================================================================
# Family base class
# =====================================================================

class Family(ABC):
    """
    GLM family specification.

    Defines the relationship between the mean and variance of the
    response distribution, along with a link function.
    """

    def __init__(self, link: str | Link | None = None):
        self._link = _resolve_link(link, self._default_link())

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def _default_link(self) -> Link:
        ...

    @property
    def link(self) -> Link:
        return self._link

    @abstractmethod
    def variance(self, mu: NDArray) -> NDArray:
        """Variance function V(μ)."""
        ...

    @abstractmethod
    def deviance(self, y: NDArray, mu: NDArray, wt: NDArray) -> float:
        """Compute total deviance: 2 * Σ wt_i * d(y_i, μ_i).

        The deviance is twice the difference between the saturated
        log-likelihood and the model log-likelihood.
        """
        ...

    @abstractmethod
    def initialize(self, y: NDArray, weights: NDArray | None = None) -> NDArray:
        """Initialize μ from y for IRLS starting values.

        Must return values in the valid range for the link function.
        ``weights`` are the per-observation prior weights (``None`` ⇒ unit
        weights); only families whose R ``mustart`` depends on the prior
        weights (Binomial) consult them.
        """
        ...

    @property
    def dispersion_is_fixed(self) -> bool:
        """Whether the dispersion parameter is known a priori.

        True for Binomial (φ=1) and Poisson (φ=1).
        False for Gaussian (φ=σ² estimated from data).
        """
        return False

    @property
    def dispersion_estimator(self) -> str:
        """How an *estimated* dispersion is computed from the fit.

        ``'pearson'`` → ``Σ (Pearson residual)² / df_residual`` (R's
        ``summary.glm`` convention, and the definitional estimate for the
        quasi-likelihood families); ``'deviance'`` → ``deviance / df_residual``.
        Only consulted when ``dispersion_is_fixed`` is False. The base default is
        ``'deviance'`` to preserve each existing family's validated convention;
        the quasi and inverse-Gaussian families override to ``'pearson'`` to
        match R.
        """
        return 'deviance'

    @property
    def n_ic_dispersion_params(self) -> int:
        """Number of ML-estimated dispersion/shape parameters the information
        criteria penalize as free parameters, beyond the regression coefficients
        counted in ``rank``.

        R counts the dispersion of Gaussian (σ²) and Gamma (the shape) GLMs as a
        free parameter in both AIC and BIC — its ``logLik`` reports
        ``df = rank + 1``. The fixed-dispersion families (Binomial, Poisson) and
        the fixed-θ negative binomial do not, so this returns 0 by default.
        ``aic()`` of the affected families adds the ``+2`` for this parameter;
        recording the count here lets ``GLMSolution.bic`` re-penalize it with
        ``log(n)`` instead of leaving it at the AIC constant.
        """
        return 0

    @abstractmethod
    def log_likelihood(
        self, y: NDArray, mu: NDArray, wt: NDArray, dispersion: float
    ) -> float:
        """Compute the log-likelihood for AIC.

        This must match R's family$aic() / (-2) for consistency.
        """
        ...

    def aic(
        self, y: NDArray, mu: NDArray, wt: NDArray,
        rank: int, dispersion: float
    ) -> float:
        """Compute AIC = -2 * loglik + 2 * rank.

        For families with estimated dispersion (Gaussian), R computes
        AIC differently. Subclasses may override.
        """
        ll = self.log_likelihood(y, mu, wt, dispersion)
        return -2.0 * ll + 2.0 * rank

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(link={self._link.name!r})"


# =====================================================================
# Concrete families
# =====================================================================

class Gaussian(Family):
    """Gaussian (Normal) family. Default link: identity.

    V(μ) = 1
    Deviance = Σ wt_i * (y_i - μ_i)²  (= RSS for identity link)
    """

    @property
    def name(self) -> str:
        return 'gaussian'

    def _default_link(self) -> Link:
        return IdentityLink()

    def variance(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    def initialize(self, y: NDArray, weights: NDArray | None = None) -> NDArray:
        return y.copy()

    def deviance(self, y: NDArray, mu: NDArray, wt: NDArray) -> float:
        return float(np.sum(wt * (y - mu) ** 2))

    def log_likelihood(
        self, y: NDArray, mu: NDArray, wt: NDArray, dispersion: float
    ) -> float:
        # R: -0.5 * (sum(wt * ((y - mu)^2) / dispersion) +
        #            sum(log(2 * pi * dispersion / wt)))
        # Simplifies for wt=1 to: -n/2 * log(2πσ²) - RSS/(2σ²)
        n = float(np.sum(wt > 0))
        rss = float(np.sum(wt * (y - mu) ** 2))
        ll = -0.5 * (rss / dispersion + n * np.log(2 * np.pi * dispersion))
        return ll

    def aic(
        self, y: NDArray, mu: NDArray, wt: NDArray,
        rank: int, dispersion: float
    ) -> float:
        """Compute AIC matching R's gaussian family.

        R's gaussian()$aic uses MLE dispersion (dev/n, not dev/df) and
        adds +2 for the dispersion parameter. The formula is:
            AIC = -2 * loglik(sigma_mle) + 2 + 2 * rank
        where sigma_mle = sqrt(deviance / n), and the +2 comes from the
        gaussian family counting the dispersion as an extra parameter.
        """
        n = float(np.sum(wt > 0))
        # R uses MLE sigma, not the residual-df-corrected sigma
        rss = float(np.sum(wt * (y - mu) ** 2))
        sigma_mle_sq = rss / n  # MLE dispersion

        ll = -0.5 * (rss / sigma_mle_sq + n * np.log(2 * np.pi * sigma_mle_sq))
        # R gaussian aic() adds +2 for the dispersion parameter
        return -2.0 * ll + 2.0 + 2.0 * rank

    @property
    def dispersion_is_fixed(self) -> bool:
        return False

    @property
    def n_ic_dispersion_params(self) -> int:
        # σ² is ML-estimated and counted by AIC (the +2 above) and BIC.
        return 1


class Binomial(Family):
    """Binomial family. Default link: logit.

    V(μ) = μ(1-μ)
    Deviance = 2 * Σ wt_i * [y_i log(y_i/μ_i) + (1-y_i) log((1-y_i)/(1-μ_i))]
    """

    @property
    def name(self) -> str:
        return 'binomial'

    def _default_link(self) -> Link:
        return LogitLink()

    def variance(self, mu: NDArray) -> NDArray:
        # NUMERICAL GUARD: prevents zero variance in IRLS weights
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return mu * (1.0 - mu)

    def initialize(self, y: NDArray, weights: NDArray | None = None) -> NDArray:
        # R's default mustart: (weights * y + 0.5) / (weights + 1).
        # With unit weights this is (y + 0.5) / 2.
        w = np.ones_like(y) if weights is None else weights
        return (w * y + 0.5) / (w + 1.0)

    def deviance(self, y: NDArray, mu: NDArray, wt: NDArray) -> float:
        # NUMERICAL GUARD: prevents log(0) in deviance computation. Bound at
        # machine epsilon (not 1e-10) to match R's binomial()$dev.resids, which
        # clips mu to .Machine$double.eps; a coarser bound under-counts the
        # deviance for models with very extreme fitted probabilities.
        mu = np.clip(mu, _FLOAT64_EPS, 1 - _FLOAT64_EPS)
        # Unit deviance: 2 * [y*log(y/mu) + (1-y)*log((1-y)/(1-mu))]
        # with 0*log(0) = 0. np.where evaluates both branches, so
        # suppress harmless warnings from the unused branch.
        with np.errstate(divide='ignore', invalid='ignore'):
            term1 = np.where(y > 0, y * np.log(y / mu), 0.0)
            term2 = np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu)), 0.0)
        return 2.0 * float(np.sum(wt * (term1 + term2)))

    def log_likelihood(
        self, y: NDArray, mu: NDArray, wt: NDArray, dispersion: float
    ) -> float:
        # Binomial log-likelihood for binary data:
        # Σ wt_i * [y_i * log(μ_i) + (1 - y_i) * log(1 - μ_i)]
        # NUMERICAL GUARD: bound at machine epsilon (matching R, see deviance
        # above) so the AIC/BIC log-likelihood uses the same clip as the
        # deviance and stays consistent with R's binomial()$aic.
        mu = np.clip(mu, _FLOAT64_EPS, 1 - _FLOAT64_EPS)
        ll = float(np.sum(wt * (y * np.log(mu) + (1 - y) * np.log(1 - mu))))
        return ll

    @property
    def dispersion_is_fixed(self) -> bool:
        return True


class Poisson(Family):
    """Poisson family. Default link: log.

    V(μ) = μ
    Deviance = 2 * Σ wt_i * [y_i log(y_i/μ_i) - (y_i - μ_i)]
    """

    @property
    def name(self) -> str:
        return 'poisson'

    def _default_link(self) -> Link:
        return LogLink()

    def variance(self, mu: NDArray) -> NDArray:
        # NUMERICAL GUARD: prevents zero variance in IRLS weights
        return np.maximum(mu, 1e-10)

    def initialize(self, y: NDArray, weights: NDArray | None = None) -> NDArray:
        # R: y + 0.1 (to avoid log(0))
        return np.maximum(y, 0.1)

    def deviance(self, y: NDArray, mu: NDArray, wt: NDArray) -> float:
        # NUMERICAL GUARD: prevents log(0) in deviance computation
        mu = np.maximum(mu, 1e-10)
        # Unit deviance: 2 * [y*log(y/mu) - (y - mu)]
        # with 0*log(0) = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.where(y > 0, y * np.log(y / mu), 0.0)
        return 2.0 * float(np.sum(wt * (term - (y - mu))))

    def log_likelihood(
        self, y: NDArray, mu: NDArray, wt: NDArray, dispersion: float
    ) -> float:
        # Poisson log-likelihood:
        # Σ wt_i * [y_i * log(μ_i) - μ_i - log(y_i!)]
        from scipy.special import gammaln
        # NUMERICAL GUARD: prevents log(0) in log-likelihood computation
        mu = np.maximum(mu, 1e-10)
        ll = float(np.sum(
            wt * (y * np.log(mu) - mu - gammaln(y + 1))
        ))
        return ll

    @property
    def dispersion_is_fixed(self) -> bool:
        return True


class Gamma(Family):
    """Gamma family. Default link: inverse.

    V(μ) = μ²
    Deviance = 2 * Σ wt_i * [(y_i - μ_i)/μ_i - log(y_i/μ_i)]

    Used for positive continuous data with variance proportional to mean².
    Typical applications: cost data, survival times, insurance claims.

    References:
        R: stats::Gamma()
        McCullagh & Nelder (1989), Ch. 8
    """

    @property
    def name(self) -> str:
        return 'gamma'

    def _default_link(self) -> Link:
        return InverseLink()

    def variance(self, mu: NDArray) -> NDArray:
        # NUMERICAL GUARD: prevents zero variance when mu is near zero
        return np.maximum(mu, 1e-10) ** 2

    def initialize(self, y: NDArray, weights: NDArray | None = None) -> NDArray:
        # NUMERICAL GUARD: Gamma requires positive μ
        return np.maximum(y, 1e-10)

    def deviance(self, y: NDArray, mu: NDArray, wt: NDArray) -> float:
        # NUMERICAL GUARD: prevents division by zero and log(0)
        mu = np.maximum(mu, 1e-10)
        y_safe = np.maximum(y, 1e-10)
        # Unit deviance: -2 * (log(y/μ) - (y - μ)/μ)
        return 2.0 * float(np.sum(wt * ((y - mu) / mu - np.log(y_safe / mu))))

    def log_likelihood(
        self, y: NDArray, mu: NDArray, wt: NDArray, dispersion: float
    ) -> float:
        # Gamma log-likelihood (shape-rate parameterization):
        # shape = 1/dispersion, rate = shape/μ
        # loglik = Σ wt * (shape*log(rate) + (shape-1)*log(y) - rate*y - lgamma(shape))
        from scipy.special import gammaln
        # NUMERICAL GUARD: dispersion must be strictly positive for the
        # Gamma log-likelihood. A non-positive dispersion estimate (dev/df
        # with perfect fit) would produce log(negative) → NaN. Instead of
        # emitting a RuntimeWarning and returning NaN silently (Rule 1
        # violation), return NaN explicitly so the caller sees a clearly
        # undefined log-likelihood for a degenerate model.
        if not (np.isfinite(dispersion) and dispersion > 0):
            return float('nan')
        shape = 1.0 / dispersion
        # NUMERICAL GUARD: prevents log(0) in Gamma log-likelihood
        mu = np.maximum(mu, 1e-10)
        y_safe = np.maximum(y, 1e-10)
        rate = shape / mu
        ll = float(np.sum(wt * (
            shape * np.log(rate) + (shape - 1.0) * np.log(y_safe)
            - rate * y - gammaln(shape)
        )))
        return ll

    def aic(
        self, y: NDArray, mu: NDArray, wt: NDArray,
        rank: int, dispersion: float
    ) -> float:
        """Compute AIC matching R's ``Gamma()$aic``.

        R's Gamma family evaluates the AIC log-likelihood at a dispersion of
        ``dev / sum(wt)`` — the MLE of the dispersion under the Gamma
        distribution — NOT the moment estimate ``dev / df_residual`` that R
        reports in ``summary.glm`` and that PyStatistics stores in
        ``GLMParams.dispersion`` for standard errors. The two diverge whenever
        ``rank > 0``, so the ``dispersion`` argument (which the solver derives
        from ``df_residual``) must be ignored here and the AIC-specific
        dispersion recomputed internally.

        R also counts the estimated dispersion/shape as a free parameter,
        adding ``+2`` on top of ``2 * rank``. Concretely R computes
        ``Gamma()$aic`` as ``-2 * sum(wt * dgamma(y, 1/disp, scale=mu*disp,
        log=TRUE)) + 2`` and ``glm.fit`` adds ``2 * rank``, giving the formula
        below. ``log_likelihood`` evaluated at ``disp`` is algebraically the
        weighted ``dgamma`` sum.
        """
        dev = self.deviance(y, mu, wt)
        total_wt = float(np.sum(wt))
        # NUMERICAL GUARD: a degenerate (zero-weight or perfect) fit yields a
        # non-positive dispersion; log_likelihood returns NaN, which the base
        # aic would propagate. Surface it explicitly rather than masking.
        disp = dev / total_wt if total_wt > 0 else float('nan')
        ll = self.log_likelihood(y, mu, wt, disp)
        # +2 for the estimated dispersion parameter (R's Gamma()$aic), +2*rank
        # for the regression coefficients (R's glm.fit).
        return -2.0 * ll + 2.0 + 2.0 * rank

    @property
    def dispersion_is_fixed(self) -> bool:
        return False

    @property
    def n_ic_dispersion_params(self) -> int:
        # The Gamma shape (1/dispersion) is ML-estimated and counted by AIC
        # (the +2 above) and BIC, matching R's Gamma()$aic / logLik df.
        return 1


class NegativeBinomial(Family):
    """Negative binomial family. Default link: log.

    V(μ) = μ + μ²/θ  (where θ is the dispersion parameter)

    For fixed θ, this is a standard GLM with known variance function.
    When θ is unknown, it must be estimated via profile likelihood
    (see regression._nb_theta).

    Args:
        theta: The dispersion parameter (> 0). Larger θ means less
            overdispersion; θ → ∞ recovers Poisson. If None, theta must
            be estimated externally (e.g., via fit(family='negative-binomial')).
        link: Link function (default: log).

    References:
        R: MASS::negative.binomial(), MASS::glm.nb()
        Venables & Ripley (2002), Modern Applied Statistics with S, Ch. 7.4
    """

    def __init__(
        self, theta: float | None = None, link: str | Link | None = None
    ):
        if theta is not None and theta <= 0:
            raise ValidationError(f"theta must be positive, got {theta}")
        self.theta = theta
        super().__init__(link)

    @property
    def name(self) -> str:
        return 'negative-binomial'

    def _default_link(self) -> Link:
        return LogLink()

    def variance(self, mu: NDArray) -> NDArray:
        if self.theta is None:
            raise ValidationError(
                "Cannot compute variance without theta. "
                "Set theta or use fit(family='negative-binomial') for "
                "automatic theta estimation."
            )
        # NUMERICAL GUARD: prevents zero variance when mu is near zero
        mu = np.maximum(mu, 1e-10)
        return mu + mu ** 2 / self.theta

    def initialize(self, y: NDArray, weights: NDArray | None = None) -> NDArray:
        # Same as Poisson: y + 0.1 to avoid log(0)
        return np.maximum(y, 0.1)

    def deviance(self, y: NDArray, mu: NDArray, wt: NDArray) -> float:
        if self.theta is None:
            raise ValidationError("Cannot compute deviance without theta.")
        theta = self.theta
        # NUMERICAL GUARD: prevents log(0) in deviance computation
        mu = np.maximum(mu, 1e-10)
        # Unit deviance: 2 * (y*log(max(y,1)/mu) - (y+θ)*log((y+θ)/(μ+θ)))
        with np.errstate(divide='ignore', invalid='ignore'):
            term1 = np.where(y > 0, y * np.log(y / mu), 0.0)
            term2 = (y + theta) * np.log((y + theta) / (mu + theta))
        return 2.0 * float(np.sum(wt * (term1 - term2)))

    def log_likelihood(
        self, y: NDArray, mu: NDArray, wt: NDArray, dispersion: float
    ) -> float:
        if self.theta is None:
            raise ValidationError("Cannot compute log-likelihood without theta.")
        theta = self.theta
        from scipy.special import gammaln
        # NUMERICAL GUARD: prevents log(0) in NB log-likelihood
        mu = np.maximum(mu, 1e-10)
        # NB log-likelihood:
        # Σ wt * (lgamma(y+θ) - lgamma(θ) - lgamma(y+1)
        #         + θ*log(θ/(μ+θ)) + y*log(μ/(μ+θ)))
        ll = float(np.sum(wt * (
            gammaln(y + theta) - gammaln(theta) - gammaln(y + 1)
            + theta * np.log(theta / (mu + theta))
            + y * np.log(mu / (mu + theta))
        )))
        return ll

    @property
    def dispersion_is_fixed(self) -> bool:
        # For a given theta, the NB GLM has phi=1 (dispersion is "fixed"
        # in the sense that it's not estimated from Pearson residuals).
        #
        # AIC: the base Family.aic (-2*loglik + 2*rank) already matches
        # MASS::negative.binomial(theta)$aic, which returns -2*loglik with no
        # extra dispersion-parameter penalty — theta is treated as known here.
        # (MASS::glm.nb adds the +2 for an *estimated* theta separately, on top
        # of this family AIC; that is out of scope for a fixed-theta fit.)
        return True

    def __repr__(self) -> str:
        theta_str = f'{self.theta:.4g}' if self.theta is not None else 'None'
        return f"NegativeBinomial(theta={theta_str}, link={self._link.name!r})"


# =====================================================================
# Family name → class mapping + resolver
# =====================================================================

from pystatistics.regression._families_extra import (  # noqa: E402
    QuasiPoisson, QuasiBinomial, InverseGaussian,
)

_FAMILY_CLASSES: dict[str, type[Family]] = {
    'gaussian': Gaussian,
    'normal': Gaussian,
    'binomial': Binomial,
    'poisson': Poisson,
    'gamma': Gamma,
    'negative-binomial': NegativeBinomial,
    'nb': NegativeBinomial,
    'quasipoisson': QuasiPoisson,
    'quasibinomial': QuasiBinomial,
    'inverse-gaussian': InverseGaussian,
}


def resolve_family(family: str | Family) -> Family:
    """Resolve a family argument to a Family instance.

    Args:
        family: Either a string name ('gaussian', 'binomial', 'poisson')
                or a Family instance (passed through).

    Returns:
        Family instance.

    Raises:
        ValueError: If string name is not recognized.
        TypeError: If argument is neither string nor Family.
    """
    if isinstance(family, Family):
        return family
    if isinstance(family, str):
        cls = _FAMILY_CLASSES.get(family.lower())
        if cls is None:
            valid = ', '.join(sorted(_FAMILY_CLASSES.keys()))
            raise ValidationError(
                f"Unknown family: {family!r}. Valid families: {valid}"
            )
        return cls()
    raise ValidationError(f"family must be str or Family, got {type(family).__name__}")
