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

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


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
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return np.log(mu / (1 - mu))

    def linkinv(self, eta: NDArray) -> NDArray:
        # Clip to prevent overflow in exp
        eta = np.clip(eta, -500, 500)
        return 1.0 / (1.0 + np.exp(-eta))

    def mu_eta(self, eta: NDArray) -> NDArray:
        eta = np.clip(eta, -500, 500)
        p = 1.0 / (1.0 + np.exp(-eta))
        return np.maximum(p * (1.0 - p), 1e-10)


class LogLink(Link):
    """Log link: g(μ) = log(μ). Default for Poisson family."""

    @property
    def name(self) -> str:
        return 'log'

    def link(self, mu: NDArray) -> NDArray:
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
        return 1.0 / np.maximum(mu, 1e-10)

    def linkinv(self, eta: NDArray) -> NDArray:
        return 1.0 / np.maximum(eta, 1e-10)

    def mu_eta(self, eta: NDArray) -> NDArray:
        return -1.0 / np.maximum(eta ** 2, 1e-20)


class ProbitLink(Link):
    """Probit link: g(μ) = Φ⁻¹(μ). Alternative for Binomial family."""

    @property
    def name(self) -> str:
        return 'probit'

    def link(self, mu: NDArray) -> NDArray:
        from scipy.stats import norm
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return norm.ppf(mu)

    def linkinv(self, eta: NDArray) -> NDArray:
        from scipy.stats import norm
        return norm.cdf(eta)

    def mu_eta(self, eta: NDArray) -> NDArray:
        from scipy.stats import norm
        return np.maximum(norm.pdf(eta), 1e-10)


# =====================================================================
# Link name → class mapping
# =====================================================================

_LINK_CLASSES: dict[str, type[Link]] = {
    'identity': IdentityLink,
    'logit': LogitLink,
    'log': LogLink,
    'inverse': InverseLink,
    'probit': ProbitLink,
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
            raise ValueError(f"Unknown link: {link!r}. Valid links: {valid}")
        return cls()
    raise TypeError(f"link must be str or Link, got {type(link).__name__}")


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
    def initialize(self, y: NDArray) -> NDArray:
        """Initialize μ from y for IRLS starting values.

        Must return values in the valid range for the link function.
        """
        ...

    @property
    def dispersion_is_fixed(self) -> bool:
        """Whether the dispersion parameter is known a priori.

        True for Binomial (φ=1) and Poisson (φ=1).
        False for Gaussian (φ=σ² estimated from data).
        """
        return False

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

    def initialize(self, y: NDArray) -> NDArray:
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
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return mu * (1.0 - mu)

    def initialize(self, y: NDArray) -> NDArray:
        # R's default: (y + 0.5) / 2 for binary data
        return (y + 0.5) / 2.0

    def deviance(self, y: NDArray, mu: NDArray, wt: NDArray) -> float:
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
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
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
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
        return np.maximum(mu, 1e-10)

    def initialize(self, y: NDArray) -> NDArray:
        # R: y + 0.1 (to avoid log(0))
        return np.maximum(y, 0.1)

    def deviance(self, y: NDArray, mu: NDArray, wt: NDArray) -> float:
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
        mu = np.maximum(mu, 1e-10)
        ll = float(np.sum(
            wt * (y * np.log(mu) - mu - gammaln(y + 1))
        ))
        return ll

    @property
    def dispersion_is_fixed(self) -> bool:
        return True


# =====================================================================
# Family name → class mapping + resolver
# =====================================================================

_FAMILY_CLASSES: dict[str, type[Family]] = {
    'gaussian': Gaussian,
    'normal': Gaussian,
    'binomial': Binomial,
    'poisson': Poisson,
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
            valid = ', '.join(
                sorted(k for k in _FAMILY_CLASSES.keys() if k != 'normal')
            )
            raise ValueError(
                f"Unknown family: {family!r}. Valid families: {valid}"
            )
        return cls()
    raise TypeError(f"family must be str or Family, got {type(family).__name__}")
