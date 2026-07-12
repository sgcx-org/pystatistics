"""
Additional GLM link functions beyond the core five in ``families.py``.

These extend the canonical set (identity, logit, log, inverse, probit) with the
remaining links a working GLM user reaches for and that R's ``glm`` offers:

- ``cloglog``  — complementary log-log, ``g(mu) = log(-log(1-mu))``
- ``cauchit``  — Cauchy CDF inverse, ``g(mu) = tan(pi (mu - 1/2))``
- ``sqrt``     — square-root, ``g(mu) = sqrt(mu)`` (a Poisson alternative)
- ``inverse-squared`` — ``g(mu) = 1/mu^2`` (the inverse-Gaussian canonical link)

Each mirrors R's convention exactly and is registered into ``families._LINK_CLASSES``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatistics.regression.families import Link


class CLogLogLink(Link):
    """Complementary log-log link: g(mu) = log(-log(1-mu))."""

    @property
    def name(self) -> str:
        return 'cloglog'

    def link(self, mu: NDArray) -> NDArray:
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return np.log(-np.log(1 - mu))

    def linkinv(self, eta: NDArray) -> NDArray:
        eta = np.clip(eta, -500, 500)
        return 1.0 - np.exp(-np.exp(eta))

    def mu_eta(self, eta: NDArray) -> NDArray:
        eta = np.clip(eta, -500, 500)
        exp_eta = np.exp(eta)
        return np.maximum(exp_eta * np.exp(-exp_eta), 1e-10)


class CauchitLink(Link):
    """Cauchit link: g(mu) = tan(pi (mu - 1/2)); inverse is the Cauchy CDF."""

    @property
    def name(self) -> str:
        return 'cauchit'

    def link(self, mu: NDArray) -> NDArray:
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return np.tan(np.pi * (mu - 0.5))

    def linkinv(self, eta: NDArray) -> NDArray:
        return 0.5 + np.arctan(eta) / np.pi

    def mu_eta(self, eta: NDArray) -> NDArray:
        return np.maximum(1.0 / (np.pi * (1.0 + eta * eta)), 1e-10)


class SqrtLink(Link):
    """Square-root link: g(mu) = sqrt(mu). A common Poisson alternative."""

    @property
    def name(self) -> str:
        return 'sqrt'

    def link(self, mu: NDArray) -> NDArray:
        return np.sqrt(np.maximum(mu, 0.0))

    def linkinv(self, eta: NDArray) -> NDArray:
        return eta ** 2

    def mu_eta(self, eta: NDArray) -> NDArray:
        return 2.0 * eta


class InverseSquaredLink(Link):
    """Inverse-squared link: g(mu) = 1/mu^2. Inverse-Gaussian canonical link."""

    @property
    def name(self) -> str:
        return 'inverse-squared'

    def link(self, mu: NDArray) -> NDArray:
        return 1.0 / np.maximum(mu, 1e-10) ** 2

    def linkinv(self, eta: NDArray) -> NDArray:
        # eta = 1/mu^2  ->  mu = eta^{-1/2}
        return 1.0 / np.sqrt(np.maximum(eta, 1e-10))

    def mu_eta(self, eta: NDArray) -> NDArray:
        # d/deta eta^{-1/2} = -1/2 eta^{-3/2}
        return -0.5 * np.maximum(eta, 1e-10) ** (-1.5)
