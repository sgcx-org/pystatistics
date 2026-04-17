"""Torch-native GLM family operations for GAM P-IRLS on GPU.

The pystatistics ``Family`` / ``Link`` classes compute on numpy. The
GAM P-IRLS loop evaluates the family link, its derivative, variance,
deviance, and initialization helper at every iteration; pulling those
through numpy on the tensor-resident path would round-trip n-length
vectors across PCIe every step.

This module exposes the minimum torch ops the P-IRLS loop and its
convergence / score machinery actually need, for the four supported
family / canonical-link pairs (matching GEE's GPU family module).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class GPUFamilyOps:
    """Torch elementwise operations for a GLM family/link pair.

    Attributes
    ----------
    name, link_name : str
    link_fn : eta = g(mu)
    linkinv : mu = g^{-1}(eta)
    mu_eta : d mu / d eta
    variance : V(mu)
    deviance : sum of per-observation deviance contributions (scalar)
    mu_from_y : safe IRLS starting mean given the response
    """

    name: str
    link_name: str
    link_fn: Callable[[Any], Any]
    linkinv: Callable[[Any], Any]
    mu_eta: Callable[[Any], Any]
    variance: Callable[[Any], Any]
    deviance: Callable[[Any, Any], Any]
    mu_from_y: Callable[[Any], Any]


def resolve_gpu_family(name: str) -> GPUFamilyOps:
    """Return GPU family ops for a supported family name.

    Raises
    ------
    ValueError
        If the family is not one of gaussian / binomial / poisson / gamma.
        The CPU GAM path still supports any family the library accepts;
        unsupported families should route through ``backend='cpu'``.
    """
    import torch

    name_lc = name.lower()

    if name_lc == "gaussian":
        def _dev(y, mu):
            r = y - mu
            return (r * r).sum()

        return GPUFamilyOps(
            name="gaussian", link_name="identity",
            link_fn=lambda mu: mu,
            linkinv=lambda eta: eta,
            mu_eta=lambda eta: torch.ones_like(eta),
            variance=lambda mu: torch.ones_like(mu),
            deviance=_dev,
            mu_from_y=lambda y: y.clone(),
        )

    if name_lc == "binomial":
        # Logit-link binomial (proportions / binary y). Matches GEE.
        def _linkinv(eta):
            return torch.sigmoid(eta)

        def _mu_eta(eta):
            s = torch.sigmoid(eta)
            return s * (1.0 - s)

        def _variance(mu):
            return mu * (1.0 - mu)

        def _link_fn(mu):
            mu_c = torch.clamp(mu, min=1e-10, max=1.0 - 1e-10)
            return torch.log(mu_c / (1.0 - mu_c))

        def _dev(y, mu):
            # 2 * sum[y log(y/mu) + (1-y) log((1-y)/(1-mu))], with 0 log 0 = 0.
            mu_c = torch.clamp(mu, min=1e-10, max=1.0 - 1e-10)
            term_y = torch.where(
                y > 0, y * torch.log(y / mu_c), torch.zeros_like(y),
            )
            term_1my = torch.where(
                (1.0 - y) > 0,
                (1.0 - y) * torch.log((1.0 - y) / (1.0 - mu_c)),
                torch.zeros_like(y),
            )
            return 2.0 * (term_y + term_1my).sum()

        return GPUFamilyOps(
            name="binomial", link_name="logit",
            link_fn=_link_fn, linkinv=_linkinv, mu_eta=_mu_eta,
            variance=_variance, deviance=_dev,
            mu_from_y=lambda y: (y + 0.5) / 2.0,
        )

    if name_lc == "poisson":
        def _dev(y, mu):
            mu_c = torch.clamp(mu, min=1e-10)
            term = torch.where(
                y > 0, y * torch.log(y / mu_c), torch.zeros_like(y),
            )
            return 2.0 * (term - (y - mu_c)).sum()

        return GPUFamilyOps(
            name="poisson", link_name="log",
            link_fn=lambda mu: torch.log(torch.clamp(mu, min=1e-10)),
            linkinv=lambda eta: torch.exp(eta),
            mu_eta=lambda eta: torch.exp(eta),
            variance=lambda mu: mu,
            deviance=_dev,
            mu_from_y=lambda y: y + 0.1,
        )

    if name_lc == "gamma":
        def _dev(y, mu):
            mu_c = torch.clamp(mu, min=1e-10)
            y_c = torch.clamp(y, min=1e-10)
            return 2.0 * (-torch.log(y_c / mu_c) + (y - mu_c) / mu_c).sum()

        return GPUFamilyOps(
            name="gamma", link_name="inverse",
            link_fn=lambda mu: 1.0 / torch.clamp(mu, min=1e-10),
            linkinv=lambda eta: 1.0 / eta,
            mu_eta=lambda eta: -1.0 / (eta * eta),
            variance=lambda mu: mu * mu,
            deviance=_dev,
            mu_from_y=lambda y: torch.clamp(y, min=1e-2),
        )

    raise ValueError(
        f"GPU GAM: unsupported family {name!r}. Supported: "
        "gaussian, binomial, poisson, gamma."
    )
