"""
``norm`` — Bayesian linear regression imputation.

Draws the missing values from the posterior predictive distribution of a
Gaussian linear model: fit ``y ~ X`` on the observed rows, draw ``(beta*,
sigma*)`` from the posterior, then for each missing row draw

    y_imp = X_mis beta* + sigma* * eps,   eps ~ N(0, 1).

This is R mice's ``norm`` method. It is fully parametric (no donors), so imputed
values can fall outside the observed range — appropriate when the linear-normal
model is plausible. ``pmm`` (the default) is the semi-parametric alternative.
"""

from __future__ import annotations

import numpy as np

from pystatistics.mice.methods._linreg import bayes_linreg_draw
from pystatistics.mice.methods.registry import register


class NormMethod:
    """Bayesian linear-regression imputation (conforms to ImputationMethod)."""

    name = "norm"
    target_kind = "numeric"

    def impute(
        self,
        y_obs: np.ndarray,
        X_obs: np.ndarray,
        X_mis: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        draw = bayes_linreg_draw(y_obs, X_obs, rng)
        yhat_mis = draw.predict_draw(X_mis)
        noise = rng.standard_normal(yhat_mis.shape[0])
        return yhat_mis + draw.sigma_draw * noise


register(NormMethod())
