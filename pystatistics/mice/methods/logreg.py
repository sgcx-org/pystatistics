"""
``logreg`` — Bayesian logistic-regression imputation for binary columns.

R mice's default for two-level factors. Fit a logistic regression of the binary
target on the predictors, draw the coefficients once from the posterior normal
approximation ``N(beta_hat, (X'WX)^{-1})``, then impute each missing row by a
Bernoulli draw with the predicted probability under the drawn coefficients.

Self-contained: a compact ridge-stabilised IRLS gives both the coefficients and
their covariance ``(X'WX)^{-1}`` directly, so this method does not depend on the
internals of the regression module. The target arrives as 0/1 class indices (the
chain maps the column's two category codes to indices) and the method returns
0/1 indices.
"""

from __future__ import annotations

import numpy as np

from pystatistics.mice._encode import add_intercept
from pystatistics.mice.methods._draw import mvn_draw
from pystatistics.mice.methods.registry import register

# Ridge on X'WX, matching the numeric path's stabiliser; also tames perfect /
# quasi-complete separation, where the unpenalised MLE diverges.
_RIDGE = 1e-5
_MAX_IRLS_ITER = 50
_IRLS_TOL = 1e-8
# Clamp the linear predictor so exp() can't overflow and weights stay positive.
_ETA_CLIP = 30.0


class LogregMethod:
    """Bayesian logistic-regression imputation (conforms to ImputationMethod)."""

    name = "logreg"
    target_kind = "binary"

    def impute(
        self,
        y_obs: np.ndarray,
        X_obs: np.ndarray,
        X_mis: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        y = np.asarray(y_obs, dtype=np.float64)  # 0/1 indices
        beta_hat, cov = _fit_logistic(y, X_obs)
        beta_star = mvn_draw(beta_hat, cov, rng)

        eta = np.clip(add_intercept(X_mis) @ beta_star, -_ETA_CLIP, _ETA_CLIP)
        p = 1.0 / (1.0 + np.exp(-eta))
        return (rng.random(p.shape[0]) < p).astype(np.intp)


def _fit_logistic(y: np.ndarray, X: np.ndarray):
    """Ridge-stabilised IRLS for logistic regression.

    Returns ``(beta_hat, cov)`` where ``cov = (X'WX + ridge)^{-1}`` is the
    coefficient covariance used for the posterior draw.
    """
    Xa = add_intercept(X)
    n, k = Xa.shape
    beta = np.zeros(k, dtype=np.float64)

    # Relative ridge, scaled by the predictor cross-product magnitude.
    diag_scale = float(np.mean(np.sum(Xa * Xa, axis=0))) / max(n, 1)
    ridge = _RIDGE * max(diag_scale, 1e-12) * np.eye(k)

    XtWX = ridge  # bound in case the loop body never runs (k==0 is impossible)
    for _ in range(_MAX_IRLS_ITER):
        eta = np.clip(Xa @ beta, -_ETA_CLIP, _ETA_CLIP)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-9, None)
        XtWX = Xa.T @ (w[:, None] * Xa) + ridge
        grad = Xa.T @ (y - p)
        delta = np.linalg.solve(XtWX, grad)
        beta = beta + delta
        if np.max(np.abs(delta)) < _IRLS_TOL:
            break

    cov = np.linalg.inv(XtWX)
    return beta, cov


register(LogregMethod())
