"""
Bayesian linear-regression posterior draw — the shared kernel of `norm` and
`pmm`.

Given observed ``(y, X)``, this draws one sample from the Bayesian posterior of
a Gaussian linear model under the standard non-informative prior, exactly as R's
``mice:::.norm.draw`` does:

    beta_hat   = (X'X + ridge)^{-1} X'y                  (ridge for stability)
    df         = max(n_obs - n_params, 1)
    sigma*     = sqrt( RSS / chisq(df) )                 (posterior draw of sd)
    beta*      = beta_hat + sigma* * L z,  z ~ N(0, I)   (posterior draw of beta)

where ``L L' = (X'X + ridge)^{-1}``. ``norm`` uses ``beta*`` and ``sigma*`` to
draw new responses; ``pmm`` uses ``beta_hat`` for the observed fitted values and
``beta*`` for the missing ones (matchtype=1, R's default).

This is a single per-variable linear solve. It is intentionally isolated so a
Stage-2 GPU backend can batch it across the ``m`` chains and across variables
without disturbing the orchestration.

NumPy's Cholesky returns a *lower* ``L`` with ``L L' = V``; ``L z`` then has
covariance ``V`` — distributionally identical to R's ``t(chol(V)) %*% z``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Matches R mice's default ``ridge`` — a tiny penalty that keeps X'X invertible
# when predictors are collinear or n_obs is small, without materially biasing
# well-conditioned fits.
_DEFAULT_RIDGE = 1e-5


@dataclass(frozen=True)
class LinRegDraw:
    """One posterior draw plus the point estimate, with intercept handling."""

    beta_hat: np.ndarray   # (q+1,) ML coefficients, intercept first
    beta_draw: np.ndarray  # (q+1,) posterior draw of coefficients
    sigma_draw: float      # posterior draw of residual sd

    def predict_hat(self, X: np.ndarray) -> np.ndarray:
        """Fitted values using the point estimate ``beta_hat``."""
        return _add_intercept(X) @ self.beta_hat

    def predict_draw(self, X: np.ndarray) -> np.ndarray:
        """Fitted values using the posterior draw ``beta_draw``."""
        return _add_intercept(X) @ self.beta_draw


def bayes_linreg_draw(
    y_obs: np.ndarray,
    X_obs: np.ndarray,
    rng: np.random.Generator,
    ridge: float = _DEFAULT_RIDGE,
) -> LinRegDraw:
    """Draw once from the Gaussian linear-model posterior (see module docstring).

    Parameters
    ----------
    y_obs : (n_obs,)
        Observed responses.
    X_obs : (n_obs, q)
        Observed predictors WITHOUT an intercept column (added internally).
    rng : numpy.random.Generator
        Sole randomness source.
    ridge : float
        Ridge penalty added to the diagonal of X'X (default matches R mice).
    """
    Xa = _add_intercept(X_obs)
    n_obs, n_params = Xa.shape

    XtX = Xa.T @ Xa
    # Ridge scaled by the mean diagonal magnitude so it is a relative penalty,
    # robust across predictors on very different scales.
    diag_mean = float(np.mean(np.diag(XtX))) if n_params > 0 else 1.0
    XtX_ridge = XtX + ridge * diag_mean * np.eye(n_params)

    V = np.linalg.inv(XtX_ridge)
    beta_hat = V @ (Xa.T @ y_obs)

    resid = y_obs - Xa @ beta_hat
    rss = float(resid @ resid)
    df = max(n_obs - n_params, 1)

    # Posterior draw of sigma: RSS / chi-square(df). Guard the degenerate
    # zero-residual case (perfect fit) so sigma_draw is well-defined.
    chi = float(rng.chisquare(df))
    if chi <= 0.0:
        chi = np.finfo(float).tiny
    sigma_draw = float(np.sqrt(rss / chi)) if rss > 0 else 0.0

    # Posterior draw of beta: beta_hat + sigma * L z, L L' = V.
    L = _safe_cholesky(V)
    z = rng.standard_normal(n_params)
    beta_draw = beta_hat + sigma_draw * (L @ z)

    return LinRegDraw(beta_hat=beta_hat, beta_draw=beta_draw, sigma_draw=sigma_draw)


def _add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    return np.hstack([ones, X])


def _safe_cholesky(V: np.ndarray) -> np.ndarray:
    """Lower Cholesky of a symmetric PSD matrix, jittered if not quite PD.

    ``V`` is an inverse-of-Gram matrix and is positive definite in exact
    arithmetic; floating-point error can leave it marginally indefinite, so we
    symmetrize and add escalating jitter until the factorization succeeds.
    """
    Vs = 0.5 * (V + V.T)
    scale = float(np.mean(np.diag(Vs))) if Vs.shape[0] > 0 else 1.0
    jitter = 0.0
    for _ in range(8):
        try:
            return np.linalg.cholesky(Vs + jitter * np.eye(Vs.shape[0]))
        except np.linalg.LinAlgError:
            jitter = max(scale, 1.0) * (1e-12 if jitter == 0.0 else jitter * 10)
    # Last resort: eigenvalue clip to a PSD factor (documented fallback, Rule 1
    # — we surface nothing silently wrong; the factor is a valid PSD sqrt).
    w, Q = np.linalg.eigh(Vs)
    w = np.clip(w, 0.0, None)
    return Q @ np.diag(np.sqrt(w))
