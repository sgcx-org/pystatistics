"""
``polr`` — proportional-odds (ordinal logistic) imputation for ordered
categorical columns (R mice's default for ordered factors).

Fits a proportional-odds model of the ordered target on the predictors (reusing
pystatistics' R-validated ``ordinal`` module) with a small ridge penalty on the
slopes, draws the threshold and slope parameters jointly from their posterior
normal approximation, computes category probabilities for the missing rows, and
samples a category per row.

The ridge matters here because chained equations routinely drive an ordinal
column into (quasi-)complete separation against continuous predictors. Without
it the unpenalised slopes diverge, the optimizer burns its full iteration budget,
and the fit is rejected for a predictor-blind marginal draw — silently degrading
``polr`` to a non-conditional imputer on exactly the fits where it is selected.
The penalty keeps the fit finite, fast, and predictor-aware; the GPU backend
applies the equivalent stabilization in ``backends/_gpu_polr.py``.

The target arrives as consecutive ``0..K-1`` class indices (ordered) and the
method returns indices; the chain maps to/from the column's category codes.

Drawn thresholds are not guaranteed to stay ordered, which can make an
individual category probability slightly negative; ``sample_categories`` clips
and renormalises, so sampling stays valid.
"""

from __future__ import annotations

import warnings

import numpy as np

from pystatistics.core.exceptions import ConvergenceError, NumericalError
from pystatistics.mice.methods._draw import (
    marginal_indices,
    mvn_draw,
    sample_categories,
)
from pystatistics.mice.methods.registry import register

# Relative ridge on the slope coefficients, mirroring the logreg/polyreg
# near-separation stabiliser. Chained equations routinely drive an ordinal
# column into (quasi-)complete separation against continuous predictors (sparse
# extreme categories perfectly ordered by a numeric covariate); the unpenalised
# proportional-odds slopes then diverge, L-BFGS-B runs to max_iter, and the fit
# is rejected for a predictor-blind marginal draw. The penalty
# 0.5 * lambda * ||beta||^2 keeps the fit finite and predictor-aware. It is
# scaled by the predictor cross-product magnitude AND by n so that, relative to
# the n-scaled likelihood curvature, it stays ~_RIDGE (negligible) in
# well-identified directions while bounding the runaway in separated ones — the
# latter is what lets L-BFGS-B converge in tens of iterations instead of 200.
_RIDGE = 1e-5


def _slope_ridge(X_obs: np.ndarray) -> float:
    """Scale-aware ridge coefficient for the proportional-odds slopes.

    ``lambda = _RIDGE * mean_column_second_moment * n_obs``. The mean column
    second moment makes it invariant to predictor scaling; the ``n_obs`` factor
    puts it on the same scale as the (n-scaled) likelihood information so the
    relative penalty is ~``_RIDGE`` regardless of sample size.
    """
    n_obs = X_obs.shape[0]
    if n_obs == 0 or X_obs.shape[1] == 0:
        return 0.0
    diag_scale = float(np.mean(np.sum(X_obs * X_obs, axis=0))) / n_obs
    return _RIDGE * max(diag_scale, 1e-12) * n_obs


class PolrMethod:
    """Proportional-odds ordinal imputation (conforms to ImputationMethod)."""

    name = "polr"
    target_kind = "ordered"

    def impute(
        self,
        y_obs: np.ndarray,
        X_obs: np.ndarray,
        X_mis: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        from pystatistics.ordinal import polr as fit_polr
        from pystatistics.ordinal._likelihood import _cumulative_probs_vectorized
        from pystatistics.regression.families import LogitLink

        y = np.asarray(y_obs, dtype=np.intp)
        n_levels = int(y.max()) + 1  # all K levels are present among observed
        X_obs = np.asarray(X_obs, dtype=np.float64)
        X_mis = np.asarray(X_mis, dtype=np.float64)

        # polr carries thresholds as intercepts, so X has no intercept column.
        # A small ridge on the slopes keeps the fit finite and well-conditioned
        # under the (quasi-)separation chained equations routinely induce, so the
        # optimizer converges quickly and the conditional model stays usable.
        # The marginal-draw fallback below is retained as a last-resort guard for
        # genuinely degenerate sub-problems (e.g. n_obs < n_params); with the
        # ridge it should essentially never fire. Rule-1 documented exception:
        # not silent, and local (the next sweep retries the full model).
        ridge = _slope_ridge(X_obs)
        try:
            fit = fit_polr(y, X_obs, ridge=ridge)
        except (ConvergenceError, NumericalError, np.linalg.LinAlgError) as exc:
            warnings.warn(
                f"polr fit did not converge ({type(exc).__name__}); using a "
                f"marginal draw for this sweep step.",
                UserWarning,
                stacklevel=2,
            )
            return marginal_indices(y, X_mis.shape[0], rng)

        alpha = fit.threshold_values        # (K-1,)
        beta = fit.coefficients             # (q,)

        # Joint posterior draw of [thresholds, slopes]; vcov is in that order.
        n_thresh = alpha.shape[0]
        theta_star = mvn_draw(np.concatenate([alpha, beta]), fit.vcov, rng)
        alpha_star = theta_star[:n_thresh]
        beta_star = theta_star[n_thresh:]

        eta = X_mis @ beta_star
        probs = _cumulative_probs_vectorized(
            alpha_star, eta, LogitLink(), n_levels
        )                                    # (n_mis, K)
        return sample_categories(np.asarray(probs), rng)


register(PolrMethod())
