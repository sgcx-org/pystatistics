"""
``polr`` — proportional-odds (ordinal logistic) imputation for ordered
categorical columns (R mice's default for ordered factors).

Fits a proportional-odds model of the ordered target on the predictors (reusing
pystatistics' R-validated ``ordinal`` module), draws the threshold and slope
parameters jointly from their posterior normal approximation, computes category
probabilities for the missing rows, and samples a category per row.

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
        X_mis = np.asarray(X_mis, dtype=np.float64)

        # polr carries thresholds as intercepts, so X has no intercept column.
        # An ordinal fit can fail to converge on an awkward intermediate sweep
        # state; fall back to a marginal draw (visibly), as R's mice does for
        # MASS::polr. Rule-1 documented exception: not silent, and local.
        try:
            fit = fit_polr(y, np.asarray(X_obs, dtype=np.float64))
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
