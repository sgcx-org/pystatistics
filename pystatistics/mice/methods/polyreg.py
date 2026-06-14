"""
``polyreg`` — multinomial logistic-regression imputation for unordered
categorical columns (R mice's default for unordered factors with >2 levels).

Fits a multinomial logit of the categorical target on the predictors (reusing
pystatistics' R-validated ``multinomial`` module), draws the coefficient block
once from its posterior normal approximation, predicts class probabilities for
the missing rows under the drawn coefficients, and samples a category per row.

The target arrives as consecutive ``0..K-1`` class indices and the method returns
indices; the chain handles the mapping to/from the column's category codes.
"""

from __future__ import annotations

import warnings

import numpy as np

from pystatistics.core.exceptions import ConvergenceError, NumericalError
from pystatistics.mice._encode import add_intercept
from pystatistics.mice.methods._draw import (
    marginal_indices,
    mvn_draw,
    sample_categories,
)
from pystatistics.mice.methods.registry import register


class PolyregMethod:
    """Multinomial logistic imputation (conforms to ImputationMethod)."""

    name = "polyreg"
    target_kind = "categorical"

    def impute(
        self,
        y_obs: np.ndarray,
        X_obs: np.ndarray,
        X_mis: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        # Imported lazily: the multinomial module pulls in heavier dependencies
        # that a numeric-only MICE run should not pay for.
        from pystatistics.multinomial import multinom
        from pystatistics.multinomial._likelihood import compute_probs

        y = np.asarray(y_obs, dtype=np.intp)
        Xa_obs = add_intercept(X_obs)  # multinom carries an explicit intercept
        Xa_mis = add_intercept(X_mis)

        # A multinomial fit can fail to converge on an awkward intermediate
        # sweep state; fall back to a marginal draw (visibly). Rule-1 documented
        # exception: local, not silent, retried next iteration.
        try:
            fit = multinom(y, Xa_obs)
        except (ConvergenceError, NumericalError, np.linalg.LinAlgError) as exc:
            warnings.warn(
                f"polyreg fit did not converge ({type(exc).__name__}); using a "
                f"marginal draw for this sweep step.",
                UserWarning,
                stacklevel=2,
            )
            return marginal_indices(y, Xa_mis.shape[0], rng)

        coef = fit.coefficient_matrix          # (K-1, q+1)
        # Draw the whole coefficient block from N(coef, vcov); vcov is ordered to
        # match coef.ravel(), as is compute_probs' expected parameter vector.
        beta_star = mvn_draw(coef.ravel(), fit.vcov, rng)
        probs = compute_probs(beta_star, Xa_mis, fit.n_classes)  # (n_mis, K)
        return sample_categories(probs, rng)


register(PolyregMethod())
