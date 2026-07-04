"""
Factored (ma1, sma1, ...) parameterization for SARIMA optimization.

For seasonal ARIMA the "effective" MA polynomial has length q + sq * m
with most coefficients zero or deterministic products of the non-
seasonal and seasonal factors. Optimizing over the full effective
vector wastes scipy's finite-difference budget on a 13-dimensional
surface when only 2 parameters are truly free (for the Box-Jenkins
airline model). R's ``stats::arima`` optimizes in the factored space;
this module brings the same pattern to pystatistics.

Key functions:
    ``_factored_to_effective(...)``  — expand (ar, ma, sar, sma, mean)
        to the effective (ar_eff, ma_eff, mean) vector the likelihood
        operates on.
    ``arima_negloglik_factored(...)`` — NLL in factored space.
    ``optimize_arima_factored(...)``  — L-BFGS-B optimize in factored
        space, delegating the actual CSS/ML computation to the effective
        likelihood.

Sign conventions (critical — see the ``_multiply_ma_polynomials``
docstring below):

    AR:  e_t = y_t - Σ ar_i y_{t-i}       (AR polynomial 1 − Σ ar_i B^i)
    MA:  e_t = y_t - Σ ma_j e_{t-j}       (MA polynomial 1 + Σ ma_j B^j)

so the seasonal-expansion helpers for AR and MA differ by a sign on the
cross term.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from pystatistics.core.exceptions import ConvergenceError
from pystatistics.timeseries._arima_likelihood import arima_negloglik


# ---------------------------------------------------------------------------
# Seasonal polynomial multiplication
# ---------------------------------------------------------------------------

def _multiply_polynomials(
    nonseasonal: NDArray,
    seasonal: NDArray,
    period: int,
) -> NDArray:
    """
    Multiply non-seasonal and seasonal polynomials.

    For AR polynomials, computes the product:
        phi(B) * Phi(B^m) = (1 - phi_1*B - ... - phi_p*B^p)
                          * (1 - Phi_1*B^m - ... - Phi_P*B^{Pm})

    and returns the coefficients of the result (excluding the leading 1).

    Parameters
    ----------
    nonseasonal : NDArray
        Non-seasonal coefficients [c_1, ..., c_p] where the polynomial
        is ``1 - c_1*B - ... - c_p*B^p``.
    seasonal : NDArray
        Seasonal coefficients [C_1, ..., C_P] where the polynomial
        is ``1 - C_1*B^m - ... - C_P*B^{Pm}``.
    period : int
        Seasonal period *m*.

    Returns
    -------
    NDArray
        Combined coefficients of the product polynomial (excluding
        the leading 1 term), negated so they follow the sign convention
        ``1 - result_1*B - result_2*B^2 - ...``.
    """
    # Build polynomial representations with leading 1
    # For poly1 = 1 - c_1*B - c_2*B^2 ..., we store [1, -c_1, -c_2, ...]
    poly1 = np.zeros(1 + len(nonseasonal))
    poly1[0] = 1.0
    for i, c in enumerate(nonseasonal):
        poly1[i + 1] = -c

    max_seasonal_lag = len(seasonal) * period
    poly2 = np.zeros(1 + max_seasonal_lag)
    poly2[0] = 1.0
    for i, c in enumerate(seasonal):
        poly2[(i + 1) * period] = -c

    # Convolve the two polynomials
    product = np.convolve(poly1, poly2)

    # Return coefficients after leading 1, negated back to positive convention
    return -product[1:]


def _multiply_ma_polynomials(
    nonseasonal: NDArray,
    seasonal: NDArray,
    period: int,
) -> NDArray:
    """Multiply two MA polynomials under pystatistics' MA sign convention.

    pystatistics stores AR and MA with OPPOSITE sign conventions:
        AR:  e_t = y_t - Σ ar_i y_{t-i}       (AR polynomial 1 − Σ ar_i B^i)
        MA:  e_t = y_t - Σ ma_j e_{t-j}       (MA polynomial 1 + Σ ma_j B^j)

    ``_multiply_polynomials`` above handles the AR case (all signs
    negated). For MA we need the product of
        (1 + ma_1 B + … + ma_q B^q)(1 + sma_1 B^m + … + sma_Q B^{Qm})
    returned as ``[ma_eff_1, ma_eff_2, …]``. A straight convolution of
    ``[1, ma_1, …, ma_q]`` with ``[1, sma_1 at position m, …]`` yields
    this directly — no double negation.
    """
    poly1 = np.zeros(1 + len(nonseasonal))
    poly1[0] = 1.0
    for i, c in enumerate(nonseasonal):
        poly1[i + 1] = c

    max_seasonal_lag = len(seasonal) * period
    poly2 = np.zeros(1 + max_seasonal_lag)
    poly2[0] = 1.0
    for i, c in enumerate(seasonal):
        poly2[(i + 1) * period] = c

    product = np.convolve(poly1, poly2)
    return product[1:]



def normalize_ma_coefficients(ma: NDArray) -> tuple[NDArray, bool]:
    """Return the invertible (canonical) representative of an MA factor.

    The Gaussian likelihood of an MA polynomial ``1 + c_1 z + ... +
    c_q z^q`` is invariant under reflecting any root across the unit
    circle (with a matching sigma2 rescale), so the parameterization is
    only identified up to this choice. The statistical convention is
    the INVERTIBLE representative: it is the fundamental (Wold)
    representation whose disturbances are the innovations actually
    recoverable from data, so sigma2 is the one-step prediction-error
    variance and the MA inversion converges. This is a port of R
    ``stats::arima``'s internal ``maInvert``, which R applies post-fit
    to the fitted non-seasonal and seasonal MA blocks of ML-family
    fits (then recomputes the Hessian and sigma2 at the inverted
    coefficients) — with an added likelihood-invariance guard R does
    not have. Roots
    strictly inside the unit circle are reflected ``r -> 1/conj(r)``;
    roots on (or numerically at) the circle are left alone — the MA
    unit-root pile-up under over-differencing is a genuine boundary
    optimum where both representatives coincide.

    Callers must recompute sigma2 (and anything derived from the
    coefficients) at the returned parameters; the exact-ML profile
    sigma2 absorbs the rescale automatically. Only valid for exact-ML
    objectives — the CSS criterion is NOT reflection-invariant.

    NOTE: for pathologically clustered high-order roots (e.g. several
    near-identical pairs hugging the circle) np.roots' conditioning can
    spread the cluster by ~1e-3, so strict invertibility of the output
    is not guaranteed in that corner; the caller's likelihood-invariance
    guard bounds any damage. R's polyroot-based ``maInvert`` shares the
    same conditioning limit and runs UNGUARDED, so this corner is at
    parity-or-better with the reference.

    Parameters
    ----------
    ma : NDArray
        MA factor coefficients ``[c_1, ..., c_q]`` (sign convention
        ``1 + c_1 z + ...``). May be empty.

    Returns
    -------
    tuple[NDArray, bool]
        The invertible coefficients and whether any root was flipped.
    """
    q = len(ma)
    if q == 0:
        return ma, False

    # Roots of 1 + c_1 z + ... + c_q z^q (np.roots wants descending).
    poly = np.concatenate((ma[::-1], [1.0]))
    roots = np.roots(poly)
    # Leave near-circle roots untouched: reflecting them is numerically
    # a no-op that would only churn boundary fits.
    inside = np.abs(roots) < 1.0 - 1e-8
    if not np.any(inside):
        return ma, False

    roots = roots.copy()
    roots[inside] = 1.0 / np.conj(roots[inside])

    # Rebuild 1 + c'_1 z + ...: product of (1 - z / r_j).
    coefs = np.array([1.0 + 0.0j])
    for r in roots:
        coefs = np.convolve(coefs, np.array([1.0, -1.0 / r]))
    new_ma = coefs[1:]
    # Reflected complex roots keep their conjugate pairing, so the
    # product is real up to float noise. Anything larger means the
    # root set was corrupted — fail loud rather than return a mangled
    # polynomial.
    if np.abs(new_ma.imag).max() > 1e-8:
        raise ConvergenceError(
            "MA invertibility normalization produced complex "
            "coefficients; the fitted MA polynomial is numerically "
            "degenerate",
            iterations=0,
            reason="ma_normalization_failed",
        )
    out = new_ma.real
    # np.roots drops leading zero coefficients (degenerate degree);
    # pad back to the requested order.
    if len(out) < q:
        out = np.concatenate((out, np.zeros(q - len(out))))
    return out, True


def normalize_to_invertible(
    opt_params: NDArray,
    opt_factored: NDArray | None,
    nll: float,
    y_diff: NDArray,
    p: int, q: int, sp: int, sq: int, period: int,
    include_mean: bool,
) -> tuple[NDArray, NDArray | None, float]:
    """Normalize an exact-ML optimum to the invertible MA representative.

    The exact-ML likelihood is invariant under reflecting MA roots
    across the unit circle (with a matching sigma2 rescale), so the
    optimizer may land on the non-invertible mirror (theta -> 1/theta)
    of the standard solution: same fit, but the reported sigma2 is then
    the variance of a NON-fundamental noise (not the one-step
    prediction-error variance) and the CSS residual recursion diverges.
    This applies :func:`normalize_ma_coefficients` to each MA factor
    (preserving the multiplicative seasonal structure) and accepts the
    result only if the exact-ML objective is numerically unchanged —
    reflection is likelihood-invariant in exact arithmetic, so anything
    beyond float noise means the flip went wrong and the original
    representation is kept with a loud warning.

    Callers must recompute sigma2/residuals/vcov at the returned
    parameters. Only valid for exact-ML optima: the CSS criterion is
    not reflection-invariant, so callers must not pass CSS fits.

    Parameters
    ----------
    opt_params : NDArray
        Effective parameter vector ``[ar_eff, ma_eff, mean?]``.
    opt_factored : NDArray or None
        Factored vector ``[ar, ma, sar, sma, mean?]`` for seasonal
        fits; ``None`` for non-seasonal fits.
    nll : float
        Negative log-likelihood at the optimum.
    y_diff : NDArray
        Differenced series.
    p, q, sp, sq, period : int
        Factored SARIMA orders and period.
    include_mean : bool
        Whether the parameter vectors carry a trailing mean.

    Returns
    -------
    tuple[NDArray, NDArray | None, float]
        ``(opt_params, opt_factored, nll)`` — normalized, or the
        originals if nothing flipped or the invariance guard failed.
    """
    p_eff = p + sp * period
    q_eff = q + sq * period

    if opt_factored is not None:
        ma_norm, flip1 = normalize_ma_coefficients(opt_factored[p:p + q])
        sma_norm, flip2 = normalize_ma_coefficients(
            opt_factored[p + q + sp:p + q + sp + sq]
        )
        if not (flip1 or flip2):
            return opt_params, opt_factored, nll
        cand_factored: NDArray | None = opt_factored.copy()
        cand_factored[p:p + q] = ma_norm
        cand_factored[p + q + sp:p + q + sp + sq] = sma_norm
        cand_params = _factored_to_effective(
            cand_factored, p, q, sp, sq, period, include_mean,
            _multiply_polynomials, _multiply_ma_polynomials,
        )
    else:
        ma_norm, flipped = normalize_ma_coefficients(
            opt_params[p_eff:p_eff + q_eff]
        )
        if not flipped:
            return opt_params, opt_factored, nll
        cand_factored = None
        cand_params = opt_params.copy()
        cand_params[p_eff:p_eff + q_eff] = ma_norm

    nll_norm = arima_negloglik(
        cand_params, y_diff, (p_eff, q_eff), include_mean, "ML",
    )
    if abs(nll_norm - nll) <= 1e-6 * (1.0 + abs(nll)):
        return cand_params, cand_factored, nll_norm

    warnings.warn(
        "MA invertibility normalization changed the log-likelihood; "
        "keeping the non-invertible representation",
        stacklevel=3,
    )
    return opt_params, opt_factored, nll


def _factored_to_effective(
    factored: NDArray,
    p: int, q: int, sp: int, sq: int, period: int,
    include_mean: bool,
    multiply_ar_polynomials,
    multiply_ma_polynomials,
) -> NDArray:
    """Expand factored SARIMA parameters to the effective ARMA vector.

    Factored layout: ``[ar_1..ar_p, ma_1..ma_q, sar_1..sar_P,
    sma_1..sma_Q, mean?]``. Effective layout:
    ``[ar_eff_1..ar_eff_{p_eff}, ma_eff_1..ma_eff_{q_eff}, mean?]``
    with p_eff = p + sp*period, q_eff = q + sq*period.

    AR and MA expansion helpers are passed in from ``_arima_fit`` so
    this module doesn't import polynomial math from its parent.
    """
    ar = factored[:p]
    ma = factored[p:p + q]
    sar = factored[p + q:p + q + sp]
    sma = factored[p + q + sp:p + q + sp + sq]

    ar_eff = multiply_ar_polynomials(ar, sar, period) if sp > 0 else ar
    ma_eff = multiply_ma_polynomials(ma, sma, period) if sq > 0 else ma

    parts = [ar_eff, ma_eff]
    if include_mean:
        parts.append(factored[p + q + sp + sq:p + q + sp + sq + 1])
    return np.concatenate(parts) if parts else np.array([])


def arima_negloglik_factored(
    factored: NDArray,
    y: NDArray,
    shape: tuple[int, int, int, int],  # (p, q, sp, sq)
    period: int,
    include_mean: bool,
    method: str,
    multiply_ar_polynomials,
    multiply_ma_polynomials,
) -> float:
    """Negative log-likelihood at factored SARIMA parameters.

    Expands to effective ARMA form (via the supplied polynomial-product
    helpers) and delegates to ``arima_negloglik``. Used by
    ``optimize_arima_factored`` so scipy's L-BFGS-B optimizes on the
    small factored surface.
    """
    p, q, sp, sq = shape
    eff_params = _factored_to_effective(
        factored, p, q, sp, sq, period, include_mean,
        multiply_ar_polynomials, multiply_ma_polynomials,
    )
    p_eff = p + sp * period
    q_eff = q + sq * period
    return arima_negloglik(
        eff_params, y, (p_eff, q_eff), include_mean, method,
    )


def optimize_arima_factored(
    y_diff: NDArray,
    p: int, q: int, sp: int, sq: int, period: int,
    include_mean: bool,
    method: str,
    tol: float,
    max_iter: int,
    start_factored: NDArray,
    multiply_ar_polynomials,
    multiply_ma_polynomials,
) -> tuple[NDArray, float, bool, int, str]:
    """L-BFGS-B on the factored SARIMA parameter vector.

    Same semantics as ``_optimize_arima`` (CSS, ML, CSS-ML pipelines),
    but the optimizer sees (p + q + sp + sq + [1]) parameters instead
    of (p_eff + q_eff + [1]). On the Box-Jenkins airline model that is
    2 dims vs 14; the likelihood is the same and the per-evaluation
    cost is the same, so the savings come from ~10× fewer scipy
    evaluations to converge.
    """
    order = (p, q, sp, sq)
    ll_args = (
        y_diff, order, period, include_mean,
    )
    opts = {"maxiter": max_iter, "ftol": tol}

    if method in ("CSS", "CSS-ML"):
        result_css = minimize(
            arima_negloglik_factored,
            start_factored,
            args=ll_args + ("CSS", multiply_ar_polynomials, multiply_ma_polynomials),
            method="L-BFGS-B",
            options=opts,
        )
        opt_factored = result_css.x
        nll = result_css.fun
        converged = result_css.success
        n_iter = result_css.nit
        method_used = "CSS"

        if method == "CSS-ML":
            css_factored = opt_factored.copy()
            try:
                result_ml = minimize(
                    arima_negloglik_factored,
                    css_factored,
                    args=ll_args + ("ML", multiply_ar_polynomials, multiply_ma_polynomials),
                    method="L-BFGS-B",
                    options=opts,
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise ConvergenceError(
                    "ARIMA CSS-ML: ML refinement failed numerically "
                    f"({type(exc).__name__}: {exc}).",
                    iterations=n_iter,
                    reason="ml_refinement_failed",
                )
            if not np.isfinite(result_ml.fun):
                raise ConvergenceError(
                    "ARIMA CSS-ML: ML refinement produced a non-finite "
                    "log-likelihood.",
                    iterations=result_ml.nit,
                    reason="ml_refinement_nonfinite",
                )

            best_x = result_ml.x
            best_nll = result_ml.fun
            best_success = result_ml.success
            total_nit = result_ml.nit

            # Second ML start from the original starting values, kept
            # only if strictly better — the (ar, mean) surface has a
            # flat canyon toward the AR unit root and the CSS stage can
            # hand ML a drifted-mean basin (see _optimize_arima). Only
            # mean-carrying fits (d = D = 0) are exposed.
            if include_mean:
                try:
                    result_ml2 = minimize(
                        arima_negloglik_factored,
                        start_factored,
                        args=ll_args + ("ML", multiply_ar_polynomials,
                                        multiply_ma_polynomials),
                        method="L-BFGS-B",
                        options=opts,
                    )
                except (ValueError, np.linalg.LinAlgError):
                    result_ml2 = None
                if result_ml2 is not None:
                    total_nit += result_ml2.nit
                    # Take the second optimum when it is strictly
                    # better, or when it converged and the first start
                    # did not (a converged optimum in hand must win
                    # over a failed one regardless of its nll).
                    if (result_ml2.success and np.isfinite(result_ml2.fun)
                            and (result_ml2.fun < best_nll
                                 or not best_success)):
                        best_x = result_ml2.x
                        best_nll = result_ml2.fun
                        best_success = True

            return best_x, best_nll, best_success, total_nit, "CSS-ML"
        return opt_factored, nll, converged, n_iter, method_used

    # method == "ML"
    result_ml = minimize(
        arima_negloglik_factored,
        start_factored,
        args=ll_args + ("ML", multiply_ar_polynomials, multiply_ma_polynomials),
        method="L-BFGS-B",
        options=opts,
    )
    return (
        result_ml.x, result_ml.fun, result_ml.success, result_ml.nit, "ML",
    )
