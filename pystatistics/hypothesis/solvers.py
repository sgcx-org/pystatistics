"""
Solver dispatch for hypothesis tests.

Provides R-named functions: t_test(), chisq_test(), fisher_test(),
wilcox_test(), ks_test(), prop_test(), var_test().

Also re-exports p_adjust() for convenience.
"""

from __future__ import annotations

from typing import Literal
from numpy.typing import ArrayLike

from pystatistics.core.exceptions import ValidationError
from pystatistics.hypothesis.design import HypothesisDesign
from pystatistics.hypothesis.solution import HTestSolution
from pystatistics.hypothesis.backends.cpu import CPUHypothesisBackend
from pystatistics.hypothesis._p_adjust import p_adjust  # re-export


BackendChoice = Literal['cpu', 'gpu']
# GPU is only useful for Monte Carlo simulation (chisq/Fisher).
# All other tests fall back to CPU automatically.


def _get_backend(backend: str = 'cpu'):
    """
    Select backend for hypothesis tests.

    CPU is the default and correct choice for all scalar tests.
    GPU only helps for Monte Carlo simulations (chi-squared and
    Fisher r×c with simulate_p_value=True).
    """
    if backend in ('cpu', 'auto'):
        return CPUHypothesisBackend()
    if backend == 'gpu':
        from pystatistics.hypothesis.backends.gpu import GPUHypothesisBackend
        return GPUHypothesisBackend()
    raise ValidationError(
        f"Unknown backend: {backend!r}. Use 'cpu' or 'gpu'."
    )


def t_test(
    x: ArrayLike | HypothesisDesign,
    y: ArrayLike | None = None,
    *,
    alternative: Literal["two.sided", "less", "greater"] = "two.sided",
    mu: float = 0.0,
    paired: bool = False,
    var_equal: bool = False,
    conf_level: float = 0.95,
    backend: str = 'cpu',
) -> HTestSolution:
    """
    Student's t-test. Matches R t.test().

    Parameters
    ----------
    x : array-like or HypothesisDesign
        Sample data. 1D numeric vector.
    y : array-like or None
        Optional second sample for two-sample test.
    alternative : str
        "two.sided" (default), "less", or "greater".
    mu : float
        Hypothesized mean (one-sample) or difference in means (two-sample).
        Default 0.
    paired : bool
        If True, perform paired t-test. x and y must have same length.
    var_equal : bool
        If True, use pooled variance (Student's t).
        If False (default), use Welch's approximation with
        Welch-Satterthwaite degrees of freedom. **R defaults to Welch.**
    conf_level : float
        Confidence level for the interval. Default 0.95.
    backend : str
        'cpu' (default). Hypothesis tests are CPU-only.

    Returns
    -------
    HTestSolution
        Test result with statistic, p_value, conf_int, estimate, etc.
    """
    if isinstance(x, HypothesisDesign):
        design = x
    else:
        design = HypothesisDesign.for_t_test(
            x, y,
            mu=mu,
            paired=paired,
            var_equal=var_equal,
            alternative=alternative,
            conf_level=conf_level,
        )

    be = _get_backend(backend)
    result = be.solve(design)
    return HTestSolution(_result=result, _design=design)


def chisq_test(
    x: ArrayLike | HypothesisDesign,
    y: ArrayLike | None = None,
    *,
    correct: bool = True,
    p: ArrayLike | None = None,
    rescale_p: bool = False,
    simulate_p_value: bool = False,
    B: int = 2000,
    backend: str = 'cpu',
) -> HTestSolution:
    """
    Pearson's Chi-squared test. Matches R chisq.test().

    Parameters
    ----------
    x : array-like or HypothesisDesign
        A 2D contingency table, or a 1D vector of observed counts
        (for goodness-of-fit test). Can also be a pre-built
        HypothesisDesign.
    y : array-like or None
        If x is 1D and y is provided, a contingency table is built
        from cross-tabulation.
    correct : bool
        Apply Yates' continuity correction for 2x2 tables.
        Default True (matches R).
    p : array-like or None
        Expected proportions for GOF test. If None, assumes uniform.
    rescale_p : bool
        If True, rescale p to sum to 1.
    simulate_p_value : bool
        If True, compute p-value by Monte Carlo simulation.
    B : int
        Number of Monte Carlo replicates. Default 2000.
    backend : str
        'cpu' (default). GPU Monte Carlo will be added later.

    Returns
    -------
    HTestSolution
        Test result with statistic, p_value, and extras
        (observed, expected, residuals, stdres for independence test).
    """
    if isinstance(x, HypothesisDesign):
        design = x
    else:
        design = HypothesisDesign.for_chisq_test(
            x, y,
            correct=correct,
            p=p,
            rescale_p=rescale_p,
            simulate_p_value=simulate_p_value,
            B=B,
        )

    be = _get_backend(backend)
    result = be.solve(design)
    return HTestSolution(_result=result, _design=design)


def prop_test(
    x: ArrayLike | HypothesisDesign,
    n: ArrayLike | None = None,
    *,
    p: ArrayLike | float | None = None,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
    correct: bool = True,
    backend: str = 'cpu',
) -> HTestSolution:
    """
    Test of proportions. Matches R prop.test().

    Parameters
    ----------
    x : array-like or HypothesisDesign
        Number of successes. Scalar or vector.
    n : array-like or None
        Number of trials. Scalar or vector (same length as x).
    p : float or array-like or None
        Null hypothesis proportion(s). If None, tests equality
        of proportions (k >= 2 groups).
    alternative : str
        "two.sided" (default), "less", or "greater".
        Only "two.sided" allowed for k > 1 groups.
    conf_level : float
        Confidence level for the interval. Default 0.95.
    correct : bool
        Apply Yates' continuity correction. Default True.
    backend : str
        'cpu' (default). Hypothesis tests are CPU-only.

    Returns
    -------
    HTestSolution
        Test result with statistic, p_value, conf_int, estimate.
    """
    if isinstance(x, HypothesisDesign):
        design = x
    else:
        if n is None:
            from pystatistics.core.exceptions import ValidationError
            raise ValidationError("n (number of trials) is required for prop_test")
        design = HypothesisDesign.for_prop_test(
            x, n,
            p=p,
            alternative=alternative,
            conf_level=conf_level,
            correct=correct,
        )

    be = _get_backend(backend)
    result = be.solve(design)
    return HTestSolution(_result=result, _design=design)


def fisher_test(
    x: ArrayLike | HypothesisDesign,
    y: ArrayLike | None = None,
    *,
    alternative: str = "two.sided",
    conf_int: bool = True,
    conf_level: float = 0.95,
    simulate_p_value: bool = False,
    B: int = 2000,
    backend: str = 'cpu',
) -> HTestSolution:
    """
    Fisher's Exact Test for Count Data. Matches R fisher.test().

    Parameters
    ----------
    x : array-like or HypothesisDesign
        A 2D contingency table.
    y : array-like or None
        If x is 1D, second factor to cross-tabulate.
    alternative : str
        "two.sided" (default), "less", or "greater".
        Only "two.sided" for r x c (r > 2 or c > 2).
    conf_int : bool
        Compute confidence interval for odds ratio (2x2 only).
    conf_level : float
        Confidence level. Default 0.95.
    simulate_p_value : bool
        Use Monte Carlo for p-value.
    B : int
        Number of Monte Carlo replicates. Default 2000.
    backend : str
        'cpu' (default).

    Returns
    -------
    HTestSolution
        Test result with p_value, estimate (odds ratio for 2x2),
        conf_int (2x2 only).
    """
    if isinstance(x, HypothesisDesign):
        design = x
    else:
        design = HypothesisDesign.for_fisher_test(
            x, y,
            alternative=alternative,
            conf_int=conf_int,
            conf_level=conf_level,
            simulate_p_value=simulate_p_value,
            B=B,
        )

    be = _get_backend(backend)
    result = be.solve(design)
    return HTestSolution(_result=result, _design=design)


def wilcox_test(
    x: ArrayLike | HypothesisDesign,
    y: ArrayLike | None = None,
    *,
    alternative: str = "two.sided",
    mu: float = 0.0,
    paired: bool = False,
    exact: bool | None = None,
    correct: bool = True,
    conf_int: bool = True,
    conf_level: float = 0.95,
    backend: str = 'cpu',
) -> HTestSolution:
    """
    Wilcoxon rank-sum or signed-rank test. Matches R wilcox.test().

    Parameters
    ----------
    x : array-like or HypothesisDesign
        Numeric vector.
    y : array-like or None
        Second sample for rank-sum test, or paired sample.
    alternative : str
        "two.sided" (default), "less", or "greater".
    mu : float
        Hypothesized location (one-sample) or shift (two-sample).
    paired : bool
        If True, perform paired (signed-rank) test.
    exact : bool or None
        If None (default), use exact test for small n without ties.
    correct : bool
        Apply continuity correction for normal approximation.
    conf_int : bool
        Compute Hodges-Lehmann confidence interval.
    conf_level : float
        Confidence level. Default 0.95.
    backend : str
        'cpu' (default).

    Returns
    -------
    HTestSolution
        Test result with statistic (V or W), p_value, conf_int, estimate.
    """
    if isinstance(x, HypothesisDesign):
        design = x
    else:
        design = HypothesisDesign.for_wilcox_test(
            x, y,
            mu=mu,
            paired=paired,
            exact=exact,
            correct=correct,
            conf_int=conf_int,
            conf_level=conf_level,
            alternative=alternative,
        )

    be = _get_backend(backend)
    result = be.solve(design)
    return HTestSolution(_result=result, _design=design)


def ks_test(
    x: ArrayLike | HypothesisDesign,
    y: ArrayLike | None = None,
    *,
    alternative: str = "two.sided",
    distribution: str | None = None,
    backend: str = 'cpu',
    **dist_params: float,
) -> HTestSolution:
    """
    Kolmogorov-Smirnov test. Matches R ks.test().

    Parameters
    ----------
    x : array-like or HypothesisDesign
        Numeric vector of observations.
    y : array-like or None
        Second sample for two-sample test. If None, performs
        one-sample test against a theoretical distribution.
    alternative : str
        "two.sided" (default), "less", or "greater".
    distribution : str or None
        Distribution name for one-sample test ("norm", "unif", "exp").
        If None and y is None, defaults to standard normal.
    backend : str
        'cpu' (default). Hypothesis tests are CPU-only.
    **dist_params : float
        Distribution parameters (e.g., mean=0, sd=1 for "norm").

    Returns
    -------
    HTestSolution
        Test result with statistic (D), p_value.
    """
    if isinstance(x, HypothesisDesign):
        design = x
    else:
        design = HypothesisDesign.for_ks_test(
            x, y,
            alternative=alternative,
            distribution=distribution,
            **dist_params,
        )

    be = _get_backend(backend)
    result = be.solve(design)
    return HTestSolution(_result=result, _design=design)


def var_test(
    x: ArrayLike | HypothesisDesign,
    y: ArrayLike | None = None,
    *,
    ratio: float = 1.0,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
    backend: str = 'cpu',
) -> HTestSolution:
    """
    F-test to compare two variances. Matches R var.test().

    Parameters
    ----------
    x : array-like or HypothesisDesign
        First sample.
    y : array-like or None
        Second sample. Required unless x is a HypothesisDesign.
    ratio : float
        Hypothesized ratio of variances (var_x / var_y). Default 1.
    alternative : str
        "two.sided" (default), "less", or "greater".
    conf_level : float
        Confidence level. Default 0.95.
    backend : str
        'cpu' (default). Hypothesis tests are CPU-only.

    Returns
    -------
    HTestSolution
        Test result with statistic (F), p_value, conf_int,
        estimate (ratio of variances).
    """
    if isinstance(x, HypothesisDesign):
        design = x
    else:
        if y is None:
            raise ValidationError("y is required for var_test")
        design = HypothesisDesign.for_var_test(
            x, y,
            ratio=ratio,
            alternative=alternative,
            conf_level=conf_level,
        )

    be = _get_backend(backend)
    result = be.solve(design)
    return HTestSolution(_result=result, _design=design)
