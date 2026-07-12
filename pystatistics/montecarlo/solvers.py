"""
Solver dispatch for Monte Carlo methods.

Provides boot(), boot_ci(), and permutation_test() as the public API.
"""

from __future__ import annotations

from pystatistics.core.exceptions import ValidationError

from typing import Any, Callable, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pystatistics.core.compute.backend import resolve_backend
from pystatistics.montecarlo._common import BootParams
from pystatistics.montecarlo.design import BootstrapDesign, PermutationDesign
from pystatistics.montecarlo.solution import BootstrapSolution, PermutationSolution
from pystatistics.montecarlo.backends.cpu import (
    CPUBootstrapBackend,
    CPUPermutationBackend,
)


# Monte-Carlo resampling has no GPU float64 path (replicate counts dominate,
# not precision); the honest subset omits 'gpu_fp64'.
BackendChoice = Literal['auto', 'cpu', 'gpu']

# Public ``statistic_type`` values map to the backends' internal R-style codes.
_STATISTIC_TYPE_CODE = {"index": "i", "frequency": "f", "weight": "w"}


def _use_gpu(backend: BackendChoice | None) -> bool:
    """Resolve whether a Monte-Carlo run uses the GPU device.

    Routes through the canonical resolver so 'gpu' with no GPU, 'gpu_fp64',
    and unknown strings raise the standard library errors.
    """
    return resolve_backend(backend, supports_fp64=False).is_gpu


def _boot_gpu_vectorizable(design: BootstrapDesign) -> bool:
    """True if this bootstrap design can run on the GPU mean kernel.

    The GPU path implements ONE closed-form statistic — the sample mean of a
    1-D array under ordinary index resampling. Everything else is CPU-only.
    The statistic form is taken from the caller's explicit ``gpu_statistic``
    declaration; it is NEVER inferred from the statistic's output.
    """
    return (
        design.gpu_statistic == "mean"
        and design.method == "ordinary"
        and design.statistic_type == "i"
        and design.strata is None
        and design.data.ndim == 1
    )


def _select_boot_backend(backend: BackendChoice | None,
                         design: BootstrapDesign):
    """Choose the bootstrap backend, honouring fail-loud fidelity (Guarantee 2).

    - CPU device → CPU backend (runs the user's real statistic).
    - GPU device + a vectorizable declared-mean design → GPU backend.
    - Explicit ``backend='gpu'`` that cannot be honoured on the GPU → RAISE
      (never silently fall back to a different backend, and never silently
      compute a different statistic).
    - ``backend='auto'`` that cannot use the GPU → CPU backend (auto expressed
      no preference; the choice is disclosed via ``backend_name``).
    """
    if not _use_gpu(backend):
        return CPUBootstrapBackend()

    if _boot_gpu_vectorizable(design):
        from pystatistics.montecarlo.backends.gpu import GPUBootstrapBackend
        return GPUBootstrapBackend()

    # GPU device requested but the design cannot run on the GPU kernel.
    if backend == "gpu":
        if design.gpu_statistic != "mean":
            raise ValidationError(
                "backend='gpu' requires gpu_statistic='mean'. The GPU bootstrap "
                "path vectorizes only the sample mean; an arbitrary Python "
                "statistic cannot execute on the GPU. Pass gpu_statistic='mean' "
                "if your statistic is the mean, or use backend='cpu'."
            )
        raise ValidationError(
            "backend='gpu' with gpu_statistic='mean' supports only "
            "method='ordinary', statistic_type='index', strata=None, and 1-D "
            "data. This configuration cannot run on the GPU; use backend='cpu'."
        )
    # backend='auto' — disclosed CPU fallback.
    return CPUBootstrapBackend()


def _select_perm_backend(backend: BackendChoice | None,
                         design: PermutationDesign):
    """Choose the permutation backend, honouring fail-loud fidelity.

    Same policy as :func:`_select_boot_backend`: the GPU path vectorizes only
    the mean-difference statistic on 1-D groups; an explicit ``backend='gpu'``
    that cannot be honoured raises rather than silently substituting.
    """
    if not _use_gpu(backend):
        return CPUPermutationBackend()

    vectorizable = (
        design.gpu_statistic == "mean_diff"
        and design.x.ndim == 1
        and design.y.ndim == 1
    )
    if vectorizable:
        from pystatistics.montecarlo.backends.gpu import GPUPermutationBackend
        return GPUPermutationBackend()

    if backend == "gpu":
        if design.gpu_statistic != "mean_diff":
            raise ValidationError(
                "backend='gpu' requires gpu_statistic='mean_diff'. The GPU "
                "permutation path vectorizes only the difference in means; an "
                "arbitrary Python statistic cannot execute on the GPU. Pass "
                "gpu_statistic='mean_diff' if your statistic is mean(x)-mean(y), "
                "or use backend='cpu'."
            )
        raise ValidationError(
            "backend='gpu' with gpu_statistic='mean_diff' supports only 1-D "
            "groups x and y. This configuration cannot run on the GPU; use "
            "backend='cpu'."
        )
    return CPUPermutationBackend()


def boot(
    data: ArrayLike,
    statistic: Callable,
    n_resamples: int = 999,
    *,
    method: Literal["ordinary", "parametric", "balanced"] = "ordinary",
    statistic_type: Literal["index", "frequency", "weight"] = "index",
    strata: ArrayLike | None = None,
    ran_gen: Callable | None = None,
    mle: Any = None,
    seed: int | None = None,
    backend: BackendChoice | None = None,
    gpu_statistic: Literal["mean"] | None = None,
) -> BootstrapSolution:
    """
    Bootstrap resampling. Matches R's boot::boot().

    The statistic function signature depends on method:
    - For nonparametric (method="ordinary" or "balanced"):
        statistic(data, indices) -> array of shape (k,)
        where indices are bootstrap sample indices (statistic_type="index"),
        frequency counts ("frequency"), or weights ("weight").
    - For parametric (method="parametric"):
        statistic(simulated_data) -> array of shape (k,)
        where simulated_data is generated by ran_gen(data, mle, rng).

    Args:
        data: Original data, shape (n,) or (n, p).
        statistic: Function to compute the statistic(s) of interest.
        n_resamples: Number of bootstrap replicates. Default 999.
        method: Bootstrap variant: "ordinary", "balanced", or "parametric".
        statistic_type: Type of second argument to statistic: "index",
            "frequency", or "weight".
        strata: Stratification vector (resampling within strata).
        ran_gen: For parametric bootstrap: fn(data, mle, rng) -> sim_data.
        mle: Parameter estimates for parametric bootstrap.
        seed: Random seed for reproducibility.
        backend: Compute backend. Default None → 'cpu'. Explicit:
            'cpu', 'gpu', or 'auto'.
        gpu_statistic: Explicit declaration that ``statistic`` computes the
            sample mean, enabling the vectorized GPU kernel. Only ``"mean"`` is
            supported. Required when ``backend='gpu'`` (the GPU path never infers
            the statistic form); ``backend='gpu'`` without it raises. Ignored on
            the CPU path. When declared, it is verified against the observed
            statistic on the full sample (fail-loud) before the GPU is used.

    Returns:
        BootstrapSolution with t0, t, bias, SE.

    Examples:
        >>> import numpy as np
        >>> from pystatistics.montecarlo import boot
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> def mean_stat(data, indices):
        ...     return np.array([np.mean(data[indices])])
        >>> result = boot(data, mean_stat, n_resamples=999, seed=42)
        >>> result.t0  # observed mean
        >>> result.bias  # bootstrap bias estimate
        >>> result.standard_errors  # bootstrap standard error
    """
    if backend is None:
        backend = 'cpu'

    if statistic_type not in _STATISTIC_TYPE_CODE:
        raise ValidationError(
            f"statistic_type must be one of {list(_STATISTIC_TYPE_CODE)}, "
            f"got {statistic_type!r}"
        )

    design = BootstrapDesign.for_bootstrap(
        data=data,
        statistic=statistic,
        n_resamples=n_resamples,
        method=method,
        statistic_type=_STATISTIC_TYPE_CODE[statistic_type],
        strata=strata,
        ran_gen=ran_gen,
        mle=mle,
        seed=seed,
        gpu_statistic=gpu_statistic,
    )

    be = _select_boot_backend(backend, design)
    result = be.solve(design)

    return BootstrapSolution(_result=result, _design=design)


def boot_ci(
    boot_out: BootstrapSolution,
    *,
    conf_level: float | Sequence[float] = 0.95,
    ci_type: str | Sequence[str] = "all",
    index: int = 0,
    var_t0: float | None = None,
    var_t: NDArray | None = None,
) -> BootstrapSolution:
    """
    Compute bootstrap confidence intervals. Matches R's boot::boot.ci().

    Takes a BootstrapSolution from boot() and computes confidence intervals
    using one or more methods.

    Args:
        boot_out: Result from boot().
        conf_level: Confidence level(s). Default 0.95. Multi-level sequences
            (length > 1) are not yet supported and raise ValidationError.
        ci_type: CI type(s): "normal", "basic", "percentile", "bca",
            "studentized", or "all". "all" computes normal, basic, percentile,
            and BCa (not studentized unless var_t is provided).
        index: Which statistic to compute CI for (0-indexed into t0).
        var_t0: Variance of the observed statistic (for normal/studentized).
        var_t: Per-replicate variance estimates, shape (n_resamples,). Required
            for studentized CI.

    Returns:
        New BootstrapSolution with CI populated.

    Examples:
        >>> result = boot(data, mean_stat, n_resamples=999, seed=42)
        >>> ci_result = boot_ci(result, ci_type="percentile")
        >>> ci_result.conf_int["percentile"]  # shape (k, 2) for [lower, upper]
    """
    from pystatistics.montecarlo._ci import compute_ci

    # Normalize conf_level to a single float. Multi-level CI is not yet
    # supported: fail loud rather than silently truncating to the first level.
    if isinstance(conf_level, (list, tuple)):
        if len(conf_level) > 1:
            raise ValidationError(
                "Multi-level conf_level is not yet supported: got "
                f"{list(conf_level)!r} ({len(conf_level)} levels). Pass a single "
                "confidence level (a scalar or a length-1 sequence)."
            )
        cl = float(conf_level[0])
    else:
        cl = float(conf_level)

    # Normalize ci_type
    if isinstance(ci_type, str):
        if ci_type == "all":
            types = ["normal", "basic", "percentile", "bca"]
            if var_t is not None:
                types.append("studentized")
        else:
            types = [ci_type]
    else:
        types = list(ci_type)

    ci_dict = compute_ci(
        boot_out=boot_out,
        types=types,
        conf_level=cl,
        index=index,
        var_t0=var_t0,
        var_t=var_t,
    )

    # Create new BootParams with CI
    old_params = boot_out._result.params
    new_params = BootParams(
        t0=old_params.t0,
        t=old_params.t,
        n_resamples=old_params.n_resamples,
        bias=old_params.bias,
        standard_errors=old_params.standard_errors,
        conf_int=ci_dict,
        conf_level=cl,
    )

    from pystatistics.core.result import Result
    new_result = Result(
        params=new_params,
        info=boot_out._result.info,
        timing=boot_out._result.timing,
        backend_name=boot_out._result.backend_name,
        warnings=boot_out._result.warnings,
    )

    return BootstrapSolution(_result=new_result, _design=boot_out._design)


def permutation_test(
    x: ArrayLike,
    y: ArrayLike,
    statistic: Callable,
    n_resamples: int = 9999,
    *,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    seed: int | None = None,
    backend: BackendChoice | None = None,
    gpu_statistic: Literal["mean_diff"] | None = None,
) -> PermutationSolution:
    """
    Permutation test for two groups.

    Shuffles the combined data R times, computing the test statistic on
    each permutation. P-value uses the Phipson-Smyth correction:
    (count + 1) / (R + 1).

    Args:
        x: Group 1 data.
        y: Group 2 data.
        statistic: fn(x, y) -> float. The test statistic.
        n_resamples: Number of permutations. Default 9999.
        alternative: "two-sided", "less", or "greater".
        seed: Random seed for reproducibility.
        backend: Compute backend. Default None → 'cpu'. Explicit:
            'cpu', 'gpu', or 'auto'.
        gpu_statistic: Explicit declaration that ``statistic`` computes the
            difference in means, mean(x) - mean(y), enabling the vectorized GPU
            kernel. Only ``"mean_diff"`` is supported. Required when
            ``backend='gpu'`` (the GPU path never infers the statistic form);
            ``backend='gpu'`` without it raises. Ignored on the CPU path. When
            declared, it is verified against the observed statistic (fail-loud)
            before the GPU is used.

    Returns:
        PermutationSolution with observed_stat, perm_stats, p_value.

    Examples:
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([6, 7, 8, 9, 10])
        >>> def mean_diff(x, y): return np.mean(x) - np.mean(y)
        >>> result = permutation_test(x, y, mean_diff, n_resamples=9999, seed=42)
        >>> result.p_value
    """
    if backend is None:
        backend = 'cpu'

    design = PermutationDesign.for_permutation_test(
        x=x,
        y=y,
        statistic=statistic,
        n_resamples=n_resamples,
        alternative=alternative,
        seed=seed,
        gpu_statistic=gpu_statistic,
    )

    be = _select_perm_backend(backend, design)
    result = be.solve(design)

    return PermutationSolution(_result=result, _design=design)
