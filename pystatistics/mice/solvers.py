"""
Solver dispatch for MICE.

Public API: ``mice(data, *, n_imputations, max_iter, method, seed, ...) -> MICESolution``.

Mirrors the pystatistics dispatch contract (cf. ``mvnmle.mlest``): resolve the
backend (``None`` -> ``'cpu'``; GPU is never implicit), build the design if the
caller passed raw data, spawn one reproducible RNG stream per imputation, hand
off to the backend's single ``run`` entrypoint, and wrap the result.

Reproducibility is mandatory: ``seed`` is required, not optional. Multiple
imputation is stochastic, and an unseeded run cannot be reproduced or validated
(CLAUDE.md Rule 6). Requiring the seed makes the reproducible path the only
path.
"""

from __future__ import annotations

from pystatistics.core.exceptions import ValidationError

import warnings
from typing import Literal, Mapping, Sequence

from pystatistics.core.compute.backend import resolve_backend
from pystatistics.mice._visit import resolve_visit_sequence
from pystatistics.mice.design import MICEDesign
from pystatistics.mice.solution import MICESolution

BackendChoice = Literal["auto", "cpu", "gpu", "gpu_fp64"]


def mice(
    data_or_design,
    *,
    seed: int,
    n_imputations: int = 5,
    max_iter: int = 5,
    method: str = "pmm",
    methods: Mapping | Sequence[str] | None = None,
    visit_sequence: Sequence[int] | None = None,
    backend: BackendChoice | None = None,
    verbose: bool = False,
) -> MICESolution:
    """Multiple imputation by chained equations (numeric columns, CPU).

    Parameters
    ----------
    data_or_design : array-like or MICEDesign
        Data matrix with NaN for missing values, or a prebuilt ``MICEDesign``.
        When a design is passed, ``method``/``methods`` are ignored (the design
        already fixes per-column methods).
    seed : int
        Required RNG seed. Guarantees bit-reproducible imputations.
    n_imputations : int
        Number of imputations (completed datasets). Default 5 (R's ``m``).
    max_iter : int
        Iterations per chain. Default 5 (R's ``maxit``).
    method : str
        Default per-column method for raw-data input. Default ``'pmm'``
        (R default for numeric columns). Use ``'norm'`` for Bayesian linear
        regression imputation.
    methods : mapping or sequence, optional
        Per-column method override for raw-data input (see ``MICEDesign``).
    visit_sequence : sequence of int, optional
        Column visit order within each iteration. Defaults to incomplete
        columns in ascending index order (R "roman" default).
    backend : {'cpu', 'gpu', 'gpu_fp64', 'auto'} or None
        Compute backend = (device, precision). Default None -> 'cpu' (the
        R-validated reference path, float64). 'gpu' runs the chains batched in
        float32 on a CUDA or Apple Silicon (MPS) GPU, whichever is present,
        matching the CPU backend at the GPU/FP32 tolerance tier. 'gpu_fp64' runs
        them in float64 on CUDA only (raises on MPS, which has no float64), for
        closer parity with the CPU reference. 'auto' uses CUDA-float32 when
        available, else CPU (never MPS — opt into MPS explicitly with 'gpu').
    verbose : bool
        Print progress information.

    Returns
    -------
    MICESolution

    Examples
    --------
    >>> from pystatistics.mice import mice
    >>> imp = mice(data, m=5, method='pmm', seed=0)
    >>> datasets = imp.completed_datasets()
    """
    if (not isinstance(n_imputations, int) or isinstance(n_imputations, bool)
            or n_imputations < 1):
        raise ValidationError(
            f"n_imputations must be a positive integer, got {n_imputations!r}")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValidationError(f"max_iter must be a positive integer, got {max_iter!r}")

    if isinstance(data_or_design, MICEDesign):
        design = data_or_design
        if methods is not None or method != "pmm":
            warnings.warn(
                "method/methods are ignored when a MICEDesign is passed; the "
                "design already fixes per-column methods.",
                UserWarning,
                stacklevel=2,
            )
    else:
        design = MICEDesign.from_array(
            data_or_design, method=method, methods=methods
        )

    if not design.has_missing:
        raise ValidationError(
            "Data has no missing values; nothing to impute. "
            "MICE requires at least one NaN."
        )

    visit = resolve_visit_sequence(design.incomplete_columns, visit_sequence)
    backend_impl = _get_backend(backend)

    if verbose:
        print(
            f"MICE: {design.n} obs, {design.p} vars, "
            f"{design.missing_rate:.1%} missing, n_imputations={n_imputations}, "
            f"max_iter={max_iter}, backend={backend_impl.name}"
        )

    # Public names map to the backends' historical internal kwargs (m / maxit).
    result = backend_impl.run(
        design, m=n_imputations, maxit=max_iter, visit_sequence=visit, seed=seed
    )

    if verbose:
        print(f"Done in {result.timing.get('total_seconds', 0):.4f}s")

    return MICESolution(_result=result, _design=design)


def _get_backend(choice: BackendChoice | None):
    """Resolve the compute backend from the canonical (device, precision) target.

    None/'cpu' -> CPU (float64). 'gpu' -> CUDA or Apple Silicon (MPS), float32.
    'gpu_fp64' -> CUDA only, float64 (raises on MPS). 'auto' -> CUDA when
    available, else CPU; 'auto' never selects MPS (opt in with 'gpu').
    """
    target = resolve_backend(choice, supports_fp64=True)
    if target.device_type == "cpu":
        from pystatistics.mice.backends.cpu import CPUMiceBackend

        return CPUMiceBackend()

    from pystatistics.mice.backends.gpu import GPUMiceBackend

    return GPUMiceBackend(device=target.device_type, use_fp64=target.use_fp64)
