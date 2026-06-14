"""
Solver dispatch for MICE.

Public API: ``mice(data, *, m, maxit, method, seed, ...) -> MICESolution``.

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

import warnings
from typing import Literal, Mapping, Sequence

from pystatistics.core.compute.device import select_device
from pystatistics.mice._visit import resolve_visit_sequence
from pystatistics.mice.design import MICEDesign
from pystatistics.mice.solution import MICESolution

BackendChoice = Literal["auto", "cpu", "gpu"]


def mice(
    data_or_design,
    *,
    seed: int,
    m: int = 5,
    maxit: int = 5,
    method: str = "pmm",
    methods: Mapping | Sequence[str] | None = None,
    visit_sequence: Sequence[int] | None = None,
    backend: BackendChoice | None = None,
    use_fp64: bool = False,
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
    m : int
        Number of imputations (completed datasets). Default 5 (R default).
    maxit : int
        Iterations per chain. Default 5 (R default).
    method : str
        Default per-column method for raw-data input. Default ``'pmm'``
        (R default for numeric columns). Use ``'norm'`` for Bayesian linear
        regression imputation.
    methods : mapping or sequence, optional
        Per-column method override for raw-data input (see ``MICEDesign``).
    visit_sequence : sequence of int, optional
        Column visit order within each iteration. Defaults to incomplete
        columns in ascending index order (R "roman" default).
    backend : {'cpu', 'gpu', 'auto'} or None
        Compute backend. Default None -> 'cpu' (the R-validated reference path).
        'gpu' runs the chains batched on a CUDA GPU; 'auto' uses CUDA when
        available, else CPU. GPU results match the CPU backend at the GPU/FP32
        tolerance tier.
    use_fp64 : bool
        Only relevant on the GPU. Run the GPU chains in double precision (CUDA
        only) for closer parity with the CPU reference. Default False (FP32,
        the fast consumer-GPU path). Ignored on CPU (always double precision).
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
    if not isinstance(m, int) or isinstance(m, bool) or m < 1:
        raise ValueError(f"m must be a positive integer, got {m!r}")
    if not isinstance(maxit, int) or isinstance(maxit, bool) or maxit < 1:
        raise ValueError(f"maxit must be a positive integer, got {maxit!r}")

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
        raise ValueError(
            "Data has no missing values; nothing to impute. "
            "MICE requires at least one NaN."
        )

    visit = resolve_visit_sequence(design.incomplete_columns, visit_sequence)
    backend_impl = _get_backend(backend, use_fp64)

    if verbose:
        print(
            f"MICE: {design.n} obs, {design.p} vars, "
            f"{design.missing_rate:.1%} missing, m={m}, maxit={maxit}, "
            f"backend={backend_impl.name}"
        )

    result = backend_impl.run(
        design, m=m, maxit=maxit, visit_sequence=visit, seed=seed
    )

    if verbose:
        print(f"Done in {result.timing.get('total_seconds', 0):.4f}s")

    return MICESolution(_result=result, _design=design)


def _get_backend(choice: BackendChoice | None, use_fp64: bool):
    """Resolve the compute backend.

    None/'cpu' -> CPU. 'gpu' -> CUDA GPU (raises on MPS, which is not yet
    validated for MICE). 'auto' -> CUDA when available, else CPU; like the
    other pystatistics modules, 'auto' never selects MPS.
    """
    if choice is None or choice == "cpu":
        from pystatistics.mice.backends.cpu import CPUMiceBackend

        return CPUMiceBackend()

    if choice == "auto":
        device = select_device("auto")
        if device.device_type == "cuda":
            from pystatistics.mice.backends.gpu import GPUMiceBackend

            return GPUMiceBackend(device="cuda", use_fp64=use_fp64)
        # CPU, or MPS (never auto-selected): use the CPU reference path.
        from pystatistics.mice.backends.cpu import CPUMiceBackend

        return CPUMiceBackend()

    if choice == "gpu":
        device = select_device("gpu")  # raises RuntimeError if no GPU
        if device.device_type == "mps":
            raise NotImplementedError(
                "backend='gpu' for MICE is validated on CUDA only in this "
                "release. The chained-equations sweep has not been validated "
                "on Apple Silicon (MPS), so it is refused rather than run "
                "unverified. Use backend='cpu' (the default) on a Mac; CUDA "
                "GPUs are supported."
            )
        from pystatistics.mice.backends.gpu import GPUMiceBackend

        return GPUMiceBackend(device=device.device_type, use_fp64=use_fp64)

    raise ValueError(f"Unknown backend: {choice!r}")
