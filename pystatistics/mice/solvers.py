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
from pystatistics.mice._rng import spawn_streams
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
        Compute backend. Default None -> 'cpu'. GPU is not implemented in this
        release; requesting it raises.
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
    backend_impl = _get_backend(backend)
    streams = spawn_streams(seed, m)

    if verbose:
        print(
            f"MICE: {design.n} obs, {design.p} vars, "
            f"{design.missing_rate:.1%} missing, m={m}, maxit={maxit}, "
            f"backend={backend_impl.name}"
        )

    result = backend_impl.run(
        design, m=m, maxit=maxit, visit_sequence=visit, streams=streams
    )

    if verbose:
        print(f"Done in {result.timing.get('total_seconds', 0):.4f}s")

    return MICESolution(_result=result, _design=design)


def _get_backend(choice: BackendChoice | None):
    """Resolve the compute backend. None -> CPU; GPU not yet implemented."""
    if choice is None or choice == "cpu":
        from pystatistics.mice.backends.cpu import CPUMiceBackend

        return CPUMiceBackend()

    if choice == "auto":
        # 'auto' currently means CPU: no GPU backend ships in this release.
        # Kept as an accepted value so callers and the Stage-2 GPU path share
        # one vocabulary.
        from pystatistics.mice.backends.cpu import CPUMiceBackend

        return CPUMiceBackend()

    if choice == "gpu":
        # Fail loud rather than silently downgrading (Rule 1). select_device
        # gives a precise hardware error if no GPU exists; otherwise we still
        # refuse because the GPU chain backend is Stage 2.
        select_device("gpu")
        raise NotImplementedError(
            "backend='gpu' is not available in this release. MICE ships a "
            "validated CPU implementation first; GPU acceleration is planned. "
            "Use backend='cpu' (the default)."
        )

    raise ValueError(f"Unknown backend: {choice!r}")
