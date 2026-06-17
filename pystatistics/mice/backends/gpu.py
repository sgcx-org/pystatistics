"""
GPU backend for MICE.

Implements the same ``run`` entrypoint as the CPU backend, but runs all ``m``
imputation chains concurrently on a GPU (CUDA or Apple Silicon/MPS) with the
chain index as the leading tensor batch dimension. The batched ops are shared by
both devices, with a single device bridge in the PMM donor search (see
``_gpu_methods._insertion_rank``). Each sweep step — visit column ``j``, fit its method on
the observed rows, overwrite its missing cells — becomes a handful of batched
kernels across the ``m`` chains (batched linear solves + Cholesky, and for PMM a
batched nearest-neighbour donor search).

Why batch over chains: the ``m`` chains are statistically independent and share
the missingness pattern, so they have identical shapes at every step — an ideal
batch. The sweep stays sequential *within* a chain (column ``j`` depends on the
columns imputed before it), which is inherent to chained equations.

Validation tier: GPU results match the CPU reference distributionally at the
``GPU_FP32`` tolerance, not bit-for-bit (FP32 + a different RNG). FP64 is
available on CUDA (``use_fp64=True``) for closer parity; MPS is FP32-only
(no float64) so ``use_fp64`` is rejected there at the dispatch boundary.

Determinism: a single seeded ``torch.Generator`` on the device is the only
randomness source. Given the same seed and device, a run reproduces its own
output to within FP32 kernel non-determinism.
"""

from __future__ import annotations

from pystatistics.core.compute.timing import Timer
from pystatistics.core.compute.torch_interop import to_host_f64
from pystatistics.core.exceptions import ValidationError
from pystatistics.core.result import Result
from pystatistics.mice.backends._gpu_methods import GPU_METHODS
from pystatistics.mice.design import MICEDesign
from pystatistics.mice.solution import MICEParams

# R mice default donor count for PMM (the design does not carry a per-column
# donor count, so the GPU path uses the same default as the CPU PMMMethod).
_PMM_DONORS = 5


class GPUMiceBackend:
    """Runs MICE chains batched on a CUDA GPU."""

    def __init__(self, device: str = "cuda", use_fp64: bool = False):
        self.device = device
        self.use_fp64 = use_fp64

    @property
    def name(self) -> str:
        return f"gpu_{'fp64' if self.use_fp64 else 'fp32'}"

    def run(
        self,
        design: MICEDesign,
        *,
        m: int,
        maxit: int,
        visit_sequence: tuple[int, ...],
        seed: int,
    ) -> Result[MICEParams]:
        import torch

        # The GPU backend handles fully numeric problems only: categorical
        # columns need dummy-encoding / categorical model fits that this batched
        # path does not implement. Refuse rather than impute them wrong (a
        # categorical column can be a *predictor* even when the target is
        # numeric, so this checks every column, not just the targets).
        if design.has_categorical:
            raise ValidationError(
                "backend='gpu' supports numeric columns only. This data has "
                "categorical columns; use backend='cpu' (categorical imputation "
                "is CPU-only)."
            )

        # Validate that every incomplete column's method has a GPU kernel before
        # touching the device (fail loud at the boundary, Rule 2).
        for j in design.incomplete_columns:
            meth = design.method_for(j)
            if meth not in GPU_METHODS:
                raise ValidationError(
                    f"Method {meth!r} (column {j}) has no GPU implementation. "
                    f"GPU methods: {sorted(GPU_METHODS)}. Use backend='cpu'."
                )

        timer = Timer()
        timer.start()

        dtype = torch.float64 if self.use_fp64 else torch.float32
        dev = torch.device(self.device)
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))

        n, p = design.n, design.p
        incomplete = design.incomplete_columns
        incomplete_pos = {j: k for k, j in enumerate(incomplete)}

        with timer.section("transfer"):
            base = torch.as_tensor(design.data, device=dev, dtype=dtype)  # (n, p)
            mask = torch.as_tensor(design.missing_mask, device=dev)        # (n, p)
            # m independent copies of the data (chains diverge after init).
            data = base.unsqueeze(0).repeat(m, 1, 1)                       # (m, n, p)

        # Precompute per-incomplete-column index tensors (shared by all chains).
        obs_idx, mis_idx, pred_idx = {}, {}, {}
        for j in incomplete:
            mcol = mask[:, j]
            obs_idx[j] = torch.nonzero(~mcol, as_tuple=False).squeeze(1)
            mis_idx[j] = torch.nonzero(mcol, as_tuple=False).squeeze(1)
            pred_idx[j] = torch.tensor(
                [c for c in range(p) if c != j], device=dev, dtype=torch.long
            )

        with timer.section("init"):
            self._initialise(data, base, obs_idx, mis_idx, incomplete, m, gen, dev)

        chain_mean = torch.empty((m, maxit, len(incomplete)), dtype=dtype, device=dev)
        chain_var = torch.empty((m, maxit, len(incomplete)), dtype=dtype, device=dev)

        with timer.section("sweep"):
            for it in range(maxit):
                for j in visit_sequence:
                    method_fn = GPU_METHODS[design.method_for(j)]
                    X = data[:, :, pred_idx[j]]            # (m, n, q)
                    X_obs = X[:, obs_idx[j], :]            # (m, n_obs, q)
                    X_mis = X[:, mis_idx[j], :]            # (m, n_mis, q)
                    y_obs = data[:, obs_idx[j], j]         # (m, n_obs)

                    imputed = method_fn(
                        y_obs, X_obs, X_mis, gen, donors=_PMM_DONORS
                    )
                    data[:, mis_idx[j], j] = imputed

                for j in incomplete:
                    cells = data[:, mis_idx[j], j]          # (m, n_mis)
                    k = incomplete_pos[j]
                    chain_mean[:, it, k] = cells.mean(dim=1)
                    chain_var[:, it, k] = cells.var(dim=1, unbiased=False)

        with timer.section("download"):
            completed = to_host_f64(data)
            chain_mean_host = to_host_f64(chain_mean)
            chain_var_host = to_host_f64(chain_var)

        timer.stop()

        params = MICEParams(
            completed=completed,
            chain_mean=chain_mean_host,
            chain_var=chain_var_host,
            incomplete_columns=incomplete,
            m=m,
            maxit=maxit,
            visit_sequence=visit_sequence,
        )
        return Result(
            params=params,
            info={
                "method": "chained_equations",
                "m": m,
                "maxit": maxit,
                "device": self.device,
                "precision": "fp64" if self.use_fp64 else "fp32",
                "visit_sequence": list(visit_sequence),
            },
            timing=timer.result(),
            backend_name=self.name,
            warnings=(),
        )

    @staticmethod
    def _initialise(data, base, obs_idx, mis_idx, incomplete, m, gen, dev):
        """Fill missing cells by sampling observed values of the same column,
        independently per chain. Mutates ``data`` in place."""
        import torch

        for j in incomplete:
            observed = base[obs_idx[j], j]          # (n_obs,)
            n_obs = observed.shape[0]
            n_mis = mis_idx[j].shape[0]
            draw_idx = torch.randint(
                0, n_obs, (m, n_mis), generator=gen, device=dev
            )
            data[:, mis_idx[j], j] = observed[draw_idx]
