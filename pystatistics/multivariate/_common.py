"""
Common result types for multivariate analysis.

Frozen dataclasses following the PyStatistics pattern:
- PCAResult: principal component analysis results
- FactorResult: factor analysis results
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _is_torch_tensor(obj: object) -> bool:
    """True if ``obj`` is a ``torch.Tensor`` without force-importing torch.

    The sys.modules check keeps numpy-only callers from paying torch's
    ~800 ms cold import cost just to build or inspect a PCAResult.
    """
    import sys as _sys
    if "torch" not in _sys.modules:
        return False
    torch_mod = _sys.modules["torch"]
    return isinstance(obj, torch_mod.Tensor)


def _tensor_to_numpy(x):
    """Eager detach+cpu+numpy for a torch.Tensor; pass-through otherwise."""
    if _is_torch_tensor(x):
        return x.detach().cpu().numpy()
    return x


@dataclass(frozen=True)
class PCAResult:
    """Result from principal component analysis.

    Matches the structure of R's ``prcomp()`` return value.

    Device residency
    ----------------
    The numeric fields (``sdev``, ``rotation``, ``center``, ``scale``,
    ``x``) are annotated as ``NDArray | torch.Tensor``. When the GPU
    backend is invoked *with a device-resident input tensor* and
    ``device_resident=True`` is passed to :func:`pca`, the result's
    fields stay on the original device — the ``x`` scores matrix
    especially, which is typically the largest payload. Downstream
    computations can chain straight off ``result.x`` without paying
    the host↔device round-trip that otherwise dominates multi-step
    GPU pipelines. Call :meth:`to_numpy` (or :meth:`to` with a CPU
    device) to materialise a numpy-backed copy.

    Attributes:
        sdev: Standard deviations of principal components (length min(n, p)).
        rotation: Loadings matrix (p x n_components) -- columns are eigenvectors.
        center: Column means used for centering (length p).
        scale: Column SDs used for scaling (None if scale=False).
        x: Scores matrix (n x n_components).
        n_obs: Number of observations.
        n_vars: Number of variables.
        var_names: Variable names, or None.
    """

    sdev: NDArray
    rotation: NDArray
    center: NDArray
    scale: NDArray | None
    x: NDArray
    n_obs: int
    n_vars: int
    var_names: tuple[str, ...] | None

    @property
    def device(self) -> str:
        """Device hosting the numeric fields: ``'cpu'``, ``'cuda'``, or ``'mps'``.

        A numpy-backed result reports ``'cpu'``. A tensor-backed result
        reports the underlying tensor's device type.
        """
        if _is_torch_tensor(self.x):
            return str(self.x.device.type)
        return "cpu"

    def to_numpy(self) -> "PCAResult":
        """Return a new PCAResult with numpy-backed numeric fields.

        Idempotent on numpy-backed results (returns ``self``). On a
        tensor-backed result this is the explicit "I want CPU numpy"
        escape hatch, matching the rest of the library's
        no-silent-migration contract.
        """
        if not any(
            _is_torch_tensor(f)
            for f in (self.sdev, self.rotation, self.center, self.scale, self.x)
        ):
            return self
        return PCAResult(
            sdev=_tensor_to_numpy(self.sdev),
            rotation=_tensor_to_numpy(self.rotation),
            center=_tensor_to_numpy(self.center),
            scale=_tensor_to_numpy(self.scale),
            x=_tensor_to_numpy(self.x),
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            var_names=self.var_names,
        )

    def to(self, device: str) -> "PCAResult":
        """Move the result to ``device`` (``'cpu'``, ``'cuda'``, ``'mps'``).

        ``device='cpu'`` returns a numpy-backed PCAResult (same as
        :meth:`to_numpy`). Any other device requires torch and
        materialises the fields as ``torch.Tensor`` instances on that
        device.
        """
        if device == "cpu":
            return self.to_numpy()
        import torch
        tgt = torch.device(device)

        def _mv(v):
            if v is None:
                return None
            if _is_torch_tensor(v):
                return v.to(tgt)
            return torch.as_tensor(v, device=tgt)

        return PCAResult(
            sdev=_mv(self.sdev),
            rotation=_mv(self.rotation),
            center=_mv(self.center),
            scale=_mv(self.scale),
            x=_mv(self.x),
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            var_names=self.var_names,
        )

    @property
    def explained_variance_ratio(self):
        """Proportion of variance explained by each component.

        Returned as the same array type as ``sdev`` — numpy for a
        numpy-backed result, torch.Tensor on the same device for a
        tensor-backed result. Both numpy ndarrays and torch tensors
        support ``.sum()`` and elementwise arithmetic so the impl
        is type-polymorphic.
        """
        variances = self.sdev ** 2
        return variances / variances.sum()

    @property
    def cumulative_variance_ratio(self):
        """Cumulative proportion of variance explained."""
        return self.explained_variance_ratio.cumsum(0)

    def summary(self) -> str:
        """R-style summary matching ``summary(prcomp(...))``.

        Returns:
            Formatted string with standard deviations, proportion of variance,
            and cumulative proportion for each component.
        """
        # All quantities in the summary table are length-min(n, p), so
        # a one-shot D2H copy for tensor-backed results is cheap — no
        # reason to duplicate the formatting with a torch branch.
        r = self.to_numpy()
        n_comp = len(r.sdev)
        labels = [f"PC{i+1}" for i in range(n_comp)]

        header = "Importance of components:"
        col_width = max(12, max(len(lab) for lab in labels) + 2)

        def _fmt_row(name: str, values: NDArray) -> str:
            parts = [f"{name:<30s}"]
            for v in values:
                parts.append(f"{v:>{col_width}.6f}")
            return "".join(parts)

        rows = [
            header,
            _fmt_row("Standard deviation", r.sdev),
            _fmt_row("Proportion of Variance", r.explained_variance_ratio),
            _fmt_row("Cumulative Proportion", r.cumulative_variance_ratio),
        ]
        # Column labels
        label_row = " " * 30 + "".join(f"{lab:>{col_width}s}" for lab in labels)
        rows.insert(1, label_row)
        return "\n".join(rows)


@dataclass(frozen=True)
class FactorResult:
    """Result from factor analysis.

    Matches the structure of R's ``factanal()`` return value.

    Attributes:
        loadings: Rotated loadings matrix (p x n_factors).
        uniquenesses: Uniqueness for each variable (length p).
        communalities: 1 - uniquenesses.
        rotation_matrix: Rotation matrix, or None if no rotation.
        chi_sq: Goodness-of-fit chi-squared statistic, or None.
        p_value: p-value for chi-sq test, or None.
        dof: Degrees of freedom.
        n_factors: Number of factors extracted.
        n_obs: Number of observations.
        n_vars: Number of variables.
        var_names: Variable names, or None.
        method: Estimation method (e.g. 'ml').
        rotation_method: Rotation method (e.g. 'varimax', 'promax', 'none').
        converged: Whether the optimisation converged.
        n_iter: Number of iterations used.
        objective: Final objective function value.
    """

    loadings: NDArray
    uniquenesses: NDArray
    communalities: NDArray
    rotation_matrix: NDArray | None
    chi_sq: float | None
    p_value: float | None
    dof: int
    n_factors: int
    n_obs: int
    n_vars: int
    var_names: tuple[str, ...] | None
    method: str
    rotation_method: str
    converged: bool
    n_iter: int
    objective: float

    def summary(self) -> str:
        """R-style summary matching ``print(factanal(...))``.

        Returns:
            Formatted string with loadings, uniquenesses, SS loadings,
            proportion and cumulative variance, and the chi-squared test.
        """
        p, m = self.loadings.shape
        names = list(self.var_names) if self.var_names else [f"V{i+1}" for i in range(p)]
        factor_labels = [f"Factor{j+1}" for j in range(m)]
        name_width = max(len(n) for n in names) + 2
        col_width = 10

        lines: list[str] = []
        lines.append(f"Factor analysis with {m} factor(s), method: {self.method}")
        lines.append("")
        lines.append("Loadings:")

        # Header
        header = " " * name_width + "".join(f"{fl:>{col_width}s}" for fl in factor_labels)
        lines.append(header)

        # Loadings rows (suppress small values like R does)
        for i, name in enumerate(names):
            row = f"{name:<{name_width}s}"
            for j in range(m):
                val = self.loadings[i, j]
                if abs(val) < 0.1:
                    row += " " * col_width
                else:
                    row += f"{val:>{col_width}.3f}"
            row += f"{self.uniquenesses[i]:>{col_width}.3f}"
            lines.append(row)

        # Uniquenesses header in last column
        lines[3] = lines[3] + f"{'Uniquenesses':>{col_width + 4}s}"

        # SS loadings
        ss_loadings = np.sum(self.loadings ** 2, axis=0)
        prop_var = ss_loadings / p
        cum_var = np.cumsum(prop_var)

        lines.append("")
        row_ss = " " * name_width + "".join(f"{v:>{col_width}.3f}" for v in ss_loadings)
        row_pv = " " * name_width + "".join(f"{v:>{col_width}.3f}" for v in prop_var)
        row_cv = " " * name_width + "".join(f"{v:>{col_width}.3f}" for v in cum_var)
        lines.append(f"{'SS loadings':<{name_width}s}" + row_ss[name_width:])
        lines.append(f"{'Proportion Var':<{name_width}s}" + row_pv[name_width:])
        lines.append(f"{'Cumulative Var':<{name_width}s}" + row_cv[name_width:])

        # Chi-squared test
        if self.chi_sq is not None:
            lines.append("")
            lines.append(
                f"Test of the hypothesis that {m} factor(s) are sufficient."
            )
            lines.append(
                f"The chi square statistic is {self.chi_sq:.2f} "
                f"on {self.dof} degree(s) of freedom."
            )
            if self.p_value is not None:
                lines.append(f"The p-value is {self.p_value:.4g}")

        return "\n".join(lines)
