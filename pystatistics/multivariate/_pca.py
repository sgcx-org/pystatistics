"""
Principal Component Analysis via SVD.

Matches R's ``stats::prcomp()``, validated against R output.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.validation import check_2d, check_finite, check_array
from pystatistics.multivariate._common import PCAResult


def _fix_sign_convention(rotation: NDArray) -> NDArray:
    """Enforce R's sign convention for loadings.

    For each column (component) of the rotation matrix, ensure that the
    element with the largest absolute value is positive. This removes the
    sign ambiguity inherent in SVD and matches R's ``prcomp()`` convention.

    Args:
        rotation: Loadings matrix (p x k).

    Returns:
        Sign-corrected loadings matrix of same shape.
    """
    signs = np.sign(rotation[np.argmax(np.abs(rotation), axis=0), np.arange(rotation.shape[1])])
    # Replace any zeros with 1 (shouldn't happen with real data)
    signs = np.where(signs == 0, 1.0, signs)
    return rotation * signs[np.newaxis, :]


def pca(
    X: ArrayLike,
    *,
    center: bool = True,
    scale: bool = False,
    n_components: int | None = None,
    names: list[str] | None = None,
    backend: str | None = None,
    use_fp64: bool = False,
    method: str = "svd",
    force: bool = False,
    device_resident: bool = False,
) -> PCAResult:
    """Principal Component Analysis via SVD.

    Matches R's ``stats::prcomp()``.

    Algorithm:
        1. Center *X* (subtract column means).
        2. Optionally scale (divide by column standard deviations).
        3. Thin SVD: X_centered = U @ diag(S) @ V'.
        4. sdev = S / sqrt(n - 1).
        5. rotation = V (right singular vectors = loadings).
        6. x = U @ diag(S) (scores = X_centered @ V).

    Args:
        X: Data matrix (n x p), n observations, p variables.
        center: Whether to center columns (subtract mean). Default True.
        scale: Whether to scale columns (divide by SD). Default False.
            Equivalent to R's ``prcomp(scale. = TRUE)``.
        n_components: Number of components to retain. Default: min(n, p).
        names: Variable names for the p columns.
        backend: Compute backend. The default depends on the input
            type:
                - numpy array / CPU DataSource → ``'cpu'`` (R-reference
                  path; matches R ``prcomp()`` to rtol = 1e-10). This is
                  the regulated-industry default: unspecified means the
                  validated-against-R path.
                - torch.Tensor on GPU (e.g. ``ds.to('cuda')['X']``) →
                  ``'gpu'``. Creating a GPU DataSource is already an
                  opt-in to the GPU path, so we don't demand the user
                  repeat the choice with an explicit ``backend='gpu'``.
            Explicit values:
                - ``'cpu'``: force the numpy reference path. Raises if
                  given a GPU tensor (no silent device migration —
                  Rule 1).
                - ``'gpu'``: force GPU; raises if no GPU is available.
                - ``'auto'``: prefer GPU when available, else CPU.
        use_fp64: Only relevant when actually running on GPU. Default
            False: the GPU path uses FP32 and results match the CPU
            path at the ``GPU_FP32`` tolerance tier (rtol ≈ 1e-4) —
            statistically equivalent, not bitwise-equivalent. Set True
            on CUDA (MPS lacks FP64) to get CPU-matching precision at
            the cost of performance; note consumer NVIDIA parts have
            FP64 throughput ~1/64× FP32, so FP64-on-GPU is usually
            slower than CPU LAPACK.
        method: Algorithm to use. Only matters when actually running
            on GPU (the CPU path always uses SVD).
            ``'svd'`` (default): SVD of X. Always safe. Moderate
                (~3–4×) speedup over CPU LAPACK.
            ``'gram'``: Eigendecomposition of X'X — turns PCA into a
                big GEMM + small symmetric eigendecomp, both GPU
                sweet spots. For tall-skinny well-conditioned data
                (n ≫ p) this is typically 30–100×+ faster than the
                SVD path. **Squares the condition number**, so raises
                ``NumericalError`` on ill-conditioned data unless
                ``force=True``. Precision gates: cond(X) ≲ 1e6 for
                FP64, cond(X) ≲ 1e3 for FP32.
            ``'auto'``: Uses ``'gram'`` when n > 2p AND the condition
                check passes; falls back to ``'svd'`` otherwise. This
                is the "best GPU path that is mathematically safe for
                this data" dispatch.
        force: Bypass the Gram path's condition-number gate. Numerical
            results will be unreliable on truly ill-conditioned inputs
            — use only when you understand the data is well-conditioned
            despite the automated estimator disagreeing.
        device_resident: When ``True`` and the GPU backend runs, the
            returned :class:`PCAResult` holds its numeric fields
            (``sdev``, ``rotation``, ``center``, ``scale``, ``x``) as
            ``torch.Tensor`` instances on the fit's device rather than
            materialising them as numpy arrays. This saves the ~150 ms
            D2H copy of the scores matrix on 1M × 100 FP32 data, which
            otherwise dominates any multi-step GPU pipeline that
            consumes PCA output. Call ``result.to_numpy()`` or
            ``result.to('cpu')`` to materialise a numpy-backed copy.
            Ignored on the CPU path (result is always numpy-backed
            there). Default ``False`` preserves 1.8.0 behavior.

    Returns:
        PCAResult with sdev, rotation (loadings), scores, etc.

    Raises:
        ValidationError: If inputs are invalid.

    Validates against: R ``stats::prcomp()``.
    """
    # ---- Detect a device-resident torch.Tensor ----
    # If the caller passed in a tensor (typically from
    # ``DataSource.from_arrays(...).to('cuda')['X']``), we short-circuit
    # the numpy validation / centering machinery and route the entire
    # pipeline through the GPU backend. This is the whole point of the
    # device-resident DataSource API: pay H2D transfer once in .to(),
    # reach the ~5 ms compute ceiling on every subsequent fit.
    import sys as _sys
    _is_torch_tensor = (
        "torch" in _sys.modules
        and isinstance(X, _sys.modules["torch"].Tensor)
    )

    if _is_torch_tensor:
        import torch
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValidationError(
                f"X: expected 2-D tensor, got {X.ndim}-D"
            )
        if not torch.isfinite(X).all():
            raise ValidationError("X contains non-finite values")

        # Default-backend inference: if the user passed a torch.Tensor
        # without specifying ``backend=`` at all, the unambiguous intent
        # is "use the device this tensor is already on". A GPU
        # DataSource is itself the opt-in; requiring the user to say
        # ``backend='gpu'`` redundantly after calling ``.to('cuda')``
        # would be friction for no safety benefit.
        #
        # But if they DID say ``backend='cpu'`` explicitly, we still
        # raise — that's a contradiction between the input and the
        # explicit ask, not an implicit choice on our part.
        if backend is None:
            backend = "gpu" if X.device.type != "cpu" else "cpu"
        if backend == "cpu":
            raise ValidationError(
                "backend='cpu' was specified but X is a torch.Tensor "
                f"on device {X.device}. Either pass a numpy array / "
                "CPU DataSource to the CPU backend, or call `.to('cpu')` "
                "on the DataSource explicitly to move it back."
            )

        n, p = X.shape
        X_for_gpu = X        # tensor flows straight through to pca_gpu
        X_arr = None         # sentinel: "we're on the GPU path"
    else:
        if backend is None:
            # CPU default for numpy input — the regulated-industry
            # contract: unspecified-backend means the R-reference path.
            backend = "cpu"
        X_arr = check_array(X, "X")
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        check_2d(X_arr, "X")
        check_finite(X_arr, "X")
        n, p = X_arr.shape
        X_for_gpu = None

    if n < 2:
        raise ValidationError("X: requires at least 2 observations, got 1")

    if names is not None:
        if len(names) != p:
            raise ValidationError(
                f"names: length {len(names)} does not match number of columns {p}"
            )
        var_names: tuple[str, ...] | None = tuple(names)
    else:
        var_names = None

    max_components = min(n, p)

    if n_components is not None:
        if n_components < 1:
            raise ValidationError(
                f"n_components: must be >= 1, got {n_components}"
            )
        if n_components > max_components:
            raise ValidationError(
                f"n_components: {n_components} exceeds max possible "
                f"min(n, p) = {max_components}"
            )
    else:
        n_components = max_components

    # ---- Centering / scaling ----
    # When X arrived as a GPU tensor we leave moments on-device for as
    # long as possible. Each ``.cpu()`` call is a synchronous D2H
    # transfer that serializes against any pending GPU work — 5 little
    # sync points in a tight sequence cost ~250 ms on a 1M × 100 matrix
    # even though the data being copied is 100 floats. Instead, we
    # compute means/SDs on GPU, pass the GPU tensors straight into
    # ``pca_gpu``, and let that backend do the single host transfer at
    # the end.
    if X_arr is None:
        import torch
        # Compute moments in the target fit dtype. Promoting a 1M × 100
        # FP32 matrix to FP64 just to compute a length-p mean vector
        # touches the entire 400 MB → 800 MB working set, which
        # dominates wall time. The moments are numerically fine at
        # whatever precision the fit itself uses.
        moment_dtype = torch.float64 if use_fp64 else torch.float32
        X_for_moments = (
            X_for_gpu if X_for_gpu.dtype == moment_dtype
            else X_for_gpu.to(moment_dtype)
        )
        if center:
            col_means_gpu = X_for_moments.mean(dim=0)
        else:
            col_means_gpu = torch.zeros(
                p, device=X_for_moments.device, dtype=moment_dtype,
            )
        if scale:
            Xc_t = X_for_moments - col_means_gpu if center else X_for_moments
            col_sds_gpu = Xc_t.std(dim=0, unbiased=True)
            # Zero-variance check is the one unavoidable D2H: it
            # decides whether to raise. ``any()`` keeps transfer to
            # a single scalar; full vector only pulled on the error
            # path.
            if bool((col_sds_gpu == 0).any().item()):
                col_sds_cpu = col_sds_gpu.detach().cpu().numpy()
                zero_cols = np.where(col_sds_cpu == 0)[0].tolist()
                raise ValidationError(
                    f"X: columns {zero_cols} have zero variance; "
                    f"cannot scale constant columns"
                )
        else:
            col_sds_gpu = None
        X_centered = None
        col_means = col_means_gpu       # GPU tensor; pca_gpu handles it
        scale_values: NDArray | None = col_sds_gpu  # GPU tensor or None
    else:
        if center:
            col_means = np.mean(X_arr, axis=0)
            X_centered = X_arr - col_means
        else:
            col_means = np.zeros(p)
            X_centered = X_arr.copy()

        if scale:
            col_sds = np.std(X_centered, axis=0, ddof=1)
            zero_sd = col_sds == 0
            if np.any(zero_sd):
                zero_cols = np.where(zero_sd)[0].tolist()
                raise ValidationError(
                    f"X: columns {zero_cols} have zero variance; "
                    f"cannot scale constant columns"
                )
            X_centered = X_centered / col_sds
            scale_values = col_sds
        else:
            scale_values = None

    # ---- Backend dispatch ----
    # 'auto' prefers GPU when available (matches the design philosophy in
    # the README) but falls back to CPU cleanly. 'cpu' forces the numpy
    # reference path below; 'gpu' raises if no GPU is available rather
    # than silently falling back (Rule 1).
    if backend not in ("auto", "cpu", "gpu"):
        raise ValidationError(
            f"backend: must be 'auto', 'cpu', or 'gpu', got {backend!r}"
        )
    # If X arrived as a GPU tensor, GPU is the only sensible route (we
    # already rejected backend='cpu' above). Bypass the device-
    # detection logic — we know the device from the tensor itself.
    if X_arr is None:
        from pystatistics.multivariate.backends.gpu_pca import pca_gpu
        return pca_gpu(
            X_for_gpu,
            center=center,
            scale=scale,
            n_components=n_components,
            col_means_cpu=col_means,
            scale_values_cpu=scale_values,
            var_names=var_names,
            device=str(X_for_gpu.device.type),
            use_fp64=use_fp64,
            method=method,
            force=force,
            device_resident=device_resident,
        )

    if backend != "cpu":
        from pystatistics.core.compute.device import select_device
        dev = select_device("gpu" if backend == "gpu" else "auto")
        if dev.is_gpu:
            from pystatistics.multivariate.backends.gpu_pca import pca_gpu
            return pca_gpu(
                X_arr,
                center=center,
                scale=scale,
                n_components=n_components,
                col_means_cpu=col_means,
                scale_values_cpu=scale_values,
                var_names=var_names,
                device=dev.device_type,
                use_fp64=use_fp64,
                method=method,
                force=force,
                device_resident=device_resident,
            )
        if backend == "gpu":
            raise RuntimeError(
                "backend='gpu' requested but no GPU is available. "
                "Install PyTorch with CUDA/MPS support or use backend='cpu'."
            )
        # backend == 'auto' and no GPU: fall through to CPU (uses SVD;
        # 'method' and 'force' only matter on GPU).

    # ---- SVD (CPU reference path) ----
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # sdev = singular values / sqrt(n - 1)
    sdev = S / np.sqrt(n - 1)

    # rotation = V (right singular vectors, transposed from Vt)
    rotation = Vt.T  # shape (p, min(n, p))

    # Apply sign convention
    rotation = _fix_sign_convention(rotation)

    # scores = X_centered @ V = U @ diag(S)
    # Recompute scores using the sign-fixed rotation for consistency
    scores = X_centered @ rotation

    # ---- Truncate to n_components ----
    sdev = sdev[:n_components]
    rotation = rotation[:, :n_components]
    scores = scores[:, :n_components]

    return PCAResult(
        sdev=sdev,
        rotation=rotation,
        center=col_means,
        scale=scale_values,
        x=scores,
        n_obs=n,
        n_vars=p,
        var_names=var_names,
    )
