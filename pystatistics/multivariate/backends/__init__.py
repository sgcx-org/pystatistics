"""Multivariate backends (CPU reference, GPU performance).

The CPU path is the reference implementation living in ``_pca.py`` and
``_factor.py``; it is validated against R to near machine precision.
GPU backends live in ``gpu_pca.py`` / ``gpu_factor.py`` and are
validated against the CPU path, not directly against R — the project
accepts FP32/FP64 divergence as documented in the README's "Design
Philosophy" section.

GPU backends are imported on-demand so that pystatistics can be used
without PyTorch installed (same convention as the regression backends).
"""

__all__: list[str] = []
