"""Multinomial backends (CPU reference, GPU performance).

CPU reference lives in ``_likelihood.py`` + ``_solver.py``, validated
against R ``nnet::multinom``. GPU backend in ``gpu_likelihood.py``
wraps the same softmax cross-entropy in PyTorch and uses autograd for
the gradient. Scipy's L-BFGS-B still drives the outer optimization;
GPU accelerates the per-iteration nll+grad evaluation.

Imported on demand (same convention as regression / multivariate
backends) so pystatistics works without PyTorch installed.
"""

__all__: list[str] = []
