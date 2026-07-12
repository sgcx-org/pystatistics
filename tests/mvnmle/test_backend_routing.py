"""
Routing/API tests for the direct MVN MLE CPU backends.

Covers the dispatch contract that ``mlest(method='direct')`` exposes:
- the default / ``backend='cpu'`` path is the fast PyTorch forward-Cholesky
  FP64 estimator,
- ``solver='reference'`` forces the numpy inverse-Cholesky reference
  (the R-exact validation anchor, and the only direct path needing no PyTorch),
- when PyTorch is unavailable the default path falls back to the reference
  *loudly* (Rule 1), while the explicit reference request stays silent,
- ``solver='reference'`` is rejected for non-direct algorithms.

These are dispatch tests; numerical R-agreement lives in test_mlest.py.
"""

import sys
import json
import numpy as np
import pytest
from pathlib import Path

from pystatistics.mvnmle import mlest, datasets

REFERENCES = Path(__file__).parent / "references"


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _info(result):
    return result._result.info


class TestDefaultPath:
    """The default and ``backend='cpu'`` route to the fast forward-Cholesky
    FP64 estimator (when PyTorch is installed)."""

    @pytest.mark.skipif(not _torch_available(), reason="needs PyTorch")
    def test_unspecified_backend_uses_fast_path(self):
        result = mlest(datasets.apple)  # no backend argument
        assert result.backend_name == 'cpu_cholesky_fp64'
        assert _info(result)['parameterization'] == 'cholesky'

    @pytest.mark.skipif(not _torch_available(), reason="needs PyTorch")
    def test_backend_cpu_uses_fast_path(self):
        result = mlest(datasets.apple, backend='cpu')
        assert result.backend_name == 'cpu_cholesky_fp64'
        assert _info(result)['parameterization'] == 'cholesky'


class TestReferencePath:
    """``solver='reference'`` selects the numpy inverse-Cholesky path."""

    def test_reference_solver_uses_numpy_inverse_cholesky(self):
        result = mlest(datasets.apple, solver='reference')
        assert 'cpu' in result.backend_name
        assert _info(result)['parameterization'] == 'inverse_cholesky'

    def test_reference_solver_needs_no_torch(self, monkeypatch):
        """The canonical reference path (solver='reference') must work with
        PyTorch absent and stay silent (it is an explicit, intentional choice —
        not a fallback)."""
        monkeypatch.setitem(sys.modules, 'torch', None)  # `import torch` -> ImportError
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # any warning fails the test
            result = mlest(datasets.apple, solver='reference')
        assert _info(result)['parameterization'] == 'inverse_cholesky'

    @pytest.mark.parametrize('algorithm', ['em', 'monotone'])
    def test_reference_solver_rejected_for_non_direct(self, algorithm):
        with pytest.raises(ValueError, match="only valid with method='direct'"):
            mlest(datasets.apple, solver='reference', method=algorithm)


class TestTorchFreeFallback:
    """With PyTorch absent, the default path falls back to the reference and
    says so out loud (Rule 1 — no silent fallback)."""

    def test_default_falls_back_loudly_without_torch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, 'torch', None)  # `import torch` -> ImportError
        with pytest.warns(UserWarning, match="PyTorch is not installed"):
            result = mlest(datasets.apple, backend='cpu')
        # Falls back to the correct, R-validated numpy reference.
        assert _info(result)['parameterization'] == 'inverse_cholesky'
        assert result.converged


class TestFastReferenceAgreement:
    """The fast path and the reference agree with R and with each other."""

    @pytest.mark.skipif(not _torch_available(), reason="needs PyTorch")
    def test_fast_matches_reference_on_apple(self):
        with open(REFERENCES / "apple_reference.json") as f:
            ref = json.load(f)
        fast = mlest(datasets.apple, backend='cpu')
        reference = mlest(datasets.apple, solver='reference')
        # Both match R...
        assert abs(fast.loglik - ref['loglik']) < 1e-7
        assert abs(reference.loglik - ref['loglik']) < 1e-7
        # ...and therefore each other.
        assert abs(fast.loglik - reference.loglik) < 1e-6
        np.testing.assert_allclose(fast.muhat, reference.muhat, atol=1e-3)
        np.testing.assert_allclose(fast.sigmahat, reference.sigmahat, rtol=1e-3)


class TestAutoWithoutCuda:
    """``backend='auto'`` with no CUDA uses the fast CPU path (when PyTorch is
    installed). Skipped on CUDA machines, where auto correctly prefers the GPU."""

    @pytest.mark.skipif(not _torch_available(), reason="needs PyTorch")
    def test_auto_without_cuda_uses_fast_cpu(self):
        import torch
        if torch.cuda.is_available():
            pytest.skip("auto prefers CUDA when present; covered in test_gpu.py")
        result = mlest(datasets.apple, method='direct', backend='auto')
        assert result.backend_name == 'cpu_cholesky_fp64'
