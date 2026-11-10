"""
Routing/API tests for the direct MVN MLE CPU backends.

Covers the dispatch contract that ``mlest(method='direct')`` exposes:
- the default / ``backend='cpu'`` path is the fast PyTorch forward-Cholesky
  FP64 estimator,
- when PyTorch is unavailable the default path runs the NumPy forward-Cholesky
  backend *loudly* (Rule 1),
- the removed ``solver='reference'`` option raises a ``ValidationError``.

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


class TestRemovedReferenceSolver:
    """``solver='reference'`` (the inverse-Cholesky reference optimizer) was
    removed; asking for it fails loud with the migration path."""

    @pytest.mark.parametrize('kwargs', [
        {},
        {'backend': 'cpu'},
        {'backend': 'gpu'},
        {'method': 'em'},
        {'method': 'monotone'},
    ])
    def test_reference_solver_raises(self, kwargs):
        from pystatistics.core.exceptions import ValidationError
        with pytest.raises(ValidationError, match="has been removed"):
            mlest(datasets.apple, solver='reference', **kwargs)

    def test_message_names_the_replacement(self):
        from pystatistics.core.exceptions import ValidationError
        with pytest.raises(ValidationError, match="forward-Cholesky"):
            mlest(datasets.apple, solver='reference')

    @pytest.mark.parametrize('module', [
        'pystatistics.mvnmle.backends.cpu',
        'pystatistics.mvnmle._objectives.cpu',
    ])
    def test_removed_modules_are_gone(self, module):
        import importlib
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)

    def test_other_solver_values_are_scipy_methods(self):
        result = mlest(datasets.apple, solver='L-BFGS-B')
        assert _info(result)['method'] == 'L-BFGS-B'


class TestTorchFreeFallback:
    """With PyTorch absent, the default path runs the NumPy implementation of
    the forward-Cholesky estimator and says so out loud (Rule 1 — no silent
    fallback). Numerical coverage lives in test_numpy_cholesky.py."""

    def test_default_falls_back_loudly_without_torch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, 'torch', None)  # `import torch` -> ImportError
        with pytest.warns(UserWarning, match="PyTorch is not installed"):
            result = mlest(datasets.apple, backend='cpu')
        # Same estimator as the PyTorch path.
        assert result.backend_name == 'cpu_cholesky_numpy_fp64'
        assert _info(result)['parameterization'] == 'cholesky'
        assert result.converged


class TestAgreementWithR:
    """The default path agrees with the stored R result, whichever
    implementation (PyTorch or NumPy) runs it."""

    def test_default_matches_r_on_apple(self):
        import warnings
        with open(REFERENCES / "apple_reference.json") as f:
            ref = json.load(f)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)  # no-torch notice
            fit = mlest(datasets.apple, backend='cpu')
        assert abs(fit.loglik - ref['loglik']) < 1e-7
        np.testing.assert_allclose(fit.muhat, ref['muhat'], atol=1e-3)
        np.testing.assert_allclose(fit.sigmahat, ref['sigmahat'], rtol=1e-3)


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


class TestConflictingRequestsFailLoud:
    """Explicit-but-contradictory argument combinations must raise, never
    silently override the user's choice (Rule 1)."""

    def test_monotone_with_gpu_backend_raises(self):
        from pystatistics.core.exceptions import ValidationError
        with pytest.raises(ValidationError, match="monotone"):
            mlest(datasets.apple, method='monotone', backend='gpu')

    def test_em_gpu_fp64_is_a_known_backend(self):
        # Documented backend='gpu_fp64' must not be rejected as "Unknown
        # backend" for EM. On CUDA machines it runs (EM is float64 on CUDA
        # anyway); without CUDA it raises the GPU-availability RuntimeError.
        from pystatistics.core.exceptions import ValidationError
        try:
            res = mlest(datasets.apple, method='em', backend='gpu_fp64')
        except RuntimeError:
            pass  # no CUDA on this machine — correct loud failure
        except ValidationError as exc:
            pytest.fail(f"gpu_fp64 rejected as unknown for EM: {exc}")


class TestPrecisionAwareTolDefault:
    """mlest must not clobber the backend's precision-calibrated tol."""

    def test_direct_backend_receives_no_tol_when_unspecified(self, monkeypatch):
        captured = {}
        from pystatistics.mvnmle.backends import _direct

        original = _direct.run_direct_solve

        def spy(objective_factory, **kwargs):
            captured.update(kwargs)
            return original(objective_factory, **kwargs)

        # Patch where the backends resolved it from.
        import pystatistics.mvnmle.backends.gpu as gpu_mod
        import pystatistics.mvnmle.backends.numpy_cholesky as numpy_mod
        monkeypatch.setattr(gpu_mod, 'run_direct_solve', spy)
        monkeypatch.setattr(numpy_mod, 'run_direct_solve', spy)

        mlest(datasets.apple)  # default: forward-Cholesky CPU path, FP64
        assert captured.get('tol') == 1e-5  # backend's own FP64 default

    def test_user_tol_still_forwarded(self, monkeypatch):
        captured = {}
        from pystatistics.mvnmle.backends import _direct
        original = _direct.run_direct_solve

        def spy(objective_factory, **kwargs):
            captured.update(kwargs)
            return original(objective_factory, **kwargs)

        import pystatistics.mvnmle.backends.gpu as gpu_mod
        import pystatistics.mvnmle.backends.numpy_cholesky as numpy_mod
        monkeypatch.setattr(gpu_mod, 'run_direct_solve', spy)
        monkeypatch.setattr(numpy_mod, 'run_direct_solve', spy)

        mlest(datasets.apple, tol=3e-6)
        assert captured.get('tol') == 3e-6
