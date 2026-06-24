"""
GPU backend for MVN MLE.

Supports CUDA (FP32 and FP64) and MPS (FP32 only).
Uses PyTorch autodiff for analytical gradients.
"""

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.mvnmle.design import MVNDesign
from pystatistics.mvnmle.solution import MVNParams
from pystatistics.mvnmle.backends._optimize import run_scaled_minimize


class GPUMLEBackend:
    """
    GPU backend for MVN MLE.

    Supports CUDA and MPS (Apple Silicon).
    Automatically selects precision based on device:
    - MPS: FP32 only (MPS doesn't support FP64)
    - CUDA consumer (RTX): FP32 by default
    - CUDA data center (A100/H100): FP64 available

    Args:
        device: 'cuda', 'mps', or 'auto'
        use_fp64: Force FP64 (CUDA only, raises on MPS)
    """

    def __init__(self, device: str = 'auto', use_fp64: bool = False):
        import torch

        self._torch = torch
        self._use_fp64 = use_fp64

        # Resolve device
        if device == 'auto':
            if torch.cuda.is_available():
                self._device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = 'mps'
            else:
                raise RuntimeError("No GPU available (need CUDA or MPS)")
        else:
            self._device = device

        # Validate MPS + FP64 conflict
        if self._device == 'mps' and use_fp64:
            raise RuntimeError(
                "MPS does not support FP64. Use use_fp64=False "
                "or use backend='cpu' for double precision."
            )

    @property
    def name(self) -> str:
        precision = 'fp64' if self._use_fp64 else 'fp32'
        return f'gpu_{self._device}_{precision}'

    def solve(
        self,
        design: MVNDesign,
        *,
        method: str | None = None,
        tol: float | None = None,
        max_iter: int = 100,
    ) -> Result[MVNParams]:
        """
        Solve MVN MLE using GPU.

        Parameters
        ----------
        design : MVNDesign
            Data design wrapper
        method : str or None
            Optimization method. Auto-selected based on precision.
        tol : float or None
            Convergence tolerance. Auto-selected based on precision.
        max_iter : int
            Maximum iterations

        Returns
        -------
        Result[MVNParams]
        """
        timer = Timer(sync_cuda=(self._device == 'cuda'))
        timer.start()
        warnings_list = []

        # Select objective class and defaults based on precision
        if self._use_fp64:
            from pystatistics.mvnmle._objectives.gpu_fp64 import GPUObjectiveFP64
            ObjectiveClass = GPUObjectiveFP64
            default_method = 'BFGS'
            default_tol = 1e-5
        else:
            from pystatistics.mvnmle._objectives.gpu_fp32 import GPUObjectiveFP32
            ObjectiveClass = GPUObjectiveFP32
            default_method = 'L-BFGS-B'
            default_tol = 1e-3

        method = method or default_method
        tol = tol or default_tol

        # Create objective
        with timer.section('objective_setup'):
            objective = ObjectiveClass(design.data, device=self._device)

        # Get initial parameters
        with timer.section('initial_parameters'):
            theta0 = objective.get_initial_parameters()

        # Run optimization on the per-observation-scaled objective so that
        # `gtol` is a meaningful, dataset-size-invariant convergence test
        # (see backends/_optimize.py). Scaling does not move the optimum, so
        # the estimates and log-likelihood below are unchanged.
        with timer.section('optimization'):
            opt = run_scaled_minimize(
                objective, theta0, method=method, tol=tol, max_iter=max_iter
            )

        # Extract parameters from the (unscaled) objective.
        with timer.section('parameter_extraction'):
            mu, sigma, loglik = objective.extract_parameters(opt.x)

        if not opt.success:
            msg = opt.message or 'Unknown convergence failure'
            warnings_list.append(f"Optimization did not converge: {msg}")

        # Clean up GPU memory
        objective.clear_cache()

        timer.stop()

        params = MVNParams(
            muhat=mu,
            sigmahat=sigma,
            loglik=loglik,
            n_iter=opt.n_iter,
            converged=opt.success,
            gradient_norm=opt.gradient_norm,
        )

        return Result(
            params=params,
            info={
                'method': method,
                'device': self._device,
                'precision': 'fp64' if self._use_fp64 else 'fp32',
                'objective_value': opt.objective_value,
                'n_function_evals': opt.n_function_evals,
                'n_gradient_evals': opt.n_gradient_evals,
                'message': opt.message,
                'parameterization': 'cholesky',
            },
            timing=timer.result(),
            backend_name=self.name,
            warnings=tuple(warnings_list),
        )
