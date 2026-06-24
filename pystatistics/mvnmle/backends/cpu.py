"""
CPU backend for MVN MLE using BFGS optimization.

Uses the R-exact CPUObjectiveFP64 with scipy.optimize.minimize.
"""

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.mvnmle.design import MVNDesign
from pystatistics.mvnmle.solution import MVNParams
from pystatistics.mvnmle._objectives.cpu import CPUObjectiveFP64
from pystatistics.mvnmle.backends._optimize import run_scaled_minimize


class CPUMLEBackend:
    """
    CPU backend for MVN MLE.

    Uses R-exact inverse Cholesky parameterization with BFGS optimization.
    This is the reference implementation matching R's mvnmle package.
    """

    @property
    def name(self) -> str:
        return 'cpu_bfgs_fp64'

    def solve(
        self,
        design: MVNDesign,
        *,
        method: str = 'BFGS',
        tol: float = 1e-5,
        max_iter: int = 100,
    ) -> Result[MVNParams]:
        """
        Solve MVN MLE using CPU.

        Parameters
        ----------
        design : MVNDesign
            Data design wrapper
        method : str
            Optimization method ('BFGS', 'L-BFGS-B', 'Nelder-Mead', 'Powell')
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum iterations

        Returns
        -------
        Result[MVNParams]
        """
        timer = Timer()
        timer.start()
        warnings_list = []

        # Create objective function
        with timer.section('objective_setup'):
            objective = CPUObjectiveFP64(design.data, validate=False)

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
                'objective_value': opt.objective_value,
                'n_function_evals': opt.n_function_evals,
                'n_gradient_evals': opt.n_gradient_evals,
                'message': opt.message,
                'parameterization': 'inverse_cholesky',
            },
            timing=timer.result(),
            backend_name=self.name,
            warnings=tuple(warnings_list),
        )
