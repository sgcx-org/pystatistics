"""
CPU backend for MVN MLE using BFGS optimization.

Uses the R-exact CPUObjectiveFP64 with scipy.optimize.minimize.
"""

import numpy as np
from scipy.optimize import minimize

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.mvnmle.design import MVNDesign
from pystatistics.mvnmle.solution import MVNParams
from pystatistics.mvnmle._objectives.cpu import CPUObjectiveFP64


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

        # Run optimization
        with timer.section('optimization'):
            opt_result = minimize(
                objective.compute_objective,
                theta0,
                jac=objective.compute_gradient,
                method=method,
                options={
                    'maxiter': max_iter,
                    'gtol': tol,
                    'disp': False,
                }
            )

        # Extract parameters
        with timer.section('parameter_extraction'):
            mu, sigma, loglik = objective.extract_parameters(opt_result.x)

        # Compute gradient norm
        grad_norm = None
        if hasattr(opt_result, 'jac') and opt_result.jac is not None:
            grad_norm = float(np.max(np.abs(opt_result.jac)))

        if not opt_result.success:
            msg = getattr(opt_result, 'message', 'Unknown convergence failure')
            warnings_list.append(f"Optimization did not converge: {msg}")

        timer.stop()

        params = MVNParams(
            muhat=mu,
            sigmahat=sigma,
            loglik=loglik,
            n_iter=getattr(opt_result, 'nit', 0),
            converged=opt_result.success,
            gradient_norm=grad_norm,
        )

        return Result(
            params=params,
            info={
                'method': method,
                'objective_value': float(opt_result.fun),
                'n_function_evals': getattr(opt_result, 'nfev', 0),
                'n_gradient_evals': getattr(opt_result, 'njev', 0),
                'message': getattr(opt_result, 'message', ''),
                'parameterization': 'inverse_cholesky',
            },
            timing=timer.result(),
            backend_name=self.name,
            warnings=tuple(warnings_list),
        )
