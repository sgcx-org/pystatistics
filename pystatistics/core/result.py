"""
Generic result container for all PyStatistics computations.

The Result class provides a standardized envelope that all domain-specific
results use. This enables shared tooling for timing, logging, reproducibility,
and serialization while allowing domains to define their own parameter structures.

Design decisions:
    - Generic over parameter payload P for type safety
    - info dict for flexible metadata (converged, iterations, diagnostics)
    - timing is optional (don't burden unit tests)
    - Immutable (frozen=True) for reproducibility
"""

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Any

P = TypeVar('P')  # Parameter payload type


@dataclass(frozen=True)
class Result(Generic[P]):
    """
    Immutable result envelope for statistical computations.
    
    Type Parameters:
        P: The domain-specific parameter payload type
        
    Attributes:
        params: Domain-specific parameters (coefficients, estimates, etc.)
        info: Structured metadata (method, convergence, diagnostics)
        timing: Execution timing breakdown, or None if not measured
        backend_name: Identifier of the backend that produced this result
        warnings: Non-fatal issues encountered during computation
        
    The frozen=True ensures results are immutable after creation, which is
    important for reproducibility and prevents accidental modification.
    
    Examples:
        >>> # Direct method (no convergence notion)
        >>> Result(
        ...     params=LinearParams(coefficients=beta),
        ...     info={'method': 'qr', 'rank': 5},
        ...     timing={'total_seconds': 0.01},
        ...     backend_name='cpu_qr'
        ... )
        
        >>> # Iterative method
        >>> Result(
        ...     params=MVNParams(mu=mu, sigma=sigma),
        ...     info={'method': 'em', 'converged': True, 'iterations': 23},
        ...     timing={'total_seconds': 0.5, 'e_step': 0.3, 'm_step': 0.2},
        ...     backend_name='cpu_em'
        ... )
    """
    params: P
    info: dict[str, Any]
    timing: dict[str, float] | None
    backend_name: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    
    def has_warning(self, substring: str) -> bool:
        """Check if any warning contains the given substring."""
        return any(substring in w for w in self.warnings)
