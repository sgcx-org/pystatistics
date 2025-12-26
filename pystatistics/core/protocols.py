"""
Core protocols for PyStatistics.

These define structural interfaces that domain-specific implementations must satisfy.
We use Protocol (structural typing) rather than ABC (nominal typing) to allow
flexibility while maintaining type safety.

Design Principles:
    - Minimal contracts: prescribe only what's truly universal
    - Capability-driven: use supports() for optional features
    - Type-safe: use generics to preserve type information through pipelines
"""

from typing import Protocol, TypeVar, Any, runtime_checkable

# Type variables for generic payloads
P = TypeVar('P')  # Parameter payload type
D = TypeVar('D')  # DataSource type


@runtime_checkable
class DataSource(Protocol):
    """
    Minimal protocol for any data container used in statistical computation.
    
    Domain-specific implementations (RegressionDesign, MVNDesign, SurvivalDesign)
    implement this protocol and add their own methods for domain-specific access.
    
    This protocol intentionally prescribes very little—it exists to establish
    a common interface for tooling (profiling, logging, serialization) without
    forcing domains to pretend they have the same structure.
    
    The supports() method enables capability-driven extension: streaming, GPU-native
    tensors, second-pass computation, etc. can be added without breaking changes.
    """
    
    @property
    def n_observations(self) -> int:
        """Number of statistical units (rows, subjects, etc.)."""
        ...
    
    @property
    def metadata(self) -> dict[str, Any]:
        """
        Domain-specific metadata.
        
        This is intentionally unstructured—each domain defines what's relevant.
        
        Examples:
            Regression: {'n': 100, 'p': 5, 'has_intercept': True, 'rank': 5}
            MVN: {'n': 100, 'p': 5, 'n_complete': 80, 'n_patterns': 3}
            Survival: {'n': 100, 'n_events': 45, 'n_censored': 55}
        """
        ...
    
    def supports(self, capability: str) -> bool:
        """
        Check if this data source supports a given capability.
        
        Standard capability strings (domains may define additional ones):
            'materialize': Can return full data as numpy arrays
            'stream': Can yield data in batches
            'gpu_tensors': Can return data as PyTorch tensors on device
            'second_pass': Supports multiple iterations over data
            'sufficient_stats': Can compute sufficient statistics directly
        
        Args:
            capability: The capability to check
            
        Returns:
            True if the capability is supported, False otherwise
            
        Note:
            Unknown capabilities MUST return False, never raise.
            This allows forward-compatible capability checking.
        """
        ...


@runtime_checkable
class Backend(Protocol[D, P]):
    """
    Protocol for computational backends.
    
    Each backend knows how to take a domain-specific DataSource and produce
    a domain-specific parameter payload. The backend handles all hardware-
    specific computation (CPU/GPU, precision, batching).
    
    Backends are stateless—all configuration is passed via the DataSource
    or at construction time. This makes them easy to test and swap.
    
    Type Parameters:
        D: The DataSource type this backend accepts
        P: The parameter payload type this backend produces
    """
    
    @property
    def name(self) -> str:
        """
        Backend identifier.
        
        Convention: '{device}_{algorithm}'
        Examples: 'cpu_qr', 'cpu_svd', 'gpu_cholesky', 'cpu_em'
        """
        ...
    
    def solve(self, design: D) -> 'Result[P]':
        """
        Execute the statistical computation.
        
        Args:
            design: Domain-specific data container implementing DataSource
            
        Returns:
            Result envelope containing parameter payload and metadata
            
        Raises:
            ConvergenceError: If iterative method fails to converge
            NumericalError: If numerical issues prevent solution (singularity, etc.)
            ValidationError: If design is invalid for this backend
        """
        ...
