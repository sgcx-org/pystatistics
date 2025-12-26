"""
Capability string constants for PyStatistics.

This module is the SINGLE SOURCE OF TRUTH for capability strings.
Import from here, never use raw strings.

Usage:
    from pystatistics.core.capabilities import (
        CAPABILITY_MATERIALIZED,
        CAPABILITY_STREAMING,
        CAPABILITY_REPEATABLE,
    )
    
    if ds.supports(CAPABILITY_MATERIALIZED):
        X, y = ds['X'], ds['y']
"""

# Data can be returned as full numpy arrays in memory
CAPABILITY_MATERIALIZED = 'materialized'

# Data can be yielded in batches (for datasets larger than RAM)
CAPABILITY_STREAMING = 'streaming'

# Data is already on GPU as PyTorch tensors
CAPABILITY_GPU_NATIVE = 'gpu_native'

# Data can be iterated multiple times (for residuals, fitted values)
CAPABILITY_REPEATABLE = 'repeatable'

# Sufficient statistics can be computed/provided directly
CAPABILITY_SUFFICIENT_STATISTICS = 'sufficient_statistics'

# All capabilities as a frozenset for validation
ALL_CAPABILITIES = frozenset({
    CAPABILITY_MATERIALIZED,
    CAPABILITY_STREAMING,
    CAPABILITY_GPU_NATIVE,
    CAPABILITY_REPEATABLE,
    CAPABILITY_SUFFICIENT_STATISTICS,
})

__all__ = [
    'CAPABILITY_MATERIALIZED',
    'CAPABILITY_STREAMING', 
    'CAPABILITY_GPU_NATIVE',
    'CAPABILITY_REPEATABLE',
    'CAPABILITY_SUFFICIENT_STATISTICS',
    'ALL_CAPABILITIES',
]
