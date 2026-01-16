"""
Optimization algorithms for GPU AG.

Registry of available algorithms:
- BRUTEFORCE: Exhaustive grid search (current best, RMSE 2.99m)
- CMAES: Covariance Matrix Adaptation Evolution Strategy (EvoTorch)
- SNES: Separable Natural Evolution Strategy (EvoTorch)
- SCIPY_DE: Differential Evolution with GPU batch evaluation (TODO)
- PSO: Particle Swarm Optimization (TODO)
- MONTECARLO: Random sampling

Usage:
    from optimizers import get_optimizer, AVAILABLE_ALGORITHMS

    optimizer = get_optimizer('BRUTEFORCE', angle_range=2.5, mse_weight=5.0)
    result = optimizer.optimize(segment_indices, start_shift, ...)
"""

from .base import BaseBlockOptimizer, OptimizeResult
from .objective import (
    BlockData,
    BeamPrefix,
    prepare_block_data,
    compute_loss_batch,
    compute_score_batch,
    compute_std_batch,
    GPU_DTYPE,
)

# Registry of available optimizers
_OPTIMIZER_REGISTRY = {}


def register_optimizer(name: str):
    """Decorator to register an optimizer class."""
    def decorator(cls):
        _OPTIMIZER_REGISTRY[name.upper()] = cls
        return cls
    return decorator


# Import algorithm modules to trigger registration
from . import bruteforce
from . import greedy_bruteforce
from . import continuous_beam
from . import scipy_de
from . import evotorch_algos


def get_optimizer(name: str, **kwargs) -> BaseBlockOptimizer:
    """
    Get optimizer instance by name.

    Args:
        name: Algorithm name (case-insensitive)
        **kwargs: Arguments passed to optimizer constructor

    Returns:
        Optimizer instance

    Raises:
        ValueError: If algorithm not found
    """
    name_upper = name.upper()
    if name_upper not in _OPTIMIZER_REGISTRY:
        available = ', '.join(sorted(_OPTIMIZER_REGISTRY.keys()))
        raise ValueError(f"Unknown optimizer '{name}'. Available: {available}")
    return _OPTIMIZER_REGISTRY[name_upper](**kwargs)


def list_optimizers() -> list:
    """List available optimizer names."""
    return sorted(_OPTIMIZER_REGISTRY.keys())


# Convenience export
AVAILABLE_ALGORITHMS = property(lambda self: list_optimizers())

__all__ = [
    'BaseBlockOptimizer',
    'OptimizeResult',
    'BlockData',
    'BeamPrefix',
    'prepare_block_data',
    'compute_loss_batch',
    'compute_score_batch',
    'compute_std_batch',
    'GPU_DTYPE',
    'get_optimizer',
    'register_optimizer',
    'list_optimizers',
]
