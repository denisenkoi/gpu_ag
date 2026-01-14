"""Base class for block optimizers."""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import numpy as np


class BaseBlockOptimizer(ABC):
    """
    Abstract base class for segment block optimization algorithms.

    All optimizers must implement optimize() and return the same tuple format.
    """

    def __init__(
        self,
        device: str = 'cuda',
        angle_range: float = 2.5,
        angle_step: float = 0.2,
        mse_weight: float = 5.0,
        chunk_size: Optional[int] = None,
    ):
        """
        Initialize optimizer.

        Args:
            device: 'cuda' or 'cpu'
            angle_range: Max angle deviation from trajectory (degrees)
            angle_step: Angle grid step for bruteforce (degrees)
            mse_weight: Weight for MSE in score formula
            chunk_size: GPU chunk size (auto-detect if None)
        """
        self.device = device
        self.angle_range = angle_range
        self.angle_step = angle_step
        self.mse_weight = mse_weight
        self.chunk_size = chunk_size

    @abstractmethod
    def optimize(
        self,
        segment_indices: List[Tuple[int, int]],
        start_shift: float,
        trajectory_angle: float,
        well_md: np.ndarray,
        well_tvd: np.ndarray,
        well_gr: np.ndarray,
        type_tvd: np.ndarray,
        type_gr: np.ndarray,
        **kwargs
    ) -> Tuple[float, float, np.ndarray, float]:
        """
        Optimize angles for a block of segments.

        Args:
            segment_indices: List of (start_idx, end_idx) for each segment
            start_shift: Initial TVD shift at block start
            trajectory_angle: Reference trajectory angle (degrees)
            well_md: Well MD array
            well_tvd: Well TVD array
            well_gr: Well GR (gamma ray) array
            type_tvd: TypeLog TVD array
            type_gr: TypeLog GR array
            **kwargs: Algorithm-specific parameters

        Returns:
            Tuple of:
            - best_pearson: Best Pearson correlation achieved
            - best_end_shift: End TVD shift for best solution
            - best_angles: Array of optimal angles (degrees) for each segment
            - best_start_shift: Start shift (same as input for most cases)
        """
        pass

    @property
    def name(self) -> str:
        """Algorithm name for logging."""
        return self.__class__.__name__.replace('Optimizer', '').upper()
