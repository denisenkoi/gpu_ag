"""
GPU-enabled emulator processor.

Extends base EmulatorProcessor with GPU executor support.
Select executor via .env: AUTOGEOSTEERING_EXECUTOR=cpu|python|gpu

Usage:
    from emulator_processor_gpu import GpuEmulatorProcessor
    processor = GpuEmulatorProcessor(config, results_dir)
"""
import os
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

from emulator_processor import EmulatorProcessor
from optimizers.base_autogeosteering_executor import BaseAutoGeosteeringExecutor

logger = logging.getLogger(__name__)


class GpuEmulatorProcessor(EmulatorProcessor):
    """
    Emulator processor with GPU executor support.

    Executor selection via AUTOGEOSTEERING_EXECUTOR env var:
    - 'cpu' or 'daemon': C++ daemon (original)
    - 'python': Python executor with scipy DE
    - 'gpu': GPU executor with multi-population DE (default if CUDA available)
    """

    def _create_executor(self) -> BaseAutoGeosteeringExecutor:
        """Create executor based on AUTOGEOSTEERING_EXECUTOR setting."""
        import torch
        from uuid import uuid4

        # Create unique work directory
        unique_work_dir = Path(self.work_dir) / f"executor_{uuid4().hex[:8]}"
        unique_work_dir.mkdir(parents=True, exist_ok=True)

        # Determine executor type
        executor_type = os.getenv('AUTOGEOSTEERING_EXECUTOR', 'auto').lower()

        # Auto-select: GPU if available, else Python
        if executor_type == 'auto':
            if torch.cuda.is_available():
                executor_type = 'gpu'
                logger.info("Auto-selected GPU executor (CUDA available)")
            else:
                executor_type = 'python'
                logger.info("Auto-selected Python executor (no CUDA)")

        if executor_type == 'gpu':
            # GPU executor
            from gpu_executor import GpuAutoGeosteeringExecutor
            logger.info(f"Creating GPU AutoGeosteering executor: {unique_work_dir}")
            return GpuAutoGeosteeringExecutor(str(unique_work_dir), self.results_dir)

        elif executor_type == 'python':
            # Python executor (scipy DE)
            from optimizers.python_autogeosteering_executor import PythonAutoGeosteeringExecutor
            logger.info(f"Creating Python AutoGeosteering executor: {unique_work_dir}")
            return PythonAutoGeosteeringExecutor(str(unique_work_dir), self.results_dir)

        elif executor_type in ('cpu', 'daemon'):
            # C++ daemon executor
            from optimizers.autogeosteering_executor import AutoGeosteeringExecutor
            logger.info(f"Creating C++ AutoGeosteering executor: {unique_work_dir}")
            return AutoGeosteeringExecutor(str(unique_work_dir))

        else:
            raise ValueError(f"Unknown executor type: {executor_type}. "
                           f"Use: auto, gpu, python, cpu, daemon")
