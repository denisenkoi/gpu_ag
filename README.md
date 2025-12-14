# GPU Autogeosteering Optimization

PyTorch-based vectorized optimization for autogeosteering.

## Goal

Accelerate Differential Evolution optimization by 10-100x using GPU batch processing.

Current CPU implementation processes population (500 individuals) sequentially.
GPU version processes entire population in parallel using tensor operations.

## Structure

```
gpu_ag/
├── cpu_baseline/     # Reference CPU implementation (from multi_drilling_emulator)
├── gpu_ag/           # GPU implementation with PyTorch
│   ├── torch_objects/    # Well, TypeWell as tensors
│   ├── torch_rewards/    # Batch correlation functions
│   ├── torch_optimizer/  # EvoTorch DE integration
│   └── converters/       # CPU <-> GPU conversion
├── tests/            # Test fixtures and unit tests
└── docs/             # Documentation
```

## Jira

- Project: MDE
- Epic: ME-20 (GPU Vectorized Optimization for Autogeosteering)
- Tasks: ME-21 through ME-26

## Reference Values

Target final_fun: 0.046
Reference shifts at optimum:
- shift[0]: -0.00852
- shift[1]: -0.00897
- shift[2]: -0.00949
- shift[3]: -0.01001

## Requirements

- Python 3.10+
- PyTorch 2.0+
- EvoTorch
- NumPy, SciPy (for CPU baseline)
