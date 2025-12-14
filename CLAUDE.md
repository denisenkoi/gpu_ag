# GPU Autogeosteering Optimization Project

## Project Structure

```
gpu_ag/
├── cpu_baseline/           # Скопированный CPU код из multi_drilling_emulator
│   ├── slicer.py           # Main slicer
│   ├── emulator.py
│   ├── emulator_processor.py
│   ├── ag_objects/         # Well, TypeWell, Segment
│   ├── ag_numerical/       # optimizer_fit (CPU DE)
│   ├── ag_rewards/         # correlations
│   └── optimizers/         # PythonAutoGeosteeringExecutor
│
├── gpu_ag/                 # GPU implementation (NEW)
│   ├── torch_objects/      # TorchWell, TorchTypeWell, TorchSegmentBatch
│   ├── torch_rewards/      # batch_projection, batch_correlations
│   ├── torch_optimizer/    # EvoTorch DE wrapper
│   └── converters/         # CPU <-> GPU converters
│
├── tests/
│   └── fixtures/           # Test data
│
└── docs/
```

## Development Rules

- **cpu_baseline/** - НЕ ТРОГАТЬ, это reference для сравнения
- **gpu_ag/** - разработка GPU версии
- Jira project: MDE (Multidrilling Emulator)
- Epic: ME-20
- Tasks: ME-21 through ME-26

## Workflow

1. Разрабатываем в gpu_ag/
2. Тестируем: CPU (cpu_baseline) vs GPU (gpu_ag) на одних данных
3. Когда готово - интегрируем обратно в multi_drilling_emulator

## Key Files

- `cpu_baseline/ag_objects/ag_obj_well.py` - Well class (reference)
- `cpu_baseline/ag_objects/ag_obj_typewell.py` - TypeWell class (reference)
- `cpu_baseline/ag_numerical/ag_func_optimizer.py` - CPU DE optimizer
- `cpu_baseline/ag_rewards/ag_func_correlations.py` - CPU reward functions

## Comments in code

- English only (code style from existing codebase)
