# GPU Autogeosteering - Agent Instructions

## Project Goal

Accelerate Differential Evolution optimization 10-100x using GPU batch processing.

Current CPU: scipy.differential_evolution (popsize=500, maxiter=1000, sequential)
Target GPU: EvoTorch + PyTorch (batch parallel processing)

## Jira

- Project: MDE (Multidrilling Emulator)
- Epic: ME-20 (GPU Vectorized Optimization for Autogeosteering)
- Tasks: ME-21 through ME-26

## Task Triggers

**ВАЖНО:**
- Триггеры кладем в `/mnt/e/Projects/Rogii/sc/task_queue/`
- BAT файлы кладем в `/mnt/e/Projects/Rogii/bats/`
- В триггере указываем только имя файла (без пути): `"script_path": "slicer_de_3iter.bat"`

## Architecture

### Key Insight: Numpy Convergence

Один рефакторинг - два результата:

```
Python objects (Well, Segment, TypeWell)
            ↓
    numpy arrays (pure data)
            ↓
    ┌───────┴───────┐
    ↓               ↓
  Numba           PyTorch
  @jit            torch.tensor
  CPU 3-5x        GPU 10-100x
```

### Data Structure

```python
# Вместо объектов - numpy arrays dict
well_data = {
    'md': np.array(...),      # (N,)
    'vs': np.array(...),      # (N,)
    'tvd': np.array(...),     # (N,)
    'value': np.array(...),   # (N,)
}

segments_data = np.array([...])  # (K, 6) - start_idx, end_idx, start_vs, end_vs, start_shift, end_shift
typewell_data = np.array([...])  # (M, 2) - tvd, value
```

### Repository Structure

```
gpu_ag/
├── cpu_baseline/     # Reference CPU code (DO NOT MODIFY!)
│   ├── slicer.py
│   ├── ag_objects/   # Well, TypeWell, Segment
│   ├── ag_numerical/ # optimizer_fit (CPU DE)
│   ├── ag_rewards/   # correlations
│   └── optimizers/   # PythonAutoGeosteeringExecutor
│
├── gpu_ag/           # GPU implementation (DEVELOP HERE)
│   ├── numpy_objects/    # numpy-based Well, TypeWell, Segment
│   ├── torch_rewards/    # batch_projection, batch_correlations
│   ├── torch_optimizer/  # EvoTorch DE wrapper
│   └── converters/       # Python objects -> numpy arrays
│
├── bats/             # Test scripts
└── docs/             # Documentation
```

## Reference Values (for validation)

- Target final_fun: 0.046
- Optimal shifts:
  - shift[0]: -0.00852
  - shift[1]: -0.00897
  - shift[2]: -0.00949
  - shift[3]: -0.01001

## PLAN / Checkpoints

### Phase 1: Verify CPU Baseline Works

- [x] Create isolated repository gpu_ag
- [x] Copy CPU baseline from multi_drilling_emulator
- [x] Create test batch file (bats/slicer_de_3iter.bat)
- [ ] **Run CPU slicer with DE mode (3 iterations)**
- [ ] **Compare results with reference from multi_drilling_emulator agent:**
  - Iteration 1: final_fun ~0.048, shifts near target
  - Iteration 2: final_fun ~0.047
  - Iteration 3: final_fun ~0.046
  - Check that results match (tolerance ~0.001)

### Phase 2: Refactor to Numpy Arrays

- [ ] Create converters/well_to_numpy.py (Well -> numpy arrays dict)
- [ ] Create converters/typewell_to_numpy.py (TypeWell -> numpy arrays)
- [ ] Create converters/segments_to_numpy.py (Segments list -> numpy 2D array)
- [ ] Create numpy_objects/objective_function_numpy.py (pure numpy version)
- [ ] Replace deepcopy with numpy.copy (eliminate 20-30% overhead)
- [ ] **Test: Run slicer with numpy-based objective function**
- [ ] **Verify results match original CPU baseline**

### Phase 3: Numba Acceleration (Optional)

- [ ] Add @numba.jit(nopython=True) to calc_synt_curve
- [ ] Add @numba.jit(nopython=True) to find_intersections
- [ ] Benchmark: compare with pure numpy version

### Phase 4: PyTorch/GPU Implementation

- [ ] Create torch_objects/torch_well.py (numpy arrays -> torch.tensor)
- [ ] Create torch_rewards/batch_projection.py
- [ ] Create torch_rewards/batch_correlations.py (pearson, mse, intersections)
- [ ] Integrate EvoTorch DE optimizer
- [ ] Create GPUAutoGeosteeringExecutor
- [ ] **Benchmark: CPU vs GPU on same data**

## Key CPU Files (reference)

- `cpu_baseline/ag_numerical/ag_func_optimizer.py` - DE optimizer with scipy
- `cpu_baseline/ag_rewards/ag_func_correlations.py` - objective_function_optimizer, pearson, mse
- `cpu_baseline/optimizers/python_autogeosteering_executor.py` - CPU executor
- `cpu_baseline/ag_objects/ag_obj_well.py` - Well class
- `cpu_baseline/ag_objects/ag_obj_typewell.py` - TypeWell class

## Bottlenecks in objective_function (2M calls per optimization)

| Bottleneck           | % Time | Solution                      |
|----------------------|--------|-------------------------------|
| deepcopy(segments)   | 20-30% | numpy.copy() - instant        |
| calc_synt_curve      | ~10%   | numba @jit or torch batch     |
| find_intersections   | ~5-10% | numba @jit or torch batch     |

## Development Rules

1. **cpu_baseline/** - DO NOT TOUCH, this is reference for comparison
2. **gpu_ag/** - all GPU development here
3. Comments in code: English only
4. Test on same data: CPU result must match GPU result (tolerance ~0.001)
