# GPU Autogeosteering - Agent Instructions

## Project Goal

Accelerate Differential Evolution optimization 10-100x using GPU batch processing.

Current CPU: scipy.differential_evolution (popsize=500, maxiter=1000, sequential)
Target GPU: EvoTorch + PyTorch (batch parallel processing)

## Jira

- Project: MDE (Multidrilling Emulator)
- Epic: ME-20 (GPU Vectorized Optimization for Autogeosteering)
- Tasks: ME-21 through ME-26

## Запуск из WSL (ПРЕДПОЧТИТЕЛЬНО)

**GPU доступна из WSL напрямую!**

```bash
# Проверка GPU
/usr/lib/wsl/lib/nvidia-smi

# Окружение с PyTorch 2.8 + CUDA 12.8 + RTX 5090:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

# Запуск скриптов напрямую (без триггеров):
cd /mnt/e/Projects/Rogii/gpu_ag
python benchmark_batch.py
```

**Преимущества WSL:**
- PyTorch 2.8.0+cu128 с нативной поддержкой RTX 5090 (sm_120)
- Не нужны триггеры и bat файлы
- Прямой доступ к консольному выводу
- Файловая система общая: `/mnt/e/Projects/...`

**Память GPU:**
- RTX 5090: 32GB
- LLM (llama-server) занимает ~30GB
- Для batch 500 нужно ~1-2GB - должно хватить

## Task Triggers (альтернатива для Windows)

**Используй если нужен Windows Python (Anaconda rl env):**

- Триггеры кладем в `/mnt/e/Projects/Rogii/sc/task_queue/`
- BAT файлы кладем в `/mnt/e/Projects/Rogii/bats/`
- В триггере указываем только имя файла (без пути): `"script_path": "slicer_de_3iter.bat"`
- **TIMESTAMP:** Используй `date +%s` для актуального timestamp! Триггеры старше 2 часов помечаются stale
- После создания триггера ставь будильник на 2-3 минуты для проверки
- Если триггер не работает - спроси агента AUTO для диагностики

**Пример:**
```bash
# Получить timestamp
date +%s  # например: 1765803456

# Создать триггер
{"task_id":"task_1765803456","type":"run_bat","script_path":"my_test.bat","bot_id":"SSAndAG","created_at":1765803456}
```

## Data Sources

**Все данные берутся из JSON файлов (StarSteer не используется):**

```
/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de/
├── AG_DATA/InitialData/slicing_well.json   # Well + TypeWell данные
└── interpretation.json                      # Сегменты интерпретации

/mnt/e/Projects/Rogii/gpu_ag/cpu_baseline/
└── test_checkpoint_values.json              # Эталонные значения для валидации
```

**Загрузка данных:**
```python
with open(well_json_path, 'r') as f:
    well_data_json = json.load(f)
well = Well(well_data_json)
typewell = TypeWell(well_data_json)
```

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

### Phase 1: Verify CPU Baseline Works ✅ COMPLETED

- [x] Create isolated repository gpu_ag
- [x] Copy CPU baseline from multi_drilling_emulator
- [x] Create test batch file (bats/slicer_de_3iter.bat)
- [x] **Run CPU slicer with DE mode (3 iterations)** - SUCCESS
- [x] **Compare results with reference from multi_drilling_emulator agent:**
  - INIT final_fun: 0.0493 (MATCH)
  - shifts: -0.00250...-0.00293 (MATCH)
  - Time per optimization: ~330 sec (MATCH)
- [x] **Create test_checkpoint.py** - reference values for numpy validation
  - objective_function_result: 0.00042138593474439547
  - Intermediate results: tvt, synt_curve, value (124 points, indices 4221-4344)
  - File: cpu_baseline/test_checkpoint_values.json

### Phase 2: Numpy Refactoring ✅ COMPLETED

**Результаты валидации [2025-12-15 03:34]:**

**Checkpoint 2.1: Конвертеры данных** ✅
- [x] numpy_funcs/converters.py - well_to_numpy, typewell_to_numpy, segments_to_numpy

**Checkpoint 2.2: calc_horizontal_projection** ✅
- [x] numpy_funcs/projection.py - numpy версия
- [x] **TVT:** Max difference = 0.00e+00 (идеальное совпадение)
- [x] **SYNT_CURVE:** Max difference = 0.00e+00 (идеальное совпадение)

**Checkpoint 2.3: objective_function_optimizer** ✅
- [x] numpy_funcs/correlations.py - pearson_numpy, mse_numpy
- [x] numpy_funcs/self_correlation.py - find_intersections_numpy
- [x] numpy_funcs/objective.py - objective_function_numpy
- [x] **РЕЗУЛЬТАТ:** 0.00042138593474439444 vs эталон 0.00042138593474439547
- [x] **РАЗНИЦА:** 1.03e-18 (в пределах floating point precision)

### Phase 3: PyTorch/GPU Implementation ✅ COMPLETED

**Цель:** Перевести numpy на torch.tensor, запустить на GPU. После этого - ТОЛЬКО тензорные расчеты.

**Checkpoint 3.1: Тензорные структуры данных** ✅
- [x] torch_funcs/converters.py (numpy_to_torch, segments_numpy_to_torch)
- [x] device='cuda', dtype=float64

**Checkpoint 3.2: Тензорная проекция** ✅
- [x] torch_funcs/projection.py - torch версия calc_horizontal_projection
- [x] **ВАЛИДАЦИЯ:** TVT diff = 0.00e+00

**Checkpoint 3.3: Тензорная objective_function** ✅
- [x] torch_funcs/correlations.py - torch pearson, mse
- [x] torch_funcs/objective.py - torch objective_function
- [x] **ВАЛИДАЦИЯ:** diff = 2.71e-19

**Checkpoint 3.4: Batch processing (popsize=500)** ✅
- [x] torch_funcs/batch_objective.py - батчевая версия
- [x] **ВАЛИДАЦИЯ:** diff = 3.28e-10
- [!] **PERFORMANCE ISSUE:** 72 сек для 500 особей (sequential self-correlation)

### Phase 4: Full GPU Optimization Run ✅ COMPLETED

**Цель:** Полный прогон DE оптимизации на GPU.

**Benchmark достигнут [2025-12-17]:**
- Full DE optimization: **4.5 sec** (vs CPU ~330 sec)
- Speedup: **70-150x** (target was 10-100x) ✅✅✅

**Checkpoint 4.1: GPU DE Optimizer** ✅
- [x] `torch_funcs/gpu_optimizer.py` - DE на PyTorch
  - mutation: `v = x_r1 + F * (x_r2 - x_r3)`
  - crossover: `u = where(rand < CR, v, x)`
  - selection: `x = where(f(u) < f(x), u, x)`
  - Всё на GPU tensors, **73x speedup**

**Checkpoint 4.2: GPU Executor** ✅
- [x] `gpu_optimizer_fit.py` - drop-in replacement для CPU optimizer_fit
  - Конвертация Well/TypeWell → numpy → torch
  - **153x speedup** vs CPU

**Checkpoint 4.3: Интеграция со slicer** ✅
- [x] `slicer_gpu.py` - monkey-patching wrapper
- [x] Запуск: `python slicer_gpu.py --de --starsteer-dir <path>`

**Checkpoint 4.4: Валидация** ✅
- [x] Full pipeline test: `test_full_pipeline.py`
- [x] Results match CPU baseline (Pearson diff < 0.001)
- [x] **72.7x speedup** achieved

**Итоговый результат:**
- CPU: ~330 сек на скважину
- GPU: **~4.5 сек на скважину** (73x speedup)

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
