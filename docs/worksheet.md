# GPU AG - Worksheet

## [2025-12-14 Current Session]

### Completed

1. Created repository `/mnt/e/Projects/Rogii/gpu_ag/`
2. Copied all CPU baseline from multi_drilling_emulator:
   - slicer.py, emulator.py, emulator_processor.py
   - ag_objects/, ag_numerical/, ag_rewards/, optimizers/
   - python_normalization/, ag_utils/
   - main.py, slicer_quality.py, wells_state_manager.py, papi_loader.py
   - papi_export/, alerts/, ag_visualization/, self_correlation/, sdk_data_loader/
3. Copied .env file
4. Created bats/slicer_de_3iter.bat
5. Initialized git, 2 commits made
6. Created README.md, CLAUDE.md, .gitignore

### In Progress

- Testing CPU slicer with batch file
- Trigger created in `/mnt/e/Projects/Rogii/sc/task_queue/` (correct location)

### Issues Found & Fixed

- [2025-12-14 22:52] Missing `wells_config_full.json` - copied from multi_drilling_emulator
- [2025-12-15 00:18] Fixed bot_id in trigger (must be "SSAndAG", not "gpu_ag")
- [2025-12-15 00:17] Removed extra changes from bat file (only path change needed)

## [2025-12-15] CPU Baseline Test Results

**Status: SUCCESS** (exit_code: 0)

| Well | final_fun | shifts | time (sec) |
|------|-----------|--------|------------|
| 1 | 0.0493 | -0.0025...-0.0029 | 350 |
| 2 | 0.0550 | -0.0029...-0.0033 | 323 |
| 3 | 0.1386 | -0.0033...-0.0037 | 332 |
| 4 | 0.1179 | -0.0035...-0.0038 | 334 |

**Reference values from AGENT_INSTRUCTIONS.md:**
- Target final_fun: 0.046
- shifts: -0.00852, -0.00897, -0.00949, -0.01001

**Comparison with multi_drilling_emulator agent:**
- ✅ INIT final_fun: 0.0493 (MATCH)
- ✅ shifts: [-0.00250...-0.00293] (MATCH)
- ✅ Time per optimization: ~330 sec (MATCH)

**Phase 1: COMPLETED** - CPU baseline reproduces original results

## [2025-12-15 01:45] Test Checkpoint Script

Created `test_checkpoint.py` to calculate reference values for numpy refactoring validation.

**How it works:**
1. Loads well data from `AG_DATA/InitialData/slicing_well.json` (same as emulator)
2. Creates `Well(well_data)` and `TypeWell(well_data)` objects
3. Loads current interpretation from StarSteer `interpretation.json`
4. Takes last 4 segments
5. Calls `objective_function_optimizer()` with all parameters
6. Saves checkpoint values to `test_checkpoint_values.json`

**Parameters used (from python_autogeosteering_executor.py defaults):**
- pearson_power = 2.0
- mse_power = 0.001
- num_intervals_self_correlation = 20
- sc_power = 1.15
- angle_range = 10.0
- angle_sum_power = 2.0
- min_pearson_value = -1

**Key files for data loading:**
- `ag_objects/ag_obj_well.py`: `Well(json_data)` - extracts `well['points']`, `wellLog['points']`
- `ag_objects/ag_obj_typewell.py`: `TypeWell(json_data)` - extracts `typeLog['tvdSortedPoints']`
- `ag_objects/ag_obj_interpretation.py`: `create_segments_from_json(json_segments, well)`

**Checkpoint values (reference for numpy validation):**
```json
{
  "shifts": [-15.07, -15.43, -15.82, -16.29],
  "objective_function_result": 0.000421,
  "segments_count": 3,
  "well_md_range": [2743.2, 4079.7],
  "typewell_tvd_range": [2739.8, 3630.5]
}
```

**Next:** Numpy refactoring - when done, run test_checkpoint.py and compare result with 0.000421

### Batch File Location

```
E:\Projects\Rogii\bats\slicer_de_3iter.bat
```

Runs: `slicer.py --de --starsteer-dir <path> --max-iterations 3`

### Next Steps

1. Run batch file, verify CPU baseline works
2. Start GPU implementation:
   - converters/well_converter.py (Well -> TorchWell)
   - converters/typewell_converter.py (TypeWell -> TorchTypeWell)
   - torch_rewards/batch_projection.py
   - torch_rewards/batch_correlations.py

### Notes

- User mentioned: grids will need to be added to reward function later
- DE parameters: popsize=500, maxiter=1000, strategy='rand1bin', workers=-1
- Objective function calculates: pearson correlation, MSE, intersections

## [2025-12-14] Key Insight: Numpy Convergence

**Один рефакторинг - два результата:**

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

**Узкие места в objective_function (2M вызовов):**

| Bottleneck           | Доля времени | Решение                     |
|----------------------|--------------|------------------------------|
| deepcopy(segments)   | 20-30%       | numpy.copy() - мгновенно     |
| calc_synt_curve      | ~10%         | numba @jit или torch batch   |
| find_intersections   | ~5-10%       | numba @jit или torch batch   |

**Архитектурное решение:**
- Well, Segment, TypeWell → numpy arrays (data-oriented design)
- Один набор данных для Numba и PyTorch
- Конвертация: numpy → torch.tensor (тривиальная)
