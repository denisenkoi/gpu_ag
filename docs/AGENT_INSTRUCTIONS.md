# GPU Autogeosteering - Agent Instructions

## Project Goal

GPU-ускоренная интерпретация скважины с использованием EvoTorch.

## Jira

- Project: MDE (Multidrilling Emulator)
- Epic: ME-20 (GPU Vectorized Optimization for Autogeosteering)

## Текущее состояние (2025-12-19)

### ДОСТИГНУТО ✓

1. **EvoTorch SNES работает**
   - 5 restarts × 200 iter × 100 pop
   - ~5 сек на шаг
   - fun ~0.1-0.3 (хорошо!)

2. **Итеративная интерпретация работает**
   - `optimize_step()` - один шаг оптимизации
   - `update_slicing_well()` - обновление файла (standalone режим)
   - Шаг 30м, lookback 200м
   - Success rate: ~83% (5/6 итераций)

3. **Сшивка сегментов работает**
   - Truncation at lookback point
   - Interpolation of stitch_shift
   - frozen_md защищает начальную часть

### Рабочие файлы
| Файл | Описание |
|------|----------|
| `test_gpu_de_correct.py` | Итеративный тест с EvoTorch SNES |
| `torch_funcs/batch_objective.py` | Batch objective function |
| `torch_funcs/converters.py` | Конвертеры данных |
| `cpu_baseline/` | Reference implementation (DO NOT TOUCH) |

### Запуск
```bash
python test_gpu_de_correct.py           # 1 итерация
python test_gpu_de_correct.py -n 5      # 5 итераций
python test_gpu_de_correct.py -n 0      # до конца скважины
python test_gpu_de_correct.py --no-standalone  # без обновления slicing_well
```

## Реализовано (2025-12-20)

### 1. GPU Executor с EvoTorch SNES ✓
- Создан `gpu_executor.py` для интеграции со slicer_gpu.py
- EvoTorch SNES вместо scipy DE
- Параметры: `GPU_N_RESTARTS=5`, `GPU_POPSIZE=100`, `GPU_MAXITER=200`

### 2. Поддержка PseudoTypeLog ✓
- Параметр `USE_PSEUDOTYPELOG=false` в .env
- При true: typeLog → pseudoTypeLog, tvdTypewellShift = 0

### 3. Предотвращение утечки данных ✓
- `pseudoLogEndMd` в ag_params (INIT)
- `_update_pseudo_log_end_md()` обновляет ag_config.json (STEP)
- Формула: `pseudo_log_end_md = current_md - lookback`

### Jira: RND-765

## СЛЕДУЮЩИЙ ЭТАП: Тестирование pseudoLogEndMd

### Задача
Проверить что pseudoLogEndMd влияет на данные typeLog при каждом STEP.

### План
1. **Модифицировать код**: при каждом STEP записывать typeLog snapshot в JSON
   - Файл: `typeLog_history.json`
   - Формат: `{MD: [typeLog_points_sample], ...}`
   - Записывать первые/последние N точек для сравнения

2. **Запустить тест**: slicer_gpu.py на Well1798~EGFDL

3. **Анализ**: сравнить typeLog snapshots
   - Если они все чуть-чуть разные → pseudoLogEndMd работает
   - Если одинаковые → C++ пока не использует этот параметр

4. **Сравнение с reference**: проверить качество интерпретации

### Ключевые файлы
- `gpu_executor.py` — GPU executor с EvoTorch SNES
- `slicer.py` — pseudoLogEndMd (INIT + STEP)
- `ag_config.json` — хранит pseudoLogEndMd для каждого STEP

### Параметры
- `tvdTypewellShift`: 21.56м (typeLog) / 0 (pseudoTypeLog)
- `LOOKBACK_DISTANCE`: 200м
- Well: Well1798~EGFDL, startMd: 3718.56м = 12200ft

## КРИТИЧЕСКИЕ ПРАВИЛА

### 1. НИКОГДА НЕ ПИСАТЬ САМОПИСНЫЕ ОПТИМИЗАТОРЫ!
**ВСЕГДА использовать коробочные решения:**
- EvoTorch (SNES, XNES, CMAES, CEM)
- scipy.optimize
- optuna

Самописный DE был удалён 2025-12-18 после многочисленных проблем.

### 2. cpu_baseline/ - НЕ ТРОГАТЬ
Это reference implementation. Любые изменения только после согласования.

### 3. Результаты в worksheet.md
Все эксперименты и их результаты записывать в `docs/worksheet.md`.

## Data Sources

```
/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de/
├── AG_DATA/InitialData/slicing_well.json   # Well + TypeWell + interpretations (READ)
└── interpretation.json                      # Output for StarSteer (WRITE ONLY!)
```

### КРИТИЧНО: interpretation.json = WRITE ONLY
**НИКОГДА не читать из interpretation.json!**
- Этот файл перезаписывается GPU output'ом
- Всегда читать интерпретацию из `slicing_well.json`
- В standalone режиме (тесты) обновляем `slicing_well.json` после каждого шага

### Ключевые поля в slicing_well.json
- `autoGeosteeringParameters.startMd` - начало оптимизации
- `interpretation.segments` - ручная интерпретация
- `starredInterpretation.segments` - reference для сравнения

## Запуск из WSL

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm
cd /mnt/e/Projects/Rogii/gpu_ag
python test_gpu_de_correct.py
```

## EvoTorch алгоритмы

| Алгоритм | Описание |
|----------|----------|
| **SNES** | Separable NES (используем) |
| **XNES** | Exponential NES |
| **CMAES** | CMA-ES (золотой стандарт) |
| **CEM** | Cross-Entropy Method |
