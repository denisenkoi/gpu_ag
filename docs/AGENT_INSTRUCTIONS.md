# GPU Autogeosteering - Agent Instructions

## Project Goal

Довести надёжность GPU DE оптимизации до ~100% сходимости при минимальном увеличении времени.

**Предыдущая фаза (RND-777):** Достигнуто 95% успеха с zeros+noise init, 28x speedup vs CPU.

## Jira

- Project: MDE (Multidrilling Emulator)
- Epic: ME-20 (GPU Vectorized Optimization for Autogeosteering)
- Предыдущая задача: RND-777

## Текущее состояние

### Достигнуто (RND-777)
- GPU DE работает: **2.1 сек** на оптимизацию (vs CPU ~60 сек)
- Speedup: **28x**
- Надёжность: **95%** с zeros+noise init (19/20 успехов)
- Параметры: popsize=500, maxiter=500, mutation=(0.5, 1.0), CR=0.7

### Проблема
При многократных запусках (production) 5% провалов недопустимо.
Нужно довести до ~100% без значительного увеличения времени.

### Ключевое наблюдение
**GPU недоиспользуется!**
- RTX 5090: 32GB, тысячи CUDA cores
- Текущий batch: 500 особей × 3 параметра = 1500 чисел
- Это <1% возможностей GPU
- Можно запустить 5-15 популяций параллельно почти без увеличения времени

## План исследований

### Шаг 1: Multi-population (3-15 штук одновременно) ✅ COMPLETED
- Запустить N независимых популяций в одном batch
- Измерить время для N = 3, 5, 10, 15
- Измерить успешность (должна быть ~100% при N≥3)
- **Результат: N=10, 50 runs → 100% success, ~10 sec/run**
- Файлы: `test_multi_de_scaling.py`, `test_multi_de_reliability.py`

### Шаг 2: Паттерная инициализация
- Вместо zeros для всех - разные стартовые гипотезы:
  - zeros + noise (нейтраль)
  - линейный тренд +2° (наклон вниз)
  - линейный тренд -2° (наклон вверх)
  - выпуклая дуга (прогиб вниз)
  - вогнутая дуга (прогиб вверх)
- Как фильтры в CNN - покрывают разные формы траектории
- Файл: `test_pattern_init.py`

### Шаг 3: Monte Carlo инициализация
- Много маленьких вбросов по пространству параметров
- Быстрая проверка разных регионов
- Файл: `test_monte_carlo_init.py`

### Шаг 4: Последовательная интерпретация всей скважины
- Протестировать на полной скважине (не только 4 сегмента)
- Проверить масштабируемость
- Файл: `test_full_well.py`

## Запуск из WSL

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm
cd /mnt/e/Projects/Rogii/gpu_ag
python <test_script>.py
```

## Существующие скрипты

| Файл | Описание |
|------|----------|
| `test_stability.py` | Тест 20 запусков LHS vs zeros init |
| `test_gpu_de.py` | Базовый тест GPU DE |
| `test_cpu_vs_gpu_eval.py` | Сравнение CPU и GPU objective function |
| `test_full_cpu_de.py` | CPU scipy DE для сравнения |
| `gpu_optimizer_fit.py` | Drop-in replacement для CPU optimizer_fit |
| `torch_funcs/gpu_optimizer.py` | GPU DE implementation |
| `torch_funcs/batch_objective.py` | Batch objective function |

## Data Sources

```
/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de/
├── AG_DATA/InitialData/slicing_well.json   # Well + TypeWell данные
└── interpretation.json                      # Сегменты интерпретации
```

## Ключевые параметры DE (scipy defaults - работают!)

```python
mutation = (0.5, 1.0)   # dithered F
recombination = 0.7     # CR
popsize = 500
maxiter = 500
```

## Reference из RND-777

### Результаты теста стабильности (20 запусков)

| Init Method | Success | Mean fun | Time/run |
|-------------|---------|----------|----------|
| LHS (500x500) | 4/20 (20%) | ~0.35 | ~2.1s |
| **Zeros+noise** | **19/20 (95%)** | ~0.18 | ~2.1s |
| LHS (1000x1000) | 10/20 (50%) | ~0.27 | ~8.4s |

### Zeros+noise init код
```python
center = torch.zeros(D, device=device, dtype=dtype)
noise_scale = bound_range * 0.01  # 1% от диапазона
population = center + torch.randn(popsize, D) * noise_scale
population = torch.clamp(population, lb, ub)
```

### Математика multi-population
При 95% успеха одного запуска:
- 2 популяции: 1 - 0.05² = 99.75%
- 3 популяции: 1 - 0.05³ = 99.99%
- 5 популяций: 1 - 0.05⁵ = 99.99997%

## Development Rules

1. **cpu_baseline/** - DO NOT TOUCH, reference
2. Все тесты в корне gpu_ag/
3. Comments in code: English only
4. Результаты записывать в worksheet.md
