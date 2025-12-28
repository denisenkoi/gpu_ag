# GPU Autogeosteering - Agent Instructions

## Jira

- Project: RND (area=rogii)
- Epic: RND-770 (Python AG Integration)
- Текущие задачи:
  - **RND-805**: Bug - 11 скважин с RMSE 100+ метров

---

## ТЕКУЩАЯ ЗАДАЧА: Ночное тестирование стабильности (RND-805)

### Цель

Тестировать CMA-ES на 11 проблемных скважинах:
- Качество (RMSE)
- Воспроизводимость (разброс между прогонами)
- Стабильность (нет падений)

### ⚠️ ПРОБЛЕМА PSO

PSO падает с `ValueError: operands could not be broadcast together with shapes (0,) (50,4)`
когда ВСЕ частицы получают inf (projection failure). **Пока тестируем только CMA-ES.**

---

## 11 проблемных скважин

| # | Скважина | RMSE (baseline) |
|---|----------|-----------------|
| 1 | Well1509~EGFDL | 344.2m |
| 2 | Well1466~EGFDL | 275.6m |
| 3 | Well1150~EGFDL | 267.2m |
| 4 | Well1401~EGFDL | 265.7m |
| 5 | Well1736~EGFDL | 223.8m |
| 6 | Well1300~EGFDL | 168.6m |
| 7 | Well1322~EGFDL | 166.6m |
| 8 | Well809~EGFDL | 157.0m |
| 9 | Well530~EGFDL | 156.5m |
| 10 | Well1615~EGFDL | 129.8m |
| 11 | Well1258~EGFDL | 107.5m |

---

## План тестов

### Конфигурации CMA-ES для тестирования

| # | Config | min_pearson | angle_penalty | restarts | popsize |
|---|--------|-------------|---------------|----------|---------|
| 1 | soft_mp03 | 0.3 | exp (1M×100^x) | 5 | 50 |
| 2 | soft_mp03_r10 | 0.3 | exp | 10 | 50 |
| 3 | soft_mp01 | 0.1 | exp | 5 | 50 |

### Прогонов на конфигурацию: 5

### Метрики для сбора

- **Avg RMSE** - средний RMSE по всем слайсам
- **Max RMSE** - максимальный RMSE (выброс)
- **Delta** - отклонение в конечной точке
- **Coverage 3m** - % точек с RMSE < 3m
- **Crashes** - количество падений

---

## Команды

### Подготовка wells_config.json

```bash
# Сбросить все 11 скважин
cd /mnt/e/Projects/Rogii/gpu_ag
python -c "
import json
wells = [
    'Well1509~EGFDL', 'Well1466~EGFDL', 'Well1150~EGFDL', 'Well1401~EGFDL',
    'Well1736~EGFDL', 'Well1300~EGFDL', 'Well1322~EGFDL', 'Well809~EGFDL',
    'Well530~EGFDL', 'Well1615~EGFDL', 'Well1258~EGFDL'
]
config = {'wells': [
    {'well_name': w, 'lateral_log_name': 'GR', 'typewell_name': 'Well2004',
     'typewell_log_name': 'GR', 'horizon_name': 'EGFDL', 'grid_name': '',
     'enable_log_normalization': True, 'process': True, 'processed': False}
    for w in wells
]}
with open('/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de/wells_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print('Created config for', len(wells), 'wells')
"
```

### Запуск CMA-ES теста

```bash
cd /mnt/e/Projects/Rogii/gpu_ag && \
source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate vllm && \
PYTHON_MIN_PEARSON_VALUE=0.3 \
GPU_ALGORITHM=SNES \
python cpu_baseline/slicer.py --de \
  --starsteer-dir "/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de" \
  --max-iterations 0 2>&1 | tee slicer_cmaes_run1.log
```

---

## После каждого прогона

1. Извлечь метрики:
```bash
# Avg RMSE
grep "RMSE=" slicer_cmaes_run*.log | grep -oP "RMSE=\K[0-9.]+" | awk '{sum+=$1; count++} END {print "Avg RMSE:", sum/count}'

# Max RMSE
grep "RMSE=" slicer_cmaes_run*.log | grep -oP "RMSE=\K[0-9.]+" | sort -n | tail -1

# Crashes
grep -c "ERROR\|exception\|ValueError" slicer_cmaes_run*.log
```

2. Записать в worksheet.md
3. Сбросить processed и запустить следующий прогон

---

## КРИТИЧЕСКИЕ ПРАВИЛА

1. **Будильники**: ВСЕГДА ставить после запуска теста (project="SSAndAG")
2. **Результаты записывать в worksheet.md**
3. **После компактизации** - ЧИТАТЬ этот файл и worksheet.md!
4. **5 прогонов** каждой конфигурации для статистики

---

## Экспоненциальный penalty (текущий)

Формула: `penalty = 1_000_000 × 100^(max_excess)` где `max_excess = max(0, |angle| - range)`

Изменённые файлы:
- `torch_funcs/batch_objective.py`
- `torch_funcs/objective.py`
- `numpy_funcs/objective.py`

---

## НОВАЯ ЗАДАЧА: Grid Search параметров reward функции (RND-806)

### Цель

Подобрать оптимальные параметры reward функции для минимизации RMSE.

### Текущие параметры

| Параметр | Переменная | Текущее значение |
|----------|------------|------------------|
| Pearson power | `PYTHON_PEARSON_POWER` | 2.0 |
| MSE power | `PYTHON_MSE_POWER` | 0.001 |
| Angle sum power | (hardcoded) | 2.0 |
| Min Pearson | `PYTHON_MIN_PEARSON_VALUE` | 0.3 |

### Выборка скважин (25 штук)

| Категория | Кол-во | RMSE baseline | Скважины |
|-----------|--------|---------------|----------|
| WORST (>=100m) | 7 | 121-208m | Well1466, Well1509, Well1736, Well1401, Well1300, Well530, Well1150 |
| BAD (50-100m) | 5 | 66-83m | Well1258, Well531, Well1908, Well1615, Well522 |
| MEDIUM (10-50m) | 5 | 12-40m | Well973, Well1682, Well32, Well1221, Well1416 (random) |
| WEAK (3-10m) | 4 | 3.6-8.7m | Well1313, Well1810, Well772, Well124 (random) |
| GOOD (<3m) | 4 | 0.2-2.8m | Well189, Well1703, Well1613, Well460 (random) |

**Распределение:** 12 плохих (48%), 13 рандомных из средних/хороших (52%)

### Список скважин для конфига

```python
wells = [
    'Well1466~EGFDL', 'Well1509~EGFDL', 'Well1736~EGFDL', 'Well1401~EGFDL', 'Well1300~EGFDL',
    'Well530~EGFDL', 'Well1150~EGFDL', 'Well1258~EGFDL', 'Well531~EGFDL', 'Well1908~EGFDU',
    'Well1615~EGFDL', 'Well522~EGFDL', 'Well973~EGFDU', 'Well1682~EGFDL', 'Well32~EGFDL',
    'Well1221~EGFDL', 'Well1416~EGFDU', 'Well1313~EGFDL', 'Well1810~EGFDL', 'Well772~EGFDL',
    'Well124~EGFDU', 'Well189~EGFDL', 'Well1703~EGFDL', 'Well1613~EGFDL', 'Well460~EGFDL'
]
```

### План grid search

| # | pearson_power | mse_power | angle_sum_power | Примечание |
|---|---------------|-----------|-----------------|------------|
| 1 | 2.0 | 0.001 | 2.0 | baseline |
| 2 | 1.0 | 0.001 | 2.0 | слабее pearson |
| 3 | 3.0 | 0.001 | 2.0 | сильнее pearson |
| 4 | 2.0 | 0.01 | 2.0 | сильнее mse |
| 5 | 2.0 | 0.0001 | 2.0 | слабее mse |
| 6 | 2.0 | 0.001 | 1.0 | слабее angle_sum |
| 7 | 2.0 | 0.001 | 3.0 | сильнее angle_sum |
