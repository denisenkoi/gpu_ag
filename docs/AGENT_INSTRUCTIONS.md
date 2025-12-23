# GPU Autogeosteering - Agent Instructions

## Jira

- Project: RND (area=rogii)
- Epic: RND-770 (Python AG Integration)
- Текущие задачи:
  - RND-790: Improve reward function (комментарий с результатами добавлен)
  - RND-792: Research gamma smoothing effect

## Цель

Улучшение функции награды и стабильности оптимизации GPU optimizer.

## Ключевые находки (из RND-790)

**Тест воспроизводимости CMA-ES (4 Run на Well1798~EGFDL):**

| Run | Final RMSE | Sum Objective |
|-----|------------|---------------|
| 1 | 6.970м | 0.032 |
| 2 | 6.970м | 0.031 |
| 3 | 4.921м | 0.140 |
| 4 | 6.970м | 0.031 |

**КРИТИЧНО:** Корреляция objective↔RMSE **ОТРИЦАТЕЛЬНАЯ!**
- Лучший RMSE (4.92м) = Худший objective (0.140)
- Функция награды оптимизирует НЕ то, что нужно

## Чекпоинты

### 1. Сглаживание GR (RND-792)
- [ ] Добавить параметр сглаживания (moving average / Gaussian filter)
- [ ] Протестировать влияние на стабильность
- [ ] Сравнить RMSE с разными уровнями сглаживания
- [ ] Гипотеза: гладкая GR → гладкий ландшафт → лучший градиентный спуск

### 2. Другие методы оптимизации
- [ ] Попробовать scipy.minimize (L-BFGS-B, Powell)
- [ ] Попробовать optuna
- [ ] Комбинации: грубый поиск (CMA-ES) + локальная дооптимизация
- [ ] Цель: стабильность + скорость

### 3. Улучшение функции награды
- [ ] Добавить накопительный RMSE в objective
- [ ] Увеличить вес angle_sum_penalty
- [ ] Дисперсия GR как модулятор штрафа
- [ ] Grid penalty

## Запуск тестов

```bash
cd /mnt/e/Projects/Rogii/gpu_ag
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

# CMA-ES (текущий алгоритм)
GPU_ALGORITHM=CMAES GPU_ANGLE_GRID_STEP=1.25 PYTHON_ANGLE_RANGE=5.0 \
python slicer_gpu.py --de --starsteer-dir "/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de" --max-iterations 0
```

**ВАЖНО:** `--max-iterations 0` = неограниченно (default=1!)

## Ключевые файлы

| Файл | Описание |
|------|----------|
| `gpu_executor.py` | GPU executor с CMA-ES |
| `torch_funcs/batch_objective.py` | Batch objective function |
| `.env` | Параметры оптимизации |

## Текущая формула objective

```python
objective = (1-pearson)² × mse^power × (1 + angle_penalty + angle_sum_penalty)
```

## Метрики

- ~97,000 evaluations/шаг (243 populations × 8 × 50)
- ~1 мин на шаг (~45 шагов на полную скважину)
- Тестовая скважина: Well1798~EGFDL (MD: 3772 → 5079м)

## КРИТИЧЕСКИЕ ПРАВИЛА

1. **Запускаем ВСЕГДА по всей скважине:** `--max-iterations 0`
2. **НЕ МЕНЯТЬ КОД без явного подтверждения пользователя!**
3. **cpu_baseline/ — НЕ ТРОГАТЬ** без согласования
4. **Результаты записывать в worksheet.md**
