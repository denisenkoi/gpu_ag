# GPU Autogeosteering - Agent Instructions

## Jira

- Project: RND (area=rogii)
- Epic: RND-770 (Python AG Integration)
- Task: RND-790 (Improve reward function for GPU optimizer)

## Цель

Улучшение функции награды для повышения качества интерпретации GPU оптимизатора.

## Чекпоинты

### 1. Усовершенствование штрафов за углы
- [ ] Добавить штраф за угол между последним frozen и первым оптимизируемым сегментом
- [ ] Вычислять дисперсию GR вокруг угла (пол сегмента влево + пол сегмента вправо)
- [ ] Модулировать штраф дисперсией: низкая → больше штраф, высокая → меньше штраф
- [ ] Протестировать на полной скважине Well1798~EGFDL

### 2. Комбинации существующих параметров
- [ ] Эксперименты с MSE_POWER (1.0, 2.0, 0.5)
- [ ] Эксперименты с DIP_ANGLE_RANGE (2°, 5°, 10°)
- [ ] Эксперименты с ANGLE_SUM_POWER
- [ ] Найти оптимальную комбинацию

### 3. Детектор перегибов из C++
- [ ] Изучить логику детектора перегибов в AG daemon
- [ ] Портировать в Python/PyTorch
- [ ] Интегрировать в objective function

### 4. Использование гридов
- [ ] Изучить формат гридов в slicing_well.json
- [ ] Добавить grid penalty в objective

### 5. Обновление pseudoTypeLog (self-correlation)
- [ ] Реализовать корреляцию горизонтальной скважины на себя
- [ ] Динамически расширять pseudo данными из well
- [ ] Тестирование

## Ключевые файлы

| Файл | Описание |
|------|----------|
| `gpu_executor.py` | GPU executor с EvoTorch SNES |
| `torch_funcs/batch_objective.py` | Batch objective function |
| `torch_funcs/converters.py` | calc_segment_angles_torch |
| `.env` | Параметры оптимизации |

## Текущая формула objective

```python
objective = (1-pearson)² × mse^power × (1 + angle_penalty + angle_sum_penalty)
```

Где:
- `angle_penalty`: штраф за |angle| > DIP_ANGLE_RANGE
- `angle_sum_penalty`: сумма |angle_diffs| между соседними сегментами

## Запуск тестов

```bash
cd /mnt/e/Projects/Rogii/gpu_ag
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm
python slicer_gpu.py --de --starsteer-dir "/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de" --max-iterations 0
```

## КРИТИЧЕСКИЕ ПРАВИЛА

1. **Не писать самописные оптимизаторы** — только EvoTorch, scipy, optuna
2. **cpu_baseline/ — НЕ ТРОГАТЬ** без согласования
3. **Результаты в worksheet.md**
