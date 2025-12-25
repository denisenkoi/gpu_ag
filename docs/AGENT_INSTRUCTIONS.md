# GPU Autogeosteering - Agent Instructions

## Jira

- Project: RND (area=rogii)
- Epic: RND-770 (Python AG Integration)
- Текущие задачи:
  - RND-790: Improve reward function (комментарий с результатами добавлен)
  - RND-792: Research gamma smoothing effect
  - RND-794: Add self-correlation to reward function
  - **RND-800: StarSteer remaining_md_meters mismatch** (АКТИВНЫЙ БАГ)

---

## ✅ RND-800 - АНАЛИЗ ЗАВЕРШЁН

### Чекпоинты исследования

- [x] Найти где вычисляется remaining_md_meters
- [x] Понять почему can_add_more=True если логи закончились
- [x] Проверить почему TrajectoryUpdated не триггерит AG
- [x] Проверить поведение при малом шаге слайса
- [x] Понять расхождение max_source_md vs конец траектории
- [x] Проверить когда создаётся slicing_well.json (DLL)
- [x] Проверить подписку AG на events от slicing_well
- [ ] **Реализовать fix** (ждём решения)

### Root Cause: ДВЕ проблемы!

**Проблема 1:** `copyLogRange` делает early return без LogUpdated (если нет данных)

**Проблема 2:** AG ИГНОРИРУЕТ события от slicing_well!
- Event генерируется для TARGET well (slicing_well) с НОВЫМ UUID
- AG подписан на SOURCE well (исходная скважина)
- `isRelatedObject(slicing_well_UUID)` → FALSE

### Ключевые находки

| Вопрос | Ответ |
|--------|-------|
| TrajectoryUpdated → slicing_well.json? | ❌ НЕТ! JSON только при AG START |
| AG подписан на slicing_well? | ❌ НЕТ! Только на source well |
| Малый шаг (1м) безопасен? | ❌ НЕТ! Event не сгенерируется |
| max_source_md откуда? | max(trajectory, все логи) - TGAS=17200ft |

### Цепочка данных Well850
```
GR лог:      17018 ft (закончился)
Траектория:  17073 ft
TGAS лог:    17200 ft ← max_source_md берётся отсюда
```

### Рекомендуемое решение
**Вариант 3:** После каждого slice явно экспортировать slicing_well.json.

Решает ВСЕ проблемы:
- Slice за пределами логов
- Малый шаг слайса
- Независимость от LogUpdated/TrajectoryUpdated

### Документация
- Jira: RND-800
- Worksheet: [2025-12-25 01:35] и [2025-12-25 02:00]

---

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

### 3. Улучшение функции награды (RND-790)
- [ ] Добавить накопительный RMSE в objective
- [ ] Увеличить вес angle_sum_penalty
- [ ] Дисперсия GR как модулятор штрафа
- [ ] Grid penalty

### 4. Self-correlation в reward function (RND-794)
- [ ] Добавить self-correlation компонент в objective
- [ ] Протестировать влияние на качество интерпретации

### 5. Настройки оптимизатора (DONE)
- [x] Single CMA-ES mode (230x speedup)
- [x] PYTHON_LOOKBACK_DISTANCE=200м (лучше чем 50м)
- [x] PYTHON_ANGLE_RANGE=10° (запас для сложных случаев)
- [x] Race condition fix в _wait_for_fresh_data_file()

### 6. Тест по всем скважинам
- [ ] Запустить slicer_gpu.py по всем 103 скважинам
- [ ] Мониторить с будильником (scheduler_wakeup_tool)
- [ ] При первом падении НЕ перезапускать - обсудить с пользователем
- [ ] Собрать статистику качества по всем скважинам

## Запуск тестов

**КРИТИЧНО:** Запускать НАПРЯМУЮ через bash в background, НЕ через task_runner!

```bash
# Запуск slicer в background (ПРАВИЛЬНЫЙ СПОСОБ)
cd /mnt/e/Projects/Rogii/gpu_ag && \
source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate vllm && \
nohup python cpu_baseline/slicer.py --de \
  --starsteer-dir "/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de" \
  --max-iterations 0 > slicer_all_wells.log 2>&1 &
echo "Started, PID: $!"
```

**Мониторинг:**
```bash
tail -f /mnt/e/Projects/Rogii/gpu_ag/slicer_all_wells.log
```

**ВАЖНО:**
- `--max-iterations 0` = неограниченно (default=1!)
- StarSteer запускается АВТОМАТИЧЕСКИ slicer'ом, НЕ нужно запускать отдельно!

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
