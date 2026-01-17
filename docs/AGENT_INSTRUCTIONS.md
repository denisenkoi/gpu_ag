# BEAM Optimization Research - Agent Instructions

---

## Jira

- Project: RND (area=rogii)
- Epic: RND-770 (Python AG Integration)

---

## КРИТИЧЕСКИЕ ПРАВИЛА

0. **НЕ УБИВАТЬ ПРОЦЕССЫ БЕЗ СЛОВА "УБИВАЙ" ОТ ПОЛЬЗОВАТЕЛЯ!**
1. **TVT = TVD - shift** (НЕ TVD + shift!)
2. **После компактизации — читать этот файл и worksheet.md**
3. **GPU: проверять nvidia-smi перед запуском**
4. **НИКОГДА НЕ МЕНЯТЬ angle_step в процессе экспериментов!**
   - Стандарт BruteForce: **0.2** для обычных сегментов
   - Для длинных сегментов: **0.1** (автоматически)
5. **ЧЕСТНОЕ СРАВНЕНИЕ:** BEAM и BF должны использовать одинаковый LANDING_MODE!
   - DLS mode (по умолчанию) - для сравнения с текущим BEAM
   - 87_200 mode - для сравнения со старым BF (5.37m)

---

## ТЕКУЩАЯ ЗАДАЧА: Скрупулёзный анализ скважин

### Чекпоинт

- [ ] **1. Изучить Well1093~ASTNL до конца**
  - Найти где BF решение [-3.10, -1.60, -1.60, -1.60] в рейтинге BEAM Stage 0
  - Понять почему локальный score предпочитает крутые углы
  - Попробовать увеличить lookback
  - Попробовать diverse beam
  - Документировать все находки

- [ ] **2. Создать инструментарий для анализа скважин**
  - Скрипт для сравнения BEAM vs BF посегментно
  - Поиск конкретного решения в рейтинге BEAM
  - Визуализация divergence точки

- [ ] **3. Систематический анализ проблемных скважин**
  - Well1781~EGFDL (+8.50m)
  - Well42~EGFDL (+8.47m)
  - Well460~EGFDL (+6.09m)
  - Найти общие паттерны

- [ ] **4. Найти "систематически сложные" скважины**
  - Скважины где ВСЕ алгоритмы ошибаются
  - Проверить историю в БД

---

## Текущие результаты

| Алгоритм | RMSE | Wells better | Примечание |
|----------|------|--------------|------------|
| **BEAM V2 (incr MSE)** | **6.01m** | 34/100 | **ЛИДЕР при DLS mode** |
| BF (DLS mode) | 6.45m | 25/100 | Честное сравнение |
| BF (87_200 mode) | 5.37m | - | Другой landing mode! |

---

## Ключевые файлы

| Файл | Назначение |
|------|------------|
| `full_well_optimizer.py` | Главный скрипт запуска |
| `optimizers/lookback_beam_v2.py` | Векторизованный BEAM |
| `optimizers/objective.py` | Метрики (compute_loss_batch, BeamPrefix) |

---

## Архив предыдущей фазы

- `docs/worksheet_20260118_beam_improvement.md`
- `docs/AGENT_INSTRUCTIONS_20260118_beam_improvement.md`

Достижения:
- Инкрементальный MSE: Well498 +12.22m → +0.02m
- BEAM обогнал BF при честном сравнении (6.01m vs 6.45m)

---

## ЗАПУСК

```bash
cd /mnt/e/Projects/Rogii/gpu_ag_2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vllm

# Одна скважина:
python full_well_optimizer.py --well Well1093~ASTNL \
  --algorithm LOOKBACK_BEAM_V2 --angle-range 2.5 --mse-weight 5 \
  --description "debug Well1093"

# С debug логами:
# Смотреть вывод "DEBUG REF", "Top-5 by score"
```
