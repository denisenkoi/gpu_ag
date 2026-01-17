# BEAM Optimization Research - Agent Instructions

---

## Jira

- Project: RND (area=rogii)
- Epic: RND-770 (Python AG Integration)

---

## ТЕКУЩАЯ ЗАДАЧА: Исследование BEAM алгоритмов

### Проблема

LOOKBACK_BEAM_V2 показывает лучший RMSE (6.25m vs 6.62m GREEDY), но:
- На некоторых скважинах результаты хуже baseline
- Только 53/100 скважин улучшены (53%)
- Нужно понять причины и улучшить

### Текущие результаты

| Алгоритм | RMSE | Wells improved | Время |
|----------|------|----------------|-------|
| Baseline | 6.93m | - | - |
| **LOOKBACK_BEAM_V2 (incr MSE)** | **6.01m** | 56/100 | ~120s |
| LOOKBACK_BEAM_V2 (old) | 6.25m | 53/100 | 537s |
| GREEDY_BF | 6.62m | 54/100 | 507s |
| BF best_std | 5.37m | 86/100 | ~6400s |

### Тестовые скважины (18 шт)

Сравнение: BF run `20260116_010959` (5.37m) vs BEAM run `20260117_084032` (6.25m)

**TOP по ухудшению (BEAM vs BF):**
| Well | BEAM | BF | diff |
|------|------|-----|------|
| Well1615~EGFDL | -13.54m | -1.07m | +12.47m |
| Well498~EGFDL | +12.22m | +0.20m | +12.02m |
| Well1153~EGFDL | +13.74m | -2.15m | +11.59m |
| Well1093~ASTNL | -12.09m | -1.34m | +10.75m |
| Well460~EGFDL | +11.85m | -4.45m | +7.41m |
| Well131~EGFDL | -18.12m | -12.00m | +6.12m |
| Well1898~EGFDU | -24.13m | -19.84m | +4.29m |
| Well1675~EGFDL | +22.31m | +21.86m | +0.45m |

**Рандомные (10 шт):**
- Well115~EGFDL, Well156~EGFDL, Well166~EGFDL, Well916~EGFDL
- Well1143~ASTNL, Well1235~EGFDL, Well1333~ASTNL, Well1509~EGFDL
- Well1613~EGFDL, Well1890~EGFDL

**Полный список для --wells:**
```
Well1093~ASTNL Well1143~ASTNL Well1153~EGFDL Well115~EGFDL Well1235~EGFDL Well131~EGFDL Well1333~ASTNL Well1509~EGFDL Well156~EGFDL Well1613~EGFDL Well1615~EGFDL Well166~EGFDL Well1675~EGFDL Well1890~EGFDL Well1898~EGFDU Well460~EGFDL Well498~EGFDL Well916~EGFDL
```

---

### Чекпоинты

- [ ] **1. Анализ "плохих" скважин**
  - Визуализировать скважины с большим расхождением
  - Понять паттерны ошибок
  - Сравнить с результатами BruteForce

- [ ] **2. Параметры BEAM**
  - Исследовать влияние beam_width (100 vs 200 vs 500)
  - Исследовать lookback distances (pearson_lookback, std_lookback)
  - Исследовать score_ratio (balance между score и STD селекцией)

- [ ] **3. Накопление ошибок**
  - Проанализировать как ошибки накапливаются на длинных скважинах
  - Возможно нужна стратегия "перезапуска" beam при плохих scores

- [ ] **4. Гибридный подход**
  - Попробовать BEAM для начала + BF для финала
  - Или GREEDY + BEAM комбинацию

---

## КРИТИЧЕСКИЕ ПРАВИЛА

0. **НЕ УБИВАТЬ ПРОЦЕССЫ БЕЗ СЛОВА "УБИВАЙ" ОТ ПОЛЬЗОВАТЕЛЯ!**
1. **TVT = TVD - shift** (НЕ TVD + shift!)
2. **После компактизации — читать этот файл и worksheet.md**
3. **GPU: проверять nvidia-smi перед запуском**
4. **НИКОГДА НЕ МЕНЯТЬ angle_step в процессе экспериментов!**
   - Стандарт BruteForce: **0.2** для обычных сегментов
   - Для длинных сегментов: **0.1** (автоматически)
   - Изменение шага ТОЛЬКО после долгого обсуждения с пользователем
   - В экспериментах ЗАПРЕЩЕНО менять шаг для "оптимизации"

---

## ЗАПУСК

**ВАЖНО:** LOOKBACK_BEAM/V2 обрабатывает ВСЮ скважину за один вызов (не по блокам).
Параметр `--block-overlap` для него **НЕ ИСПОЛЬЗУЕТСЯ**.

```bash
cd /mnt/e/Projects/Rogii/gpu_ag_2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vllm

# Тест одной скважины:
python full_well_optimizer.py --well Well239~EGFDL \
  --algorithm LOOKBACK_BEAM_V2 --angle-range 2.5 --mse-weight 5

# 18 тестовых скважин на 3090:
CUDA_VISIBLE_DEVICES=1 python full_well_optimizer.py \
  --wells Well1093~ASTNL Well1143~ASTNL Well1153~EGFDL Well115~EGFDL \
          Well1235~EGFDL Well131~EGFDL Well1333~ASTNL Well1509~EGFDL \
          Well156~EGFDL Well1613~EGFDL Well1615~EGFDL Well166~EGFDL \
          Well1675~EGFDL Well1890~EGFDL Well1898~EGFDU Well460~EGFDL \
          Well498~EGFDL Well916~EGFDL \
  --algorithm LOOKBACK_BEAM_V2 --angle-range 2.5 --mse-weight 5 \
  --description "BEAM V2: 18 test wells"

# С изменёнными параметрами beam:
LOOKBACK_BEAM_WIDTH=200 LOOKBACK_STAGE_SIZE=2 python full_well_optimizer.py ...
```

---

## Ключевые файлы

| Файл | Назначение |
|------|------------|
| `full_well_optimizer.py` | Главный скрипт запуска |
| `optimizers/lookback_beam_v2.py` | Векторизованный BEAM |
| `optimizers/greedy_bruteforce.py` | GREEDY_BF |
| `optimizers/objective.py` | Метрики (compute_loss_batch, BeamPrefix) |
| `interpretation_visualizer.py` | Визуализация REF vs OPT |

---

## Гипотезы для исследования

1. **Beam width слишком маленький** - 100 кандидатов может быть мало для длинных скважин
2. **Lookback слишком короткий** - не видим ошибки накопившиеся раньше
3. **Score != качество** - высокий Pearson на блоке не гарантирует хороший финальный результат
4. **Нужна end-to-end метрика** - оценка по всей скважине, не по блокам

---

## [2026-01-17 23:40] ТЕКУЩИЙ АНАЛИЗ: Почему BEAM теряет правильное решение

### Ключевая находка (Well498, dls mode)

BF с разным числом сегментов на блок даёт РАЗНЫЕ результаты:

| Сегментов на блок | Топ-1 (победитель) | REF[5seg] ранг | Финал RMSE |
|-------------------|-------------------|----------------|------------|
| 3 | [2.31,2.11,1.11] | **32/100** из 17,576 | **+12.04m** |
| 4 | [2.31,1.31,2.71,3.31] | **12/100** из 456,976 | **+8.89m** |
| 5 | [2.31,2.11,1.51,3.91,-0.09] | **1/100** из 11.8M | **+0.27m** |

**REF[5seg]** = решение-победитель с 5 сегментами. На 3 сегментах оно на 32 месте, на 4 - на 12 месте, на 5 - на 1 месте.

### Проблема BEAM

- BEAM начинает с `first_block_size=4` сегментов
- На Stage 0 BEAM выбирает [2.31,1.31,2.71,3.31] (то же что BF на 4 сегментах)
- Правильное решение [2.31,2.11,1.51,3.91] было на **12 месте** в топ-100
- BEAM его теряет на следующих стадиях → финал **+12.21m** вместо +0.27m

### Что делали

1. Добавили `--segments-per-block` параметр в full_well_optimizer.py
2. Добавили логирование топ-10 и поиск REF[5seg] в bruteforce.py
3. Добавили поиск REF[5seg] в lookback_beam_v2.py (в процессе)

### ВЫПОЛНЕНО

1. ✓ Добавлен поиск REF[5seg] в BEAM
2. ✓ Запущен BEAM с трекингом - REF[5seg] на 7 месте Stage 0, LOST на Stage 1
3. ✓ Проверено beam_width=500 - не помогает
4. ✓ STD почти одинаковый (~10.33-10.34) - не различает

### Следующие шаги

1. **Diverse beam** - реализовать сохранение разнообразия по углам/STD
2. **End-to-end score** - оценка по всей траектории на каждой стадии
3. **Chunked first_block** - для first_block_size=5 нужен chunked processing

### НОВЫЕ РЕЗУЛЬТАТЫ (2026-01-17 23:45)

**BF результаты по сегментам (angle_step=0.2, 26 вариантов на сегмент):**

| Сегментов | Всего комбинаций | REF[5seg] ранг | Финал |
|-----------|------------------|----------------|-------|
| 3 | 17,576 (26³) | 32/17576 | +12.04m |
| 4 | 456,976 (26⁴) | 12/456976 | +8.89m |
| 5 | 11,881,376 (26⁵) | **1/11881376** | **+0.27m** |

**BEAM трекинг REF[5seg]:**

| Стадия | Всего комбинаций | REF ранг | REF score | Top-1 score |
|--------|------------------|----------|-----------|-------------|
| Stage 0 (4 seg) | 456,976 | 7/456976 | -1.712 | -1.627 |
| Stage 1 (+2 seg) | 67,600 | **62266/67600** | **-41.242** | -2.758 |

**КРИТИЧЕСКИЙ ВЫВОД:** REF продолжение [-0.09,-0.29] имеет score=-41.2 (на 62266 месте из 67600).
Это в **15 раз хуже** чем Top-1. Отрицательные углы катастрофически ухудшают локальную метрику.
Lookback=300 точек не видит общую картину - BF с 5 сегментами за раз видит и выбирает REF как топ-1.

### Изменённые файлы

- `full_well_optimizer.py` - добавлен --segments-per-block
- `optimizers/bruteforce.py` - добавлен топ-10 лог и поиск REF[5seg]
- `optimizers/lookback_beam_v2.py` - добавлен поиск REF[5seg] на Stage 0
