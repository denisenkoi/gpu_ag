# GPU AG - Worksheet

## Текущие задачи

- RND-792: Сглаживание GR
- RND-790: Улучшение функции награды (комментарий добавлен)

---

## [2025-12-23 12:45] Начало нового блока работ

### Следующие шаги

1. **Сглаживание GR (RND-792)**
   - Гипотеза: гладкая GR → гладкий ландшафт → стабильнее оптимизация

2. **Другие методы оптимизации**
   - scipy.minimize (L-BFGS-B, Powell)
   - optuna
   - Комбинации алгоритмов

3. **Улучшение objective function**
   - Проблема: objective не коррелирует с RMSE (отрицательная корреляция!)

---

## Заметки

---

## [2025-12-23 14:26] Тест 243 vs 867 популяций

**Исправлено:**
1. Добавлен `load_dotenv()` в slicer.py (не загружался .env)
2. Добавлены GPU параметры в cpu_baseline/.env
3. Исправлено логирование: fun + evals вместо correlation

**Сравнение (Well1798~EGFDL, 1 итерация):**

| Метрика | 867 pop (range=10°) | 243 pop (range=5°) |
|---------|---------------------|---------------------|
| RMSE | 0.227м | 0.226м |
| Max dev | 0.715м | 0.714м |
| Total evals | 346,800 | 194,400 |
| Время | ~16 мин | ~2 мин |

**Вывод:** Качество идентичное при range=5° vs range=10°. Скорость в 8 раз выше.

---

## [2025-12-23 15:20] Расширенное логирование CSV

**Добавлено:**
1. `eval_count` в `TorchObjectiveWrapper` - фактическое количество evaluations
2. Детальное логирование в `_log_optimization_result`:
   - MD range (start_md - end_md)
   - fun (objective)
   - pearson, mse
   - angle_penalty, angle_sum_penalty
   - angles (список углов сегментов)
   - actual_evals / expected_evals

**CSV формат:** `results_wells_grid_15/optimization_statistics.csv`
```
timestamp,well_name,step_type,start_md,end_md,segments_count,fun,pearson,mse,angle_penalty,angle_sum_penalty,angles,actual_evals,expected_evals,elapsed_time,success
```

**Тест полной скважины:** 22/~45 итераций выполнено, затем StarSteer crash (FileNotFoundError - не связано с логированием).

**Результат:** CSV формат работает корректно, все метрики записываются.
