# GPU AG - Worksheet (Grid Search RND-806)

## Текущий статус

**Тест:** Baseline (#1)
**Параметры:** pearson=2.0, mse=0.5, angle=2.0
**Статус:** TODO

---

## Результаты Grid Search

| # | Config | Avg RMSE | Max RMSE | Wells OK | Примечание |
|---|--------|----------|----------|----------|------------|
| 1 | baseline (p=2.0/m=0.5/a=2.0) | - | - | - | TODO |
| 2 | pearson=1.0 | - | - | - | TODO |
| 3 | pearson=3.0 | - | - | - | TODO |
| 4 | mse=0.1 | - | - | - | TODO |
| 5 | mse=1.0 | - | - | - | TODO |
| 6 | angle=1.0 | - | - | - | TODO |
| 7 | angle=3.0 | - | - | - | TODO |

---

## Заметки

[2025-12-28 13:30] Подготовка к Grid Search:
- Обновлена инструкция с MSE baseline = 0.5
- Все 3 параметра через env vars
- Единая функция награды
- Автоматический retry при timeout
- Суффикс параметров в названии интерпретации

---
