# GPU Autogeosteering - Agent Instructions

**ВНИМАНИЕ:** Основная инструкция в `/mnt/e/Projects/Rogii/sc/docs/AGENT_INSTRUCTIONS.md`!
Эта копия для справки.

---

## GRID SEARCH ПЛАН (RND-806)

| # | pearson | mse | angle | Статус |
|---|---------|-----|-------|--------|
| 1 | 1.0 | 1.0 | 1.0 | BASELINE |
| 2 | 1.0 | 1.5 | 1.0 | TODO |
| 3 | 1.0 | 2.0 | 1.0 | TODO |
| 4 | 1.0 | 3.0 | 1.0 | TODO |
| 5 | 1.0 | 4.0 | 1.0 | TODO |
| 6 | 1.0 | 1.0 | 2.0 | TODO |
| 7 | 1.0 | 1.0 | 4.0 | TODO |

## Запуск

```bash
cd /mnt/e/Projects/Rogii/gpu_ag/cpu_baseline
export PYTHON_PEARSON_POWER=1.0
export PYTHON_MSE_POWER=1.0
export PYTHON_ANGLE_SUM_POWER=1.0
python slicer.py --de --max-iterations 0
```
