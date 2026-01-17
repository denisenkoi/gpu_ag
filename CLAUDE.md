# GPU AG Worktree - BEAM Optimization Research

**Это worktree** от основного репозитория `/mnt/e/Projects/Rogii/gpu_ag/`
**Ветка:** `beam_ag_optimization`
**Цель:** Исследование и улучшение BEAM алгоритмов (LOOKBACK_BEAM, GREEDY_BF)

---

## Рабочие директории

**Код:** `/mnt/e/Projects/Rogii/gpu_ag_2/` (ЭТОТ worktree, работаем напрямую здесь)
**Docs:** `/mnt/e/Projects/Rogii/gpu_ag_2/docs/` (локальные для этого worktree)
**Триггеры:** `/mnt/e/Projects/Rogii/sc/task_queue/` (если нужны BAT)
**Jira:** RND project, Epic RND-770

---

## Python + PyTorch - ВСЕГДА vllm env!

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vllm
cd /mnt/e/Projects/Rogii/gpu_ag_2
```

**НЕ ИСПОЛЬЗОВАТЬ:** base, drilling_emulator, anaconda3, или системный python!

---

## ТЕКУЩИЙ ЛУЧШИЙ РЕЗУЛЬТАТ (без утечки данных)

| Алгоритм | RMSE | Улучшение | Время | Wells improved |
|----------|------|-----------|-------|----------------|
| **LOOKBACK_BEAM_V2** | **6.25m** | **9.8%** | 537s | 53/100 |
| GREEDY_BF | 6.62m | 4.5% | 507s | 54/100 |
| BF best_std | 6.70m | 3.3% | 6431s | 51/100 |

Baseline RMSE: 6.93m

---

## Основные алгоритмы

| Алгоритм | Файл | Описание |
|----------|------|----------|
| BruteForce | `optimizers/bruteforce.py` | Перебор всех комбинаций в блоке |
| GREEDY_BF | `optimizers/greedy_bruteforce.py` | Иерархический beam по 2-3 сегмента |
| LOOKBACK_BEAM | `optimizers/lookback_beam.py` | Beam search с lookback (V1, медленный) |
| LOOKBACK_BEAM_V2 | `optimizers/lookback_beam_v2.py` | Векторизованный beam (14x быстрее) |

---

## Запуск экспериментов

```bash
cd /mnt/e/Projects/Rogii/gpu_ag_2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vllm

# Одна скважина (LOOKBACK_BEAM_V2):
python full_well_optimizer.py --well Well239~EGFDL \
  --algorithm LOOKBACK_BEAM_V2 --angle-range 2.5 --mse-weight 5 --block-overlap 1

# Все 100 скважин:
python full_well_optimizer.py --all --algorithm LOOKBACK_BEAM_V2 \
  --angle-range 2.5 --mse-weight 5 --block-overlap 1 \
  --description "BEAM V2 experiment"

# На конкретной GPU:
CUDA_VISIBLE_DEVICES=1 python full_well_optimizer.py ...
```

---

## Визуализация

```bash
python interpretation_visualizer.py --well Well239~EGFDL --run-id <RUN_ID> \
  --output results/viz/well239.png
```

---

## PostgreSQL логирование

БД `gpu_ag` на localhost (user: rogii, pass: rogii123)
Мониторинг: `python status.py`

---

## КРИТИЧЕСКИЕ ПРАВИЛА

1. **TVT = TVD - shift** (НЕ TVD + shift!)
2. **tvd_shift ВСЕГДА применяется к TypeLog!**
3. **После компактизации — читать docs/AGENT_INSTRUCTIONS.md и docs/worksheet.md**
4. **GPU: 5090 может быть занята, проверять через nvidia-smi**
5. **НЕ МЕНЯТЬ ЛОГИКУ без явного подтверждения пользователя!**
6. **Exit code 1 НЕ ОЗНАЧАЕТ падение!** Долгие процессы (BF, BEAM) могут работать минуты.
   - ВСЕГДА проверять БД: `SELECT run_id, finished_at, optimized_rmse FROM runs ORDER BY id DESC LIMIT 5`
   - Или CSV файл результатов
   - НЕ делать выводы по exit code без проверки реального статуса!

---

## Ключевые структуры данных

```python
@dataclass
class BeamPrefix:
    """Накопленное состояние для beam search."""
    synthetic: torch.Tensor    # накопленный synthetic GR
    zone_gr: torch.Tensor      # накопленный zone GR
    zone_gr_smooth: torch.Tensor  # сглаженный GR
    tvt: torch.Tensor          # накопленный TVT
    end_shift: float           # накопленный сдвиг

@dataclass
class BeamCandidate:
    """Один кандидат в beam."""
    prefix: BeamPrefix
    angles: np.ndarray
    score: float
    std: float
```

---

## Метрики

| Метрика | Описание |
|---------|----------|
| **RMSE all points** | По всем точкам всех скважин (интерполяция к 1м) |
| **RMSE endpoint** | По последним точкам каждой скважины |
| **Pearson** | Корреляция synthetic GR с well GR |
| **MSE** | Ошибка между synthetic и well GR |
| **STD(bin_means)** | Self-correlation качество (меньше = лучше) |
