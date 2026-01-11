# Воспроизведение результатов

## Лучший результат: RMSE = 3.37m (11.1 ft)

**Параметры:**
- algorithm: BRUTEFORCE
- angle_range: 2.5
- mse_weight: 5

## Запуск

```bash
cd /mnt/e/Projects/Rogii/gpu_ag
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

# Все 100 скважин (~50 минут на 5090)
python full_well_optimizer.py --all --algorithm BRUTEFORCE --angle-range 2.5 --mse-weight 5

# Одна скважина
python full_well_optimizer.py --wells Well162~EGFDL --algorithm BRUTEFORCE --angle-range 2.5 --mse-weight 5
```

## Датасет

**Файл:** `dataset/gpu_ag_dataset.pt` (115MB, не в git)

**Создание датасета:**
```bash
cd /mnt/e/Projects/Rogii/gpu_ag/dataset
python json_to_torch.py
```

**Исходные данные:** JSON интерпретации из StarSteer (100 скважин)

## Результаты тестов mse_weight (2026-01-11)

| mse_weight | RMSE | Wells improved |
|------------|------|----------------|
| 0.1 | 3.94m | 71/100 |
| 0.5 | 3.98m | 69/100 |
| 1 | 3.64m | 68/100 |
| 2 | 3.45m | 67/100 |
| 3 | 3.46m | 64/100 |
| **5** | **3.37m** | 64/100 |
| 7 | 3.44m | 65/100 |
| 10 | 3.56m | 65/100 |
| 20 | 3.62m | 65/100 |

## Сравнение с baseline

- BF ±2° mse=0.1 (старый): 4.69m (15.4 ft)
- BF ±2.5° mse=5 (новый): **3.37m (11.1 ft)**
- Улучшение: **28%**
