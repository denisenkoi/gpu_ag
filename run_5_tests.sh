#!/bin/bash
# 5 прогонов для проверки повторяемости

cd /mnt/e/Projects/Rogii/gpu_ag
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

RESULTS_DIR="results_repeatability_test"
mkdir -p $RESULTS_DIR

for i in 1 2 3 4 5; do
    echo "=========================================="
    echo "Run $i of 5"
    echo "=========================================="

    # Удаляем старый CSV
    rm -f results_wells_grid_15/optimization_statistics.csv

    # Запускаем тест
    python cpu_baseline/slicer.py --de \
        --starsteer-dir "/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de" \
        --max-iterations 0

    # Копируем результаты
    cp results_wells_grid_15/optimization_statistics.csv "$RESULTS_DIR/run_${i}_optimization.csv"
    cp results/slicer_quality_analysis_*.csv "$RESULTS_DIR/run_${i}_quality.csv" 2>/dev/null

    # Копируем интерпретацию (последняя по времени)
    LATEST_INTERP=$(ls -t temp_work/executor_*/slicing_well.json 2>/dev/null | head -1)
    if [ -n "$LATEST_INTERP" ]; then
        cp "$LATEST_INTERP" "$RESULTS_DIR/run_${i}_interpretation.json"
    fi

    echo "Run $i completed, results saved to $RESULTS_DIR/run_${i}_*.csv"
    echo ""
done

echo "All 5 runs completed!"
echo "Results in: $RESULTS_DIR/"
