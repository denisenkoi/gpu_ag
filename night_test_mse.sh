#!/bin/bash
# Ночной тест BF ±2.5° с разными mse_weight
# GPU0: 0.1, 1, 3, 7, 20
# GPU1: 0.5, 2, 5, 10, 15

cd /mnt/e/Projects/Rogii/gpu_ag
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

GPU=$1
shift
MSE_WEIGHTS="$@"

for MSE in $MSE_WEIGHTS; do
    echo "=== Starting mse_weight=$MSE on GPU$GPU ==="
    CUDA_VISIBLE_DEVICES=$GPU python full_well_optimizer.py --all --algorithm BRUTEFORCE --angle-range 2.5 --mse-weight $MSE > /tmp/bf25_mse_${MSE}.log 2>&1
    echo "=== Finished mse_weight=$MSE ==="
done

echo "=== All tests on GPU$GPU completed ==="
