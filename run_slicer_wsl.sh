#!/bin/bash
# Run GPU slicer from WSL with proper CUDA libraries

# Set NVIDIA library paths
export NVIDIA_BASE=/home/vano/miniconda3/lib/python3.13/site-packages/nvidia
export LD_LIBRARY_PATH=$NVIDIA_BASE/cudnn/lib:$NVIDIA_BASE/cublas/lib:$NVIDIA_BASE/cufft/lib:$NVIDIA_BASE/curand/lib:$NVIDIA_BASE/cusolver/lib:$NVIDIA_BASE/cusparse/lib:$NVIDIA_BASE/cusparselt/lib:$NVIDIA_BASE/cuda_runtime/lib:$NVIDIA_BASE/nvjitlink/lib:$NVIDIA_BASE/nccl/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Add nvidia-smi to PATH
export PATH=$PATH:/usr/lib/wsl/lib

# StarSteer directory (Windows path accessible from WSL)
export STARSTEER_DIR=/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de

# Run slicer
cd /mnt/e/Projects/Rogii/gpu_ag
python3 slicer_gpu.py --de --starsteer-dir "$STARSTEER_DIR" --max-iterations 1
