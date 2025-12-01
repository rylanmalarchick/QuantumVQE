#!/bin/bash
#PBS -N vqe_gpu
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=02:00:00
#PBS -o logs/gpu_output.log
#PBS -e logs/gpu_error.log
#PBS -m abe
#PBS -M rylan@example.com

# JIT-compiled VQE with GPU acceleration

cd $PBS_O_WORKDIR

# Source bashrc to get micromamba setup
source ~/.bashrc
micromamba activate vqe-gpu

echo "============================================"
echo "JIT+GPU VQE Run"
echo "============================================"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "============================================"

# Set JAX to use GPU
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run the JIT+GPU code
python src/vqe_qjit.py

echo "============================================"
echo "End time: $(date)"
echo "============================================"
