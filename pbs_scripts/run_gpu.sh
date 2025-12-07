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

# Activate conda environment using conda's activation
# Initialize conda for this shell session
eval "$(/apps/spack/opt/spack/linux-rocky8-zen4/gcc-13.2.0/anaconda3-2023.09-0-3wl2qheo6tntdwtbjdmvouw24zd4rugj/bin/conda shell.bash hook)"
conda activate vqe-gpu

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
python src/main_study/vqe_qjit.py

echo "============================================"
echo "End time: $(date)"
echo "============================================"
