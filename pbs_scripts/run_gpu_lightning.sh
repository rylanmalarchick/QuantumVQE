#!/bin/bash
#PBS -N vqe_gpu_lightning
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=02:00:00
#PBS -o logs/gpu_lightning_output.log
#PBS -e logs/gpu_lightning_error.log
#PBS -m abe
#PBS -M rylan@example.com

# TRUE GPU-Accelerated VQE using lightning.gpu
# This uses PennyLane's GPU-accelerated quantum simulator
# Compare to:
#   - main.py (Serial PennyLane Adam): ~593.95s
#   - vqe_serial_optax.py (Serial Optax+JIT CPU): TBD
#   - vqe_qjit.py (Optax+JIT CPU, lightning.qubit): ~171.79s
#   - vqe_mpi.py: ~5-8s

cd $PBS_O_WORKDIR

# Activate conda environment
eval "$(/apps/spack/opt/spack/linux-rocky8-zen4/gcc-13.2.0/anaconda3-2023.09-0-3wl2qheo6tntdwtbjdmvouw24zd4rugj/bin/conda shell.bash hook)"
conda activate vqe-gpu

echo "============================================"
echo "TRUE GPU VQE Run (lightning.gpu)"
echo "============================================"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "============================================"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi
echo "============================================"

# Enable GPU access for JAX and CUDA
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Verify GPU is accessible
python -c "import jax; print('JAX devices:', jax.devices()); print('Default backend:', jax.default_backend())"

echo "============================================"
echo "Running vqe_gpu.py (lightning.gpu)"
echo "============================================"

# Run the TRUE GPU code (lightning.gpu device)
python src/vqe_gpu.py

echo "============================================"
echo "End time: $(date)"
echo "============================================"
