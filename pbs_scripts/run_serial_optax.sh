#!/bin/bash
#PBS -N vqe_serial_optax
#PBS -l nodes=1:ppn=1
#PBS -l walltime=02:00:00
#PBS -o logs/serial_optax_output.log
#PBS -e logs/serial_optax_error.log
#PBS -m abe
#PBS -M rylan@example.com

# Serial Optax+JIT VQE (CPU-only, No MPI, No GPU)
# This benchmark isolates the optimizer+JIT effect from parallelization
# Compare to:
#   - main.py (Serial PennyLane Adam): ~593.95s
#   - vqe_qjit.py on GPU: ~171.79s
#   - vqe_mpi.py: ~5-8s

cd $PBS_O_WORKDIR

# Activate conda environment
eval "$(/apps/spack/opt/spack/linux-rocky8-zen4/gcc-13.2.0/anaconda3-2023.09-0-3wl2qheo6tntdwtbjdmvouw24zd4rugj/bin/conda shell.bash hook)"
conda activate vqe-gpu

echo "============================================"
echo "Serial Optax+JIT VQE Run (CPU-only)"
echo "============================================"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "============================================"

# CRITICAL: Force CPU-only execution (no GPU)
# This is also set in the Python code, but we reinforce it here
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

# Verify no GPU access
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "JAX_PLATFORMS: $JAX_PLATFORMS"
echo "============================================"

# Run the serial Optax+JIT code
python src/vqe_serial_optax.py

echo "============================================"
echo "End time: $(date)"
echo "============================================"
