#!/bin/bash
#PBS -N multi_gpu_smoke
#PBS -l nodes=1:ppn=8:gpus=4
#PBS -l walltime=00:30:00
#PBS -q shortq
#PBS -o logs/multi_gpu_smoke_output.log
#PBS -e logs/multi_gpu_smoke_error.log

# Multi-GPU smoke test (single node, 4 GPUs)
# Tests MPI parallelism with multiple GPUs on one node

cd $PBS_O_WORKDIR

# Create logs directory if needed
mkdir -p logs

# Activate conda environment (vqe-gpu has mpi4py)
eval "$(/apps/spack/opt/spack/linux-rocky8-zen4/gcc-13.2.0/anaconda3-2023.09-0-3wl2qheo6tntdwtbjdmvouw24zd4rugj/bin/conda shell.bash hook)"
conda activate vqe-gpu

echo "========================================"
echo "Multi-GPU Smoke Test (Single Node)"
echo "========================================"
echo "Job ID: $PBS_JOBID"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Host: $(hostname)"
echo ""
echo "Python: $(which python)"
echo "mpiexec: $(which mpiexec 2>/dev/null || echo 'not found')"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "========================================"

# Run 4 MPI ranks, one per GPU
mpiexec -n 4 python src/scaling_study/multi_node_smoke_test.py

echo "========================================"
echo "Smoke test complete"
echo "End time: $(date)"
echo "========================================"
