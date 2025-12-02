#!/bin/bash
#PBS -N multi_node_smoke
#PBS -l nodes=2:ppn=4:gpus=1
#PBS -l walltime=00:30:00
#PBS -q shortq
#PBS -o logs/multi_node_smoke_output.log
#PBS -e logs/multi_node_smoke_error.log

# Multi-node GPU smoke test
# Requests 2 nodes (shortq max), each with 1 GPU

cd $PBS_O_WORKDIR

# Create logs directory if needed
mkdir -p logs

# Activate conda environment (has mpi4py and pennylane-lightning-gpu)
eval "$(/apps/spack/opt/spack/linux-rocky8-zen4/gcc-13.2.0/anaconda3-2023.09-0-3wl2qheo6tntdwtbjdmvouw24zd4rugj/bin/conda shell.bash hook)"
conda activate vqe-lightning-gpu

echo "========================================"
echo "Multi-Node GPU Smoke Test"
echo "========================================"
echo "Job ID: $PBS_JOBID"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""
echo "Nodes allocated:"
cat $PBS_NODEFILE | sort -u
echo ""
echo "Python: $(which python)"
echo ""
echo "GPU Info (on head node):"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "nvidia-smi not available on login node"
echo "========================================"

# Run 1 process per node (2 nodes = 2 ranks)
mpiexec -n 2 --map-by ppr:1:node python src/scaling_study/multi_node_smoke_test.py

echo "========================================"
echo "Smoke test complete"
echo "End time: $(date)"
echo "========================================"
