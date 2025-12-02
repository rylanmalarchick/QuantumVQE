#!/bin/bash
#PBS -N multi_node_smoke
#PBS -l nodes=4:ppn=4:gpus=1
#PBS -l walltime=00:30:00
#PBS -o logs/multi_node_smoke_output.log
#PBS -e logs/multi_node_smoke_error.log

# Multi-node GPU smoke test
# Requests 4 nodes, each with 1 GPU

cd $PBS_O_WORKDIR

# Create logs directory if needed
mkdir -p logs

# Load CUDA module
module load cuda/12.4.0-gcc-13.2.0-ym45qpm

# Activate conda environment
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
cat $PBS_NODEFILE
echo ""
echo "Python: $(which python)"
echo "========================================"

# Run across all 4 nodes (1 rank per node, using pernode to get 1 process per node)
mpirun --map-by ppr:1:node python src/scaling_study/multi_node_smoke_test.py

echo "========================================"
echo "Smoke test complete"
echo "End time: $(date)"
echo "========================================"
