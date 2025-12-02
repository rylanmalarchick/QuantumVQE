#!/bin/bash
#PBS -N multi_node_smoke
#PBS -l select=4:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=00:10:00
#PBS -q gpu
#PBS -o results/multi_node_smoke.out
#PBS -e results/multi_node_smoke.err

# Multi-node GPU smoke test
# Requests 4 nodes, each with 1 GPU

cd $PBS_O_WORKDIR
cd ~/QuantumVQE

# Load modules
module load cuda/12.6
module load gcc/12.2.0

# Activate environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vqe-gpu

echo "========================================"
echo "Multi-Node GPU Smoke Test"
echo "========================================"
echo "Job ID: $PBS_JOBID"
echo "Nodes allocated:"
cat $PBS_NODEFILE
echo "========================================"

# Run across all 4 nodes (1 rank per node)
mpirun -np 4 --hostfile $PBS_NODEFILE python src/scaling_study/multi_node_smoke_test.py

echo "========================================"
echo "Smoke test complete"
echo "========================================"
