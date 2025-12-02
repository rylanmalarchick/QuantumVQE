#!/bin/bash
#PBS -N vqe_mpi
#PBS -l nodes=NODE_COUNT:ppn=PPN_COUNT
#PBS -l walltime=02:00:00
#PBS -o logs/mpi_NPROCS_output.log
#PBS -e logs/mpi_NPROCS_error.log
#PBS -m abe
#PBS -M rylan@example.com

# MPI Parallel VQE
# This is a template - use submit_mpi.sh to generate specific scripts

cd $PBS_O_WORKDIR

# Activate conda environment using conda's activation
# Initialize conda for this shell session
eval "$(/apps/spack/opt/spack/linux-rocky8-zen4/gcc-13.2.0/anaconda3-2023.09-0-3wl2qheo6tntdwtbjdmvouw24zd4rugj/bin/conda shell.bash hook)"
conda activate vqe-gpu

echo "============================================"
echo "MPI VQE Run with NPROCS processes"
echo "============================================"
echo "Start time: $(date)"
echo "Running on nodes: $(cat $PBS_NODEFILE | sort -u)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "MPI ranks distribution:"
cat $PBS_NODEFILE
echo "============================================"

# Run MPI code
mpiexec -n NPROCS python src/main_study/vqe_mpi.py

echo "============================================"
echo "End time: $(date)"
echo "============================================"
