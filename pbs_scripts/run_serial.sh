#!/bin/bash
#PBS -N vqe_serial
#PBS -l nodes=1:ppn=1
#PBS -l walltime=02:00:00
#PBS -o logs/serial_output.log
#PBS -e logs/serial_error.log
#PBS -m abe
#PBS -M rylan@example.com

# Serial VQE Baseline - Single CPU core

cd $PBS_O_WORKDIR

# Activate conda environment using conda's activation
# Initialize conda for this shell session
eval "$(/apps/spack/opt/spack/linux-rocky8-zen4/gcc-13.2.0/anaconda3-2023.09-0-3wl2qheo6tntdwtbjdmvouw24zd4rugj/bin/conda shell.bash hook)"
conda activate vqe-gpu

echo "============================================"
echo "Serial VQE Baseline Run"
echo "============================================"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "============================================"

# Run the serial code
python src/main.py

echo "============================================"
echo "End time: $(date)"
echo "============================================"
